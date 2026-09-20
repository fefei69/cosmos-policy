"""Section 7 evaluation of a hanoi_dense_v5 export on one H100 or H200.

Batched inference over validation (or, once selection is locked, test) rows:
per-slot XYZ error, chunk endpoint error, jaw accuracy and flip timing, value
error and optionally decoded future-frame L1/PSNR, each reported for all rows
and split by observation motion (stationary = finite-difference speed under
2 mm/s, the flag stored in the archive). The optional parity check runs the
serving adapter (single observation, seed 1) against the evaluator's own
single-observation path with the same seed and compares first-slot outputs.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from cosmos_policy.datasets.hanoi_dense_data import FRAMESKIP, HORIZON, STATIONARY_SPEED_M_PER_S

ACCEPTED_GPUS = ('H100', 'H200')
HANOI_UNDO_INJECTION = [0, 1, 3, 4]  # blank, proprio, action, future proprio: as the hanoi suite decodes
FUTURE_FRAME_INDEX = (5 - 1) * 4 + 1  # latent slot 5 -> raw frame 17 of the 25-frame packing


def select_rows(dataset, stride, seed=195):
    """Every ``stride``-th archive row per episode with a random phase, so all skip phases occur."""
    episodes = dataset.arrays['episode_indices']
    rng = np.random.default_rng(seed)
    chosen, phases = [], {}
    for ep in np.unique(episodes):
        rows = np.flatnonzero(episodes == ep)
        phase = int(rng.integers(stride)) if stride > 1 else 0
        phases[int(ep)] = phase
        chosen.append(rows[phase::stride])
    return np.sort(np.concatenate(chosen)), phases


def flip_slot(jaw_sequence, current):
    """First chunk slot whose jaw intent differs from the current intent, or -1."""
    changed = np.flatnonzero(jaw_sequence != current)
    return int(changed[0]) if len(changed) else -1


def per_sample_metrics(predicted, target, pad, current_jaw):
    """predicted/target (16, 4) absolute; pad (16,) bool; current_jaw 0/1 at the observation row."""
    valid = ~pad
    if not valid.any():
        return None  # Observation within the last three rows of its episode: no unpadded slot to score.
    errors = np.linalg.norm(predicted[:, :3] - target[:, :3], axis=1) * 1000
    last = int(np.flatnonzero(valid)[-1])
    jaw_correct = predicted[:, 3] == target[:, 3]
    true_flip, predicted_flip = flip_slot(target[valid, 3], current_jaw), flip_slot(predicted[valid, 3], current_jaw)
    return {'errors_mm': errors, 'valid': valid, 'slot1_mm': float(errors[0]), 'endpoint_mm': float(errors[last]),
            'mean_valid_mm': float(errors[valid].mean()), 'jaw_correct': jaw_correct,
            'jaw_accuracy': float(jaw_correct[valid].mean()), 'true_flip_slot': true_flip, 'predicted_flip_slot': predicted_flip}


def summarize(samples, tolerance_mm=2.0):
    if not samples:
        return {'samples': 0}
    errors = np.stack([s['errors_mm'] for s in samples])
    valid = np.stack([s['valid'] for s in samples])
    jaw = np.stack([s['jaw_correct'] for s in samples])
    targets = np.stack([s['target_jaw'] for s in samples])
    per_slot_mean = [float(errors[valid[:, j], j].mean()) if valid[:, j].any() else None for j in range(HORIZON)]
    per_slot_jaw = [float(jaw[valid[:, j], j].mean()) if valid[:, j].any() else None for j in range(HORIZON)]
    slot1 = errors[:, 0]
    endpoint = np.array([s['endpoint_mm'] for s in samples])
    truth, correct = targets[valid].astype(bool), jaw[valid].astype(bool)  # jaw holds per-slot correctness
    tp, tn = int((truth & correct).sum()), int((~truth & correct).sum())
    fp, fn = int((~truth & ~correct).sum()), int((truth & ~correct).sum())
    balanced = ((tp / (tp + fn) if tp + fn else 0.0) + (tn / (tn + fp) if tn + fp else 0.0)) / 2
    flips = [s for s in samples if s['true_flip_slot'] >= 0]
    timed = [s for s in flips if s['predicted_flip_slot'] >= 0]
    timing = np.array([FRAMESKIP * (s['predicted_flip_slot'] - s['true_flip_slot']) for s in timed], np.float64)
    spurious = sum(1 for s in samples if s['true_flip_slot'] < 0 and s['predicted_flip_slot'] >= 0)
    result = {
        'samples': len(samples),
        'xyz_mm': {'mean_valid_slots': float(errors[valid].mean()), 'p95_valid_slots': float(np.percentile(errors[valid], 95)),
                   'per_slot_mean': per_slot_mean,
                   'slot1_mean': float(slot1.mean()), 'slot1_median': float(np.median(slot1)), 'slot1_p95': float(np.percentile(slot1, 95)),
                   'slot1_within_tolerance_fraction': float((slot1 <= tolerance_mm).mean()), 'tolerance_mm': tolerance_mm,
                   'endpoint_mean': float(endpoint.mean()), 'endpoint_p95': float(np.percentile(endpoint, 95))},
        'jaw': {'accuracy_valid_slots': float(jaw[valid].mean()), 'balanced_accuracy_valid_slots': float(balanced),
                'per_slot_accuracy': per_slot_jaw, 'slot1_accuracy': float(jaw[:, 0].mean()),
                'chunks_with_true_flip': len(flips), 'flips_predicted': len(timed), 'flips_missed': len(flips) - len(timed),
                'spurious_flips': int(spurious),
                'flip_timing_rows_mean_abs': float(np.abs(timing).mean()) if len(timing) else None,
                'flip_timing_rows_median': float(np.median(timing)) if len(timing) else None,
                'flip_timing_rows_p95_abs': float(np.percentile(np.abs(timing), 95)) if len(timing) else None},
    }
    for key in ('value_abs_error', 'future_l1', 'future_psnr_db'):
        values = [s[key] for s in samples if s.get(key) is not None]
        if values:
            result[key] = {'mean': float(np.mean(values)), 'median': float(np.median(values)), 'samples': len(values)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings.pkl')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--stride', type=int, default=3, help='Evaluate every stride-th row per episode, random phase')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=5, help='Denoising steps for the primary pass')
    parser.add_argument('--also-steps', type=int, default=0, help='Optional second pass with this many steps (actions only)')
    parser.add_argument('--future', action='store_true', help='Decode the future frame and report L1/PSNR')
    parser.add_argument('--parity-samples', type=int, default=0)
    parser.add_argument('--parity-tolerance-mm', type=float, default=0.5)
    parser.add_argument('--selection', type=Path, help='Locked validation choice required for test evaluation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite an evaluation: {args.output}')
    if args.split == 'test':
        if args.selection is None:
            parser.error('Test evaluation requires --selection from completed validation-only selection')
        selection = json.loads(args.selection.read_text())
        if Path(selection['checkpoint']).resolve() != args.checkpoint.resolve():
            raise ValueError('Test checkpoint differs from the locked validation choice')
    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_dense'
    import torch
    from torch.utils.data import DataLoader, Subset
    from cosmos_policy.datasets.hanoi_dense_dataset import HanoiDenseDataset
    from cosmos_policy.experiments.robot.cosmos_utils import (
        extract_action_chunk_from_latent_sequence, extract_value_from_latent_sequence, undo_latent_injection, unnormalize_actions)
    from cosmos_policy.experiments.robot.hanoi.dense_policy import HanoiDenseInferenceConfig, load_dense_policy, predict_dense_actions, threshold_jaw

    gpu = torch.cuda.get_device_name() if torch.cuda.device_count() else 'none'
    if torch.cuda.device_count() != 1 or not any(tag in gpu for tag in ACCEPTED_GPUS):
        raise RuntimeError(f'Evaluate on one H100 or H200, not {gpu!r} x{torch.cuda.device_count()}')
    cfg = HanoiDenseInferenceConfig(str(args.checkpoint), str(args.metadata / 'dataset_statistics.json'), args.embeddings,
                                    num_denoising_steps_action=args.steps)
    model, stats, _, identity = load_dense_policy(cfg)
    dataset = HanoiDenseDataset(str(args.metadata), args.embeddings, split=args.split, representative_order=False)
    selected, phases = select_rows(dataset, args.stride)
    arrays = dataset.arrays
    raw_rows = arrays['source_observation_indices'][selected]
    current_jaw = dataset._file()['action_abs'][:, 3][raw_rows]  # jaw intent at the observation row, audit input only
    report = {'checkpoint': str(args.checkpoint.resolve()), 'split': args.split, 'contract': identity['contract'],
              'metadata': str(args.metadata.resolve()), 'gpu': gpu, 'torch': torch.__version__,
              'stride': args.stride, 'phases': phases, 'num_samples': int(len(selected)), 'batch_size': args.batch_size,
              'denoising_steps': args.steps, 'stationary_definition': f'finite-difference measured speed under {STATIONARY_SPEED_M_PER_S} m/s',
              'noise': 'seed 195 plus batch index per batch; parity uses seed 1 on single observations', 'started_at': time.time()}

    def batched_pass(steps, decode_future):
        loader = DataLoader(Subset(dataset, selected.tolist()), batch_size=args.batch_size, num_workers=4, pin_memory=True)
        samples, offset = [], 0
        skipped_all_padded = []
        with torch.no_grad():
            for b, batch in enumerate(loader):
                n = int(batch['video'].shape[0])
                data_batch = {
                    'dataset_name': 'video_data',
                    'video': batch['video'].to(dtype=torch.uint8).cuda(),
                    't5_text_embeddings': batch['t5_text_embeddings'].to(dtype=torch.bfloat16).cuda(),
                    'fps': torch.full((n,), 16, dtype=torch.bfloat16).cuda(),
                    'padding_mask': torch.zeros((n, 1, 224, 224), dtype=torch.bfloat16).cuda(),
                    'num_conditional_frames': model.config.min_num_conditional_frames,
                    'proprio': batch['proprio'].to(dtype=torch.bfloat16).cuda(),
                }
                for key in ('current_proprio_latent_idx', 'current_wrist_image_latent_idx', 'current_wrist_image2_latent_idx',
                            'current_image_latent_idx', 'current_image2_latent_idx', 'action_latent_idx',
                            'future_proprio_latent_idx', 'future_wrist_image_latent_idx', 'future_wrist_image2_latent_idx',
                            'future_image_latent_idx', 'future_image2_latent_idx', 'value_latent_idx'):
                    data_batch[key] = batch[key].to(dtype=torch.int64).cuda()
                latent, clean = model.generate_samples_from_batch(
                    data_batch, n_sample=n, num_steps=steps, seed=195 + b, is_negative_prompt=False,
                    use_variance_scale=False, return_orig_clean_latent_frames=True)
                actions = extract_action_chunk_from_latent_sequence(latent, (HORIZON, 4), data_batch['action_latent_idx']).float().cpu().numpy()
                actions = unnormalize_actions(actions, stats)
                values = extract_value_from_latent_sequence(latent, data_batch['value_latent_idx']).float().cpu().numpy()
                future_l1 = future_psnr = None
                if decode_future:
                    restored = undo_latent_injection(latent.clone(), clean, HANOI_UNDO_INJECTION)
                    decoded = ((model.decode(restored) + 1.0) * 127.5).clamp(0, 255)  # (B, C, T, H, W)
                    predicted_frames = decoded[:, :, FUTURE_FRAME_INDEX].permute(0, 2, 3, 1).float().cpu().numpy()
                    true_frames = batch['video'][:, :, FUTURE_FRAME_INDEX].permute(0, 2, 3, 1).float().numpy()
                    diff = predicted_frames - true_frames
                    future_l1 = np.abs(diff).reshape(n, -1).mean(1) / 255.0
                    mse = (diff ** 2).reshape(n, -1).mean(1)
                    future_psnr = 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-6))
                for k in range(n):
                    i = int(batch['__key__'][k])
                    predicted = threshold_jaw(actions[k])
                    target = arrays['actions'][i]
                    metric = per_sample_metrics(predicted, target, arrays['actions_is_pad'][i], float(current_jaw[offset + k]))
                    if metric is None:
                        skipped_all_padded.append(i)
                        continue
                    metric.update({'archive_index': i, 'episode': int(arrays['episode_indices'][i]),
                                   'row': int(arrays['source_observation_indices'][i]), 'stationary': bool(arrays['stationary'][i]),
                                   'target_jaw': target[:, 3], 'predicted_slot1': predicted[0].tolist(),
                                   'value_abs_error': float(abs(values[k] - float(batch['value_function_return'][k]))),
                                   'future_l1': None if future_l1 is None else float(future_l1[k]),
                                   'future_psnr_db': None if future_psnr is None else float(future_psnr[k])})
                    samples.append(metric)
                offset += n
                if (b + 1) % 50 == 0:
                    print(f'batch {b + 1}/{len(loader)}', flush=True)
        report.setdefault('all_padded_chunks_skipped', len(skipped_all_padded))
        return samples

    def tables(samples):
        return {'all': summarize(samples),
                'stationary': summarize([s for s in samples if s['stationary']]),
                'moving': summarize([s for s in samples if not s['stationary']])}

    primary = batched_pass(args.steps, args.future)
    report['metrics'] = tables(primary)
    report['per_sample'] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()
                             if k not in ('valid', 'jaw_correct', 'target_jaw')} for s in primary]
    if args.also_steps:
        report[f'metrics_{args.also_steps}_steps'] = tables(batched_pass(args.also_steps, False))
    if args.parity_samples:
        # Spread the parity subset across episodes: every k-th selected row.
        k = max(1, len(selected) // args.parity_samples)
        parity_rows = selected[::k][:args.parity_samples]
        differences, batched_differences = [], []
        by_index = {s['archive_index']: s for s in primary}
        with torch.no_grad():
            for i in parity_rows:
                if int(i) not in by_index:
                    continue  # all-padded chunk, not scored
                sample = dataset.raw_example(int(i))
                served = predict_dense_actions(cfg, model, stats, sample['image'], sample['state'], seed=1)
                item = dataset[int(i)]
                single = {
                    'dataset_name': 'video_data', 'video': item['video'][None].to(dtype=torch.uint8).cuda(),
                    't5_text_embeddings': item['t5_text_embeddings'][None].to(dtype=torch.bfloat16).cuda(),
                    'fps': torch.full((1,), 16, dtype=torch.bfloat16).cuda(),
                    'padding_mask': torch.zeros((1, 1, 224, 224), dtype=torch.bfloat16).cuda(),
                    'num_conditional_frames': model.config.min_num_conditional_frames,
                    'proprio': torch.as_tensor(item['proprio'])[None].to(dtype=torch.bfloat16).cuda(),
                }
                for key in ('current_proprio_latent_idx', 'current_wrist_image_latent_idx', 'current_wrist_image2_latent_idx',
                            'current_image_latent_idx', 'current_image2_latent_idx', 'action_latent_idx',
                            'future_proprio_latent_idx', 'future_wrist_image_latent_idx', 'future_wrist_image2_latent_idx',
                            'future_image_latent_idx', 'future_image2_latent_idx', 'value_latent_idx'):
                    single[key] = torch.tensor([int(item[key])], dtype=torch.int64).cuda()
                latent, _ = model.generate_samples_from_batch(single, n_sample=1, num_steps=args.steps, seed=1,
                                                              is_negative_prompt=False, use_variance_scale=False,
                                                              return_orig_clean_latent_frames=True)
                evaluated = threshold_jaw(unnormalize_actions(
                    extract_action_chunk_from_latent_sequence(latent, (HORIZON, 4), single['action_latent_idx']).float().cpu().numpy(), stats)[0])
                differences.append(float(np.linalg.norm(served[0, :3] - evaluated[0, :3]) * 1000))
                batched_differences.append(float(np.linalg.norm(served[0, :3] - np.asarray(by_index[int(i)]['predicted_slot1'])[:3]) * 1000))
        report['serving_parity'] = {
            'samples': len(parity_rows), 'tolerance_mm': args.parity_tolerance_mm,
            'max_first_slot_difference_mm': float(max(differences)), 'mean_first_slot_difference_mm': float(np.mean(differences)),
            'passed': bool(max(differences) <= args.parity_tolerance_mm),
            'batched_vs_single_noise_first_slot_mm': {'mean': float(np.mean(batched_differences)), 'max': float(max(batched_differences))},
        }
        report['serving_parity_passed'] = report['serving_parity']['passed']
    dataset.close()
    report['finished_at'] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    summary = {k: v for k, v in report.items() if k not in ('per_sample', 'phases')}
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
