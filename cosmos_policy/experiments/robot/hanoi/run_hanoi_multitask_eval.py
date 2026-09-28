"""Section 7 evaluation of a hanoi_multitask_v6 export on one H100 or H200, plus a language-following probe.

Batched inference over validation (or, once selection is locked, test) rows
with each row's own task prompt: the dense metrics (per-slot XYZ error,
endpoint, jaw accuracy and flip timing, value error, optional future-frame
decode) for all rows, split by motion and split by task. The optional parity
check runs the serving adapter (single observation, its prompt, seed 1)
against the evaluator's single-observation path.

``--probe-rows`` adds the language-following probe, which is what an
instruction-ignoring policy would fail while still scoring well on the metrics
above (every board on a demonstration belongs to two tasks):

* same-start decision test: for each pair of tasks that start from the same
  stack, observations of one task during its first move (lift and transit,
  where image and state are identical across the pair) are predicted under
  both prompts; each prediction is scored against the label of the task whose
  prompt was used, at that task's nearest recorded state (matched-state
  relabelling, states within 5 mm). Only rows where the two labels differ by
  more than 5 mm count as decisions. Chance is 50%.
* prompt-swap sensitivity: moving rows predicted under their own prompt and
  under the reverse task's prompt; the displacement between the two
  predictions is compared with the sampling-noise floor. Sensitivity only; a
  reversed prompt has no unique correct chunk from an arbitrary mid-episode
  arm pose.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from cosmos_policy.datasets.hanoi_dense_data import STATIONARY_SPEED_M_PER_S
from cosmos_policy.datasets.hanoi_multitask_data import REVERSE_OF, SAME_START_PAIRS, TASKS
from cosmos_policy.experiments.robot.hanoi.run_hanoi_dense_eval import (
    ACCEPTED_GPUS, FUTURE_FRAME_INDEX, HANOI_UNDO_INJECTION, per_sample_metrics, select_rows, summarize,
)

LATENT_KEYS = ('current_proprio_latent_idx', 'current_wrist_image_latent_idx', 'current_wrist_image2_latent_idx',
               'current_image_latent_idx', 'current_image2_latent_idx', 'action_latent_idx',
               'future_proprio_latent_idx', 'future_wrist_image_latent_idx', 'future_wrist_image2_latent_idx',
               'future_image_latent_idx', 'future_image2_latent_idx', 'value_latent_idx')
FIRST_MOVE_STAGES = (5, 6, 7)  # lift, transit, insert_target (motion_stage ids of the recordings)
MATCH_MM = 5.0
DECISION_MM = 5.0


def first_move_rows(dataset, task_index):
    """Archive indices of this task's rows during the first move while the ring is held (lift, transit, insertion)."""
    arrays = dataset.arrays
    candidates = np.flatnonzero(arrays['task_indices'] == task_index)
    if not len(candidates):
        return candidates
    handle = dataset._file(task_index)
    rows = arrays['source_observation_indices'][candidates]
    order = np.argsort(rows)
    sorted_rows = rows[order]
    move = handle['move_idx'][:][sorted_rows]
    held = handle['held_disk'][:][sorted_rows]
    stage = handle['motion_stage'][:][sorted_rows]
    keep = (move == 0) & (held == 1) & np.isin(stage, FIRST_MOVE_STAGES)
    return candidates[order][keep]


def matched_state_pairs(dataset, task_a, task_b, match_mm=MATCH_MM, decision_mm=DECISION_MM):
    """(row of A, nearest row of B) during the first move with states within match_mm and labels differing by more than decision_mm."""
    arrays = dataset.arrays
    rows_a, rows_b = first_move_rows(dataset, task_a), first_move_rows(dataset, task_b)
    if not len(rows_a) or not len(rows_b):
        return []
    pos_a, pos_b = arrays['cartesian_positions'][rows_a], arrays['cartesian_positions'][rows_b]
    pairs = []
    for k, a in enumerate(rows_a):
        d = np.linalg.norm(pos_b - pos_a[k], axis=1) * 1000
        j = int(np.argmin(d))
        if d[j] > match_mm:
            continue
        b = int(rows_b[j])
        valid = ~(arrays['actions_is_pad'][a] | arrays['actions_is_pad'][b])
        if not valid.any():
            continue
        gap = float(np.linalg.norm(arrays['actions'][a][valid, :3] - arrays['actions'][b][valid, :3], axis=1).mean() * 1000)
        if gap > decision_mm:
            pairs.append((int(a), b, gap))
    return pairs


def chunk_error_mm(predicted, target, pad):
    valid = ~pad
    return float(np.linalg.norm(predicted[valid, :3] - target[valid, :3], axis=1).mean() * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings_multitask.pkl')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--stride', type=int, default=3, help='Evaluate every stride-th row per episode, random phase')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=5, help='Denoising steps for the primary pass')
    parser.add_argument('--also-steps', type=int, default=0, help='Optional second pass with this many steps (actions only)')
    parser.add_argument('--future', action='store_true', help='Decode the future frame and report L1/PSNR')
    parser.add_argument('--parity-samples', type=int, default=0)
    parser.add_argument('--parity-tolerance-mm', type=float, default=0.5)
    parser.add_argument('--probe-rows', type=int, default=0, help='Language probe: decision rows per same-start pair and swap rows')
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
    horizon = int(json.loads((args.metadata / 'metadata.json').read_text())['horizon'])
    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_dense'
    os.environ.setdefault('HANOI_DENSE_HORIZON', str(horizon))
    import torch
    from torch.utils.data import DataLoader, Subset
    from cosmos_policy.datasets.hanoi_multitask_dataset import HanoiMultitaskDataset
    from cosmos_policy.experiments.robot.cosmos_utils import (
        extract_action_chunk_from_latent_sequence, extract_value_from_latent_sequence, undo_latent_injection, unnormalize_actions)
    from cosmos_policy.experiments.robot.hanoi.dense_policy import threshold_jaw
    from cosmos_policy.experiments.robot.hanoi.multitask_policy import (
        HanoiMultitaskInferenceConfig, load_multitask_policy, predict_multitask_actions)

    gpu = torch.cuda.get_device_name() if torch.cuda.device_count() else 'none'
    if torch.cuda.device_count() != 1 or not any(tag in gpu for tag in ACCEPTED_GPUS):
        raise RuntimeError(f'Evaluate on one H100 or H200, not {gpu!r} x{torch.cuda.device_count()}')
    cfg = HanoiMultitaskInferenceConfig(str(args.checkpoint), str(args.metadata / 'dataset_statistics.json'), args.embeddings,
                                        num_denoising_steps_action=args.steps, chunk_size=horizon)
    model, stats, _, identity = load_multitask_policy(cfg)
    dataset = HanoiMultitaskDataset(str(args.metadata), args.embeddings, split=args.split, representative_order=False)
    selected, phases = select_rows(dataset, args.stride)
    arrays = dataset.arrays
    current_jaw = np.zeros(len(selected), np.float32)  # jaw intent at the observation row, audit input only
    for task in TASKS:
        mask = arrays['task_indices'][selected] == task.index
        if mask.any():
            current_jaw[mask] = dataset._file(task.index)['action_abs'][:, 3][arrays['source_observation_indices'][selected[mask]]]
    report = {'checkpoint': str(args.checkpoint.resolve()), 'split': args.split, 'contract': identity['contract'], 'horizon': horizon,
              'metadata': str(args.metadata.resolve()), 'embeddings': str(Path(args.embeddings).resolve()), 'gpu': gpu, 'torch': torch.__version__,
              'stride': args.stride, 'phases': phases, 'num_samples': int(len(selected)), 'batch_size': args.batch_size,
              'denoising_steps': args.steps, 'stationary_definition': f'finite-difference measured speed under {STATIONARY_SPEED_M_PER_S} m/s',
              'tasks': [{'index': t.index, 'direction': t.direction} for t in TASKS],
              'noise': 'seed 195 plus batch index per batch; parity and probe use seed 1 on single observations', 'started_at': time.time()}

    def to_batch(batch, n):
        data_batch = {
            'dataset_name': 'video_data',
            'video': batch['video'].to(dtype=torch.uint8).cuda(),
            't5_text_embeddings': batch['t5_text_embeddings'].to(dtype=torch.bfloat16).cuda(),
            'fps': torch.full((n,), 16, dtype=torch.bfloat16).cuda(),
            'padding_mask': torch.zeros((n, 1, 224, 224), dtype=torch.bfloat16).cuda(),
            'num_conditional_frames': model.config.min_num_conditional_frames,
            'proprio': batch['proprio'].to(dtype=torch.bfloat16).cuda(),
        }
        for key in LATENT_KEYS:
            data_batch[key] = batch[key].to(dtype=torch.int64).cuda()
        return data_batch

    def batched_pass(steps, decode_future):
        loader = DataLoader(Subset(dataset, selected.tolist()), batch_size=args.batch_size, num_workers=4, pin_memory=True)
        samples, offset, skipped_all_padded = [], 0, []
        with torch.no_grad():
            for b, batch in enumerate(loader):
                n = int(batch['video'].shape[0])
                data_batch = to_batch(batch, n)
                latent, clean = model.generate_samples_from_batch(
                    data_batch, n_sample=n, num_steps=steps, seed=195 + b, is_negative_prompt=False,
                    use_variance_scale=False, return_orig_clean_latent_frames=True)
                actions = extract_action_chunk_from_latent_sequence(latent, (horizon, 4), data_batch['action_latent_idx']).float().cpu().numpy()
                actions = unnormalize_actions(actions, stats)
                values = extract_value_from_latent_sequence(latent, data_batch['value_latent_idx']).float().cpu().numpy()
                future_l1 = future_psnr = None
                if decode_future:
                    restored = undo_latent_injection(latent.clone(), clean, HANOI_UNDO_INJECTION)
                    decoded = ((model.decode(restored) + 1.0) * 127.5).clamp(0, 255)
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
                    metric.update({'archive_index': i, 'episode': int(arrays['episode_indices'][i]), 'task_index': int(arrays['task_indices'][i]),
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

    def task_tables(samples):
        return {t.direction: summarize([s for s in samples if s['task_index'] == t.index]) for t in TASKS}

    primary = batched_pass(args.steps, args.future)
    report['metrics'] = tables(primary)
    report['metrics_by_task'] = task_tables(primary)
    report['per_sample'] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()
                             if k not in ('valid', 'jaw_correct', 'target_jaw')} for s in primary]
    if args.also_steps:
        secondary = batched_pass(args.also_steps, False)
        report[f'metrics_{args.also_steps}_steps'] = tables(secondary)
        report[f'metrics_{args.also_steps}_steps_by_task'] = task_tables(secondary)

    def single_prediction(i, prompt, seed=1):
        sample = dataset.raw_example(int(i))
        return predict_multitask_actions(cfg, model, stats, sample['image'], sample['state'], prompt, seed=seed)

    if args.parity_samples:
        k = max(1, len(selected) // args.parity_samples)
        parity_rows = selected[::k][:args.parity_samples]
        differences, batched_differences = [], []
        by_index = {s['archive_index']: s for s in primary}
        with torch.no_grad():
            for i in parity_rows:
                if int(i) not in by_index:
                    continue  # all-padded chunk, not scored
                served = single_prediction(i, TASKS[int(arrays['task_indices'][i])].prompt)
                item = dataset[int(i)]
                single = {key: (item[key][None] if key in ('video', 't5_text_embeddings') else item[key]) for key in ('video', 't5_text_embeddings')}
                single = {
                    'dataset_name': 'video_data', 'video': single['video'].to(dtype=torch.uint8).cuda(),
                    't5_text_embeddings': single['t5_text_embeddings'].to(dtype=torch.bfloat16).cuda(),
                    'fps': torch.full((1,), 16, dtype=torch.bfloat16).cuda(),
                    'padding_mask': torch.zeros((1, 1, 224, 224), dtype=torch.bfloat16).cuda(),
                    'num_conditional_frames': model.config.min_num_conditional_frames,
                    'proprio': torch.as_tensor(item['proprio'])[None].to(dtype=torch.bfloat16).cuda(),
                }
                for key in LATENT_KEYS:
                    single[key] = torch.tensor([int(item[key])], dtype=torch.int64).cuda()
                latent, _ = model.generate_samples_from_batch(single, n_sample=1, num_steps=args.steps, seed=1,
                                                              is_negative_prompt=False, use_variance_scale=False,
                                                              return_orig_clean_latent_frames=True)
                evaluated = threshold_jaw(unnormalize_actions(
                    extract_action_chunk_from_latent_sequence(latent, (horizon, 4), single['action_latent_idx']).float().cpu().numpy(), stats)[0])
                differences.append(float(np.linalg.norm(served[0, :3] - evaluated[0, :3]) * 1000))
                batched_differences.append(float(np.linalg.norm(served[0, :3] - np.asarray(by_index[int(i)]['predicted_slot1'])[:3]) * 1000))
        report['serving_parity'] = {
            'samples': len(differences), 'tolerance_mm': args.parity_tolerance_mm,
            'max_first_slot_difference_mm': float(max(differences)), 'mean_first_slot_difference_mm': float(np.mean(differences)),
            'passed': bool(max(differences) <= args.parity_tolerance_mm),
            'batched_vs_single_noise_first_slot_mm': {'mean': float(np.mean(batched_differences)), 'max': float(max(batched_differences))},
        }
        report['serving_parity_passed'] = report['serving_parity']['passed']

    if args.probe_rows:
        probe = {'definition': __doc__.split('``--probe-rows``', 1)[1].strip(), 'match_mm': MATCH_MM, 'decision_mm': DECISION_MM, 'pairs': {}}
        rng = np.random.default_rng(1)
        with torch.no_grad():
            for a, b in SAME_START_PAIRS:
                for x, y in ((a, b), (b, a)):  # observations from x, scored under both prompts
                    pairs = matched_state_pairs(dataset, x, y)
                    if len(pairs) > args.probe_rows:
                        pairs = [pairs[j] for j in sorted(rng.choice(len(pairs), args.probe_rows, replace=False))]
                    correct_own, correct_swapped, swap_mm, own_mm, swapped_mm = [], [], [], [], []
                    for i, j, _ in pairs:
                        own = single_prediction(i, TASKS[x].prompt)
                        swapped = single_prediction(i, TASKS[y].prompt)
                        pad_i, pad_j = arrays['actions_is_pad'][i], arrays['actions_is_pad'][j]
                        label_x, label_y = arrays['actions'][i], arrays['actions'][j]
                        e_own_x, e_own_y = chunk_error_mm(own, label_x, pad_i), chunk_error_mm(own, label_y, pad_j)
                        e_sw_x, e_sw_y = chunk_error_mm(swapped, label_x, pad_i), chunk_error_mm(swapped, label_y, pad_j)
                        correct_own.append(e_own_x < e_own_y)
                        correct_swapped.append(e_sw_y < e_sw_x)
                        own_mm.append(e_own_x); swapped_mm.append(e_sw_y)
                        swap_mm.append(float(np.linalg.norm(own[:, :3] - swapped[:, :3], axis=1).mean() * 1000))
                    key = f'{TASKS[x].direction}_under_{TASKS[y].direction}'
                    probe['pairs'][key] = {
                        'decision_rows': len(pairs), 'candidate_rows': len(matched_state_pairs(dataset, x, y)),
                        'own_prompt_correct': float(np.mean(correct_own)) if pairs else None,
                        'swapped_prompt_correct': float(np.mean(correct_swapped)) if pairs else None,
                        'own_prompt_error_mm': float(np.mean(own_mm)) if pairs else None,
                        'swapped_prompt_error_vs_other_label_mm': float(np.mean(swapped_mm)) if pairs else None,
                        'swap_displacement_mm': float(np.mean(swap_mm)) if pairs else None,
                    }
            decisions = [v for v in probe['pairs'].values() if v['decision_rows']]
            probe['decision_accuracy'] = float(np.mean([v['own_prompt_correct'] for v in decisions] + [v['swapped_prompt_correct'] for v in decisions])) if decisions else None
            probe['decision_rows_total'] = int(sum(v['decision_rows'] for v in decisions))
            # Reverse-prompt sensitivity on moving rows, against the noise between two draws under the same prompt.
            moving = [s['archive_index'] for s in primary if not s['stationary']]
            chosen = [moving[j] for j in sorted(rng.choice(len(moving), min(args.probe_rows, len(moving)), replace=False))] if moving else []
            reverse_mm, noise_mm = [], []
            for i in chosen:
                task = int(arrays['task_indices'][i])
                own = single_prediction(i, TASKS[task].prompt, seed=1)
                reverse = single_prediction(i, TASKS[REVERSE_OF[task]].prompt, seed=1)
                redraw = single_prediction(i, TASKS[task].prompt, seed=2)
                reverse_mm.append(float(np.linalg.norm(own[:, :3] - reverse[:, :3], axis=1).mean() * 1000))
                noise_mm.append(float(np.linalg.norm(own[:, :3] - redraw[:, :3], axis=1).mean() * 1000))
            probe['reverse_prompt_sensitivity'] = {
                'rows': len(chosen),
                'swap_displacement_mm': {'mean': float(np.mean(reverse_mm)), 'median': float(np.median(reverse_mm))} if chosen else None,
                'same_prompt_redraw_mm': {'mean': float(np.mean(noise_mm)), 'median': float(np.median(noise_mm))} if chosen else None,
            }
        report['language_probe'] = probe
    dataset.close()
    report['finished_at'] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    summary = {k: v for k, v in report.items() if k not in ('per_sample', 'phases')}
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
