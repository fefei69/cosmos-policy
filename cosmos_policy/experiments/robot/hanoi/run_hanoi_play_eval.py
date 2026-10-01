"""Section 7 evaluation of a hanoi_play_k5 export on one H100 or H200, plus a goal-following probe.

Batched inference over validation (or, once selection is locked, test) rows
with each row's own goal sentence: the dense metrics (per-slot XYZ error,
endpoint, jaw accuracy and flip timing, value error, optional future-frame
decode) for all rows, split by motion, by how many moves ahead the goal lies,
by the goal's graph distance, and for the decision rows alone (approach_source
and transit stages, where the pick and the place are chosen). The optional
parity check runs the serving adapter (single observation, its sentence, seed
1) against the evaluator's single-observation path.

``--probe-rows`` adds the goal-following probe, which a goal-ignoring policy
would fail while still scoring well on the metrics above (most rows of a move
are the same whatever the goal):

* matched-state goal decision test: pairs of decision rows with the same
  current board and motion stage, measured positions within 5 mm, different
  goal boards whose labelled moves lead to different next boards, and labels
  differing by more than 5 mm over the shared valid slots; each row is
  predicted under both sentences and scored against the label of the row whose
  sentence was used. Chance is 50%.
* shuffled-goal sensitivity: decision rows predicted under their own sentence
  and under a random other board's sentence; the displacement between the two
  predictions is compared with the sampling-noise floor (a second draw under
  the own sentence). Sensitivity only.

The implied-move stitching proxy (goals at graph distance 1, 3, 7, 15) is not
implemented here; the composition test is the arm.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from cosmos_policy.datasets.hanoi_dense_data import STATIONARY_SPEED_M_PER_S
from cosmos_policy.datasets.hanoi_play_data import BOARDS, DECISION_STAGES, PROMPTS, STAGE_NAMES
from cosmos_policy.experiments.robot.hanoi.run_hanoi_dense_eval import (
    ACCEPTED_GPUS, FUTURE_FRAME_INDEX, HANOI_UNDO_INJECTION, per_sample_metrics, select_rows, summarize,
)
from cosmos_policy.experiments.robot.hanoi.run_hanoi_multitask_eval import LATENT_KEYS, chunk_error_mm

MATCH_MM = 5.0
DECISION_MM = 5.0
SEGMENT_KINDS = ('walk', 'crop', 'clip')


def decision_rows(arrays):
    """Archive indices at the goal-sensitive stages."""
    return np.flatnonzero(np.isin(arrays['motion_stages'], DECISION_STAGES))


def matched_goal_pairs(arrays, rows, match_mm=MATCH_MM, decision_mm=DECISION_MM, max_per_board=None, rng=None):
    """(row i, row j, label gap mm): same current board and motion stage, positions within match_mm, different goals whose
    labelled moves lead to different next boards, labels differing by more than decision_mm.

    Same stage and different next board are what make the pair a goal decision: the first version of this probe paired an
    approach with a transit at the same position, or two rows of the same move, where the labels differ for reasons the
    observation alone explains (ring held or not, elapsed time), so the sentence could not matter."""
    pairs = []
    keys = arrays['board_indices'][rows] * 100 + arrays['motion_stages'][rows]
    for key in np.unique(keys):
        group = rows[keys == key]
        if len(group) < 2:
            continue
        if max_per_board and len(group) > max_per_board:
            group = group[sorted((rng or np.random.default_rng(1)).choice(len(group), max_per_board, replace=False))]
        positions = arrays['cartesian_positions'][group]
        goals = arrays['goal_board_indices'][group]
        following = arrays['next_board_indices'][group]
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2) * 1000
        for a in range(len(group)):
            candidates = np.flatnonzero((distances[a] <= match_mm) & (goals != goals[a]) & (following != following[a]))
            if not len(candidates):
                continue
            b = int(candidates[np.argmin(distances[a][candidates])])
            i, j = int(group[a]), int(group[b])
            valid = ~(arrays['actions_is_pad'][i] | arrays['actions_is_pad'][j])
            if not valid.any():
                continue
            gap = float(np.linalg.norm(arrays['actions'][i][valid, :3] - arrays['actions'][j][valid, :3], axis=1).mean() * 1000)
            if gap > decision_mm:
                pairs.append((i, j, gap))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings_play.pkl')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--stride', type=int, default=9, help='Evaluate every stride-th row per walk, random phase')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=5, help='Denoising steps for the primary pass')
    parser.add_argument('--also-steps', type=int, default=0, help='Optional second pass with this many steps (actions only)')
    parser.add_argument('--future', action='store_true', help='Decode the future frame and report L1/PSNR')
    parser.add_argument('--parity-samples', type=int, default=0)
    parser.add_argument('--parity-tolerance-mm', type=float, default=0.5)
    parser.add_argument('--probe-rows', type=int, default=0, help='Goal probe: matched-goal decision pairs and shuffled-goal rows')
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
    metadata = json.loads((args.metadata / 'metadata.json').read_text())
    horizon = int(metadata['horizon'])
    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_dense'
    os.environ.setdefault('HANOI_DENSE_HORIZON', str(horizon))
    import torch
    from torch.utils.data import DataLoader, Subset
    from cosmos_policy.datasets.hanoi_play_dataset import HanoiPlayDataset
    from cosmos_policy.experiments.robot.cosmos_utils import (
        extract_action_chunk_from_latent_sequence, extract_value_from_latent_sequence, undo_latent_injection, unnormalize_actions)
    from cosmos_policy.experiments.robot.hanoi.dense_policy import threshold_jaw
    from cosmos_policy.experiments.robot.hanoi.play_policy import HanoiPlayInferenceConfig, load_play_policy, predict_play_actions

    gpu = torch.cuda.get_device_name() if torch.cuda.device_count() else 'none'
    if torch.cuda.device_count() != 1 or not any(tag in gpu for tag in ACCEPTED_GPUS):
        raise RuntimeError(f'Evaluate on one H100 or H200, not {gpu!r} x{torch.cuda.device_count()}')
    cfg = HanoiPlayInferenceConfig(str(args.checkpoint), str(args.metadata / 'dataset_statistics.json'), args.embeddings,
                                   num_denoising_steps_action=args.steps, chunk_size=horizon)
    model, stats, _, identity = load_play_policy(cfg)
    dataset = HanoiPlayDataset(str(args.metadata), args.embeddings, split=args.split, representative_order=False)
    selected, phases = select_rows(dataset, args.stride)
    arrays = dataset.arrays
    current_jaw = np.zeros(len(selected), np.float32)  # jaw intent at the observation row, audit input only
    for file_index in np.unique(arrays['file_indices'][selected]):
        mask = arrays['file_indices'][selected] == file_index
        column = dataset._file(int(file_index))['action_abs'][:, 3]
        current_jaw[mask] = column[arrays['source_observation_indices'][selected[mask]]]
    report = {'checkpoint': str(args.checkpoint.resolve()), 'split': args.split, 'contract': identity['contract'], 'horizon': horizon,
              'horizon_cap_moves': metadata.get('horizon_cap_moves'),
              'metadata': str(args.metadata.resolve()), 'embeddings': str(Path(args.embeddings).resolve()), 'gpu': gpu, 'torch': torch.__version__,
              'stride': args.stride, 'phases': phases, 'num_samples': int(len(selected)), 'batch_size': args.batch_size,
              'denoising_steps': args.steps, 'stationary_definition': f'finite-difference measured speed under {STATIONARY_SPEED_M_PER_S} m/s',
              'decision_stages': [STAGE_NAMES[s] for s in DECISION_STAGES],
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
                    metric.update({'archive_index': i, 'episode': int(arrays['episode_indices'][i]),
                                   'row': int(arrays['source_observation_indices'][i]), 'stationary': bool(arrays['stationary'][i]),
                                   'file_index': int(arrays['file_indices'][i]), 'segment_kind': SEGMENT_KINDS[int(arrays['segment_kinds'][i])],
                                   'motion_stage': int(arrays['motion_stages'][i]), 'decision': bool(np.isin(arrays['motion_stages'][i], DECISION_STAGES)),
                                   'board': BOARDS[int(arrays['board_indices'][i])], 'goal_board': BOARDS[int(arrays['goal_board_indices'][i])],
                                   'goal_moves_ahead': int(arrays['goal_moves_ahead'][i]), 'goal_graph_distance': int(arrays['goal_graph_distance'][i]),
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
                'moving': summarize([s for s in samples if not s['stationary']]),
                'decision_rows': summarize([s for s in samples if s['decision']]),
                'other_rows': summarize([s for s in samples if not s['decision']])}

    def group_tables(samples):
        return {'by_goal_moves_ahead': {str(k): summarize([s for s in samples if s['goal_moves_ahead'] == k])
                                        for k in sorted({s['goal_moves_ahead'] for s in samples})},
                'by_goal_graph_distance': {str(d): summarize([s for s in samples if s['goal_graph_distance'] == d])
                                           for d in sorted({s['goal_graph_distance'] for s in samples})},
                'by_stage': {STAGE_NAMES.get(st, str(st)): summarize([s for s in samples if s['motion_stage'] == st])
                             for st in sorted({s['motion_stage'] for s in samples})},
                'by_segment_kind': {kind: summarize([s for s in samples if s['segment_kind'] == kind])
                                    for kind in sorted({s['segment_kind'] for s in samples})}}

    primary = batched_pass(args.steps, args.future)
    report['metrics'] = tables(primary)
    report['metrics_by_group'] = group_tables(primary)
    report['per_sample'] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()
                             if k not in ('valid', 'jaw_correct', 'target_jaw')} for s in primary]
    if args.also_steps:
        secondary = batched_pass(args.also_steps, False)
        report[f'metrics_{args.also_steps}_steps'] = tables(secondary)
        report[f'metrics_{args.also_steps}_steps_by_group'] = group_tables(secondary)

    def single_prediction(i, prompt, seed=1):
        sample = dataset.raw_example(int(i))
        return predict_play_actions(cfg, model, stats, sample['image'], sample['state'], prompt, seed=seed)

    if args.parity_samples:
        k = max(1, len(selected) // args.parity_samples)
        parity_rows = selected[::k][:args.parity_samples]
        differences, batched_differences = [], []
        by_index = {s['archive_index']: s for s in primary}
        with torch.no_grad():
            for i in parity_rows:
                if int(i) not in by_index:
                    continue  # all-padded chunk, not scored
                served = single_prediction(i, PROMPTS[int(arrays['goal_board_indices'][i])])
                item = dataset[int(i)]
                single = {
                    'dataset_name': 'video_data', 'video': item['video'][None].to(dtype=torch.uint8).cuda(),
                    't5_text_embeddings': item['t5_text_embeddings'][None].to(dtype=torch.bfloat16).cuda(),
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
        probe = {'definition': __doc__.split('``--probe-rows``', 1)[1].strip(), 'match_mm': MATCH_MM, 'decision_mm': DECISION_MM}
        rng = np.random.default_rng(1)
        candidates = decision_rows(arrays)
        pairs = matched_goal_pairs(arrays, candidates, max_per_board=400, rng=rng)
        probe['candidate_pairs'] = len(pairs)
        if len(pairs) > args.probe_rows:
            pairs = [pairs[j] for j in sorted(rng.choice(len(pairs), args.probe_rows, replace=False))]
        correct_own, correct_swapped, swap_mm, own_mm, gaps = [], [], [], [], []
        with torch.no_grad():
            for i, j, gap in pairs:
                own_prompt, other_prompt = PROMPTS[int(arrays['goal_board_indices'][i])], PROMPTS[int(arrays['goal_board_indices'][j])]
                own = single_prediction(i, own_prompt)
                swapped = single_prediction(i, other_prompt)
                pad_i, pad_j = arrays['actions_is_pad'][i], arrays['actions_is_pad'][j]
                label_i, label_j = arrays['actions'][i], arrays['actions'][j]
                e_own_i, e_own_j = chunk_error_mm(own, label_i, pad_i), chunk_error_mm(own, label_j, pad_j)
                e_sw_i, e_sw_j = chunk_error_mm(swapped, label_i, pad_i), chunk_error_mm(swapped, label_j, pad_j)
                correct_own.append(e_own_i < e_own_j)
                correct_swapped.append(e_sw_j < e_sw_i)
                own_mm.append(e_own_i); gaps.append(gap)
                swap_mm.append(float(np.linalg.norm(own[:, :3] - swapped[:, :3], axis=1).mean() * 1000))
            probe['matched_goal_decisions'] = {
                'decision_pairs': len(pairs),
                'own_prompt_correct': float(np.mean(correct_own)) if pairs else None,
                'swapped_prompt_correct': float(np.mean(correct_swapped)) if pairs else None,
                'own_prompt_error_mm': float(np.mean(own_mm)) if pairs else None,
                'label_gap_mm': float(np.mean(gaps)) if pairs else None,
                'swap_displacement_mm': float(np.mean(swap_mm)) if pairs else None,
            }
            probe['decision_accuracy'] = float(np.mean(correct_own + correct_swapped)) if pairs else None
            probe['decision_rows_total'] = int(2 * len(pairs))
            # Shuffled-goal sensitivity on decision rows, against the noise between two draws under the same sentence.
            chosen_rows = [s['archive_index'] for s in primary if s['decision']]
            chosen = [chosen_rows[j] for j in sorted(rng.choice(len(chosen_rows), min(args.probe_rows, len(chosen_rows)), replace=False))] if chosen_rows else []
            shuffled_mm, noise_mm = [], []
            for i in chosen:
                goal = int(arrays['goal_board_indices'][i])
                other = int(rng.choice([g for g in range(len(BOARDS)) if g != goal]))
                own = single_prediction(i, PROMPTS[goal], seed=1)
                shuffled = single_prediction(i, PROMPTS[other], seed=1)
                redraw = single_prediction(i, PROMPTS[goal], seed=2)
                shuffled_mm.append(float(np.linalg.norm(own[:, :3] - shuffled[:, :3], axis=1).mean() * 1000))
                noise_mm.append(float(np.linalg.norm(own[:, :3] - redraw[:, :3], axis=1).mean() * 1000))
            probe['shuffled_goal_sensitivity'] = {
                'rows': len(chosen),
                'shuffled_displacement_mm': {'mean': float(np.mean(shuffled_mm)), 'median': float(np.median(shuffled_mm))} if chosen else None,
                'same_prompt_redraw_mm': {'mean': float(np.mean(noise_mm)), 'median': float(np.median(noise_mm))} if chosen else None,
            }
        report['language_probe'] = probe
    dataset.close()
    report['finished_at'] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    summary = {k: v for k, v in report.items() if k not in ('per_sample', 'phases', 'metrics_by_group')}
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
