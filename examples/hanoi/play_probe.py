"""Goal-following probe alone, on a hanoi_play_k5 export: the matched-goal decision test (same board and stage,
positions within 5 mm, different goals with different next boards) and the shuffled-goal sensitivity on decision rows.
Same definitions and seeds as run_hanoi_play_eval --probe-rows; this script only skips the batched metrics pass."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from cosmos_policy.datasets.hanoi_play_data import BOARDS, PROMPTS, STAGE_NAMES
from cosmos_policy.experiments.robot.hanoi.run_hanoi_dense_eval import select_rows
from cosmos_policy.experiments.robot.hanoi.run_hanoi_multitask_eval import chunk_error_mm
from cosmos_policy.experiments.robot.hanoi.run_hanoi_play_eval import DECISION_MM, MATCH_MM, decision_rows, matched_goal_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, default=Path('data/hanoi_cosmos/play_k5'))
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings_play.pkl')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--pairs', type=int, default=128)
    parser.add_argument('--shuffled-rows', type=int, default=128)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_dense'
    os.environ.setdefault('HANOI_DENSE_HORIZON', '16')
    import torch
    from cosmos_policy.datasets.hanoi_play_dataset import HanoiPlayDataset
    from cosmos_policy.experiments.robot.hanoi.play_policy import HanoiPlayInferenceConfig, load_play_policy, predict_play_actions
    cfg = HanoiPlayInferenceConfig(str(args.checkpoint), str(args.metadata / 'dataset_statistics.json'), args.embeddings,
                                   num_denoising_steps_action=args.steps)
    model, stats, _, identity = load_play_policy(cfg)
    dataset = HanoiPlayDataset(str(args.metadata), args.embeddings, split=args.split)
    arrays = dataset.arrays
    started = time.time()

    def predict(i, prompt, seed=1):
        sample = dataset.raw_example(int(i))
        return predict_play_actions(cfg, model, stats, sample['image'], sample['state'], prompt, seed=seed)

    rng = np.random.default_rng(1)
    candidates = decision_rows(arrays)
    pairs = matched_goal_pairs(arrays, candidates, max_per_board=400, rng=rng)
    report = {'checkpoint': str(args.checkpoint.resolve()), 'split': args.split, 'contract': identity['contract'],
              'denoising_steps': args.steps, 'match_mm': MATCH_MM, 'decision_mm': DECISION_MM, 'candidate_pairs': len(pairs),
              'rule': 'same current board and motion stage, positions within match_mm, different goal boards with different next boards, labels differing by more than decision_mm'}
    if len(pairs) > args.pairs:
        pairs = [pairs[j] for j in sorted(rng.choice(len(pairs), args.pairs, replace=False))]
    per_pair, correct_own, correct_swapped, swap_mm, own_mm = [], [], [], [], []
    with torch.no_grad():
        for i, j, gap in pairs:
            gi, gj = int(arrays['goal_board_indices'][i]), int(arrays['goal_board_indices'][j])
            own, swapped = predict(i, PROMPTS[gi]), predict(i, PROMPTS[gj])
            pad_i, pad_j = arrays['actions_is_pad'][i], arrays['actions_is_pad'][j]
            label_i, label_j = arrays['actions'][i], arrays['actions'][j]
            e_own_i, e_own_j = chunk_error_mm(own, label_i, pad_i), chunk_error_mm(own, label_j, pad_j)
            e_sw_i, e_sw_j = chunk_error_mm(swapped, label_i, pad_i), chunk_error_mm(swapped, label_j, pad_j)
            correct_own.append(e_own_i < e_own_j)
            correct_swapped.append(e_sw_j < e_sw_i)
            own_mm.append(e_own_i)
            displacement = float(np.linalg.norm(own[:, :3] - swapped[:, :3], axis=1).mean() * 1000)
            swap_mm.append(displacement)
            per_pair.append({'row': int(i), 'other_row': int(j), 'board': BOARDS[int(arrays['board_indices'][i])],
                             'stage': STAGE_NAMES[int(arrays['motion_stages'][i])], 'goal': BOARDS[gi], 'other_goal': BOARDS[gj],
                             'next_board': BOARDS[int(arrays['next_board_indices'][i])], 'other_next_board': BOARDS[int(arrays['next_board_indices'][j])],
                             'label_gap_mm': gap, 'own_correct': bool(correct_own[-1]), 'swapped_correct': bool(correct_swapped[-1]),
                             'own_error_mm': e_own_i, 'swapped_error_vs_other_label_mm': e_sw_j, 'swap_displacement_mm': displacement})
        report['matched_goal_decisions'] = {
            'decision_pairs': len(pairs), 'own_prompt_correct': float(np.mean(correct_own)), 'swapped_prompt_correct': float(np.mean(correct_swapped)),
            'own_prompt_error_mm': float(np.mean(own_mm)), 'label_gap_mm': float(np.mean([g for _, _, g in pairs])),
            'swap_displacement_mm': float(np.mean(swap_mm)), 'swap_displacement_median_mm': float(np.median(swap_mm)),
            'by_stage': {stage: {'pairs': sum(1 for p in per_pair if p['stage'] == stage),
                                 'own_correct': float(np.mean([p['own_correct'] for p in per_pair if p['stage'] == stage])),
                                 'swapped_correct': float(np.mean([p['swapped_correct'] for p in per_pair if p['stage'] == stage]))}
                         for stage in sorted({p['stage'] for p in per_pair})}}
        report['decision_accuracy'] = float(np.mean(correct_own + correct_swapped))
        selected, _ = select_rows(dataset, 9)
        chosen_rows = [int(r) for r in selected if int(arrays['motion_stages'][r]) in (2, 6) and not arrays['actions_is_pad'][r].all()]
        chosen = [chosen_rows[j] for j in sorted(rng.choice(len(chosen_rows), min(args.shuffled_rows, len(chosen_rows)), replace=False))]
        shuffled_mm, noise_mm = [], []
        for i in chosen:
            goal = int(arrays['goal_board_indices'][i])
            other = int(rng.choice([g for g in range(len(BOARDS)) if g != goal]))
            own, shuffled, redraw = predict(i, PROMPTS[goal], 1), predict(i, PROMPTS[other], 1), predict(i, PROMPTS[goal], 2)
            shuffled_mm.append(float(np.linalg.norm(own[:, :3] - shuffled[:, :3], axis=1).mean() * 1000))
            noise_mm.append(float(np.linalg.norm(own[:, :3] - redraw[:, :3], axis=1).mean() * 1000))
        report['shuffled_goal_sensitivity'] = {
            'rows': len(chosen),
            'shuffled_displacement_mm': {'mean': float(np.mean(shuffled_mm)), 'median': float(np.median(shuffled_mm)),
                                         'fraction_over_5mm': float(np.mean(np.array(shuffled_mm) > 5))},
            'same_prompt_redraw_mm': {'mean': float(np.mean(noise_mm)), 'median': float(np.median(noise_mm))}}
    report['per_pair'] = per_pair
    report['seconds'] = time.time() - started
    dataset.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_pair'}, indent=1))


if __name__ == '__main__':
    main()
