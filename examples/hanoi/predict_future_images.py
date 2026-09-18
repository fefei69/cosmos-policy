"""Decode the policy's predicted future frame for held-out examples.

Cosmos Policy predicts the auxiliary future RGB frame (latent slot 5) jointly
with the actions. This script runs the native inference path with future-state
generation enabled, decodes that slot with the VAE, and compares it with the
frame the model was trained to predict: the raw frame at
min(last selected target row + 1, episode end - 1). For each example it saves a
PNG strip [current | predicted future | recorded future | abs difference] and
records L1 and PSNR against the recorded frame, next to the trivial baseline of
copying the current frame. Works for joint_v3 and waypoint_v4 exports on one
H100 or H200. This is a qualitative check of the world-model head; it says
nothing about task success.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

ACCEPTED_GPUS = ('H100', 'H200')


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float('inf') if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def l1(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings.pkl')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--samples', type=int, default=16, help='Examples in representative (episode-interleaved) order')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--denoising-steps', type=int, default=5)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output_dir}')
    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_joint'
    import torch
    from PIL import Image
    from cosmos_policy.datasets.hanoi_joint_data import CONTRACT as JOINT_CONTRACT
    from cosmos_policy.datasets.hanoi_joint_dataset import HanoiJointDataset
    from cosmos_policy.datasets.hanoi_waypoint_data import CONTRACT as WAYPOINT_CONTRACT
    from cosmos_policy.datasets.hanoi_waypoint_dataset import HanoiWaypointDataset
    from cosmos_policy.experiments.robot.cosmos_utils import get_action
    from cosmos_policy.experiments.robot.hanoi.joint_policy import PROMPT, absolute_joint_actions, make_joint_observation
    from cosmos_policy.experiments.robot.hanoi.waypoint_policy import inference_config_for, load_policy

    gpu = torch.cuda.get_device_name() if torch.cuda.device_count() else 'none'
    if torch.cuda.device_count() != 1 or not any(tag in gpu for tag in ACCEPTED_GPUS):
        raise RuntimeError(f'Run on one H100 or H200, not {gpu!r} x{torch.cuda.device_count()}')
    contract = json.loads((args.metadata / 'metadata.json').read_text())['contract']
    dataset_class = {JOINT_CONTRACT: HanoiJointDataset, WAYPOINT_CONTRACT: HanoiWaypointDataset}[contract]
    cfg = inference_config_for(contract, args.checkpoint, args.metadata / 'dataset_statistics.json', args.embeddings)
    model, stats, _ = load_policy(cfg, contract)
    dataset = dataset_class(str(args.metadata), args.embeddings, split=args.split, representative_order=args.split == 'val')
    count = min(args.samples, len(dataset))
    args.output_dir.mkdir(parents=True)
    report = {'checkpoint': str(args.checkpoint.resolve()), 'contract': contract, 'split': args.split, 'gpu': gpu,
              'seed': args.seed, 'denoising_steps': args.denoising_steps, 'started_at': time.time(), 'samples': []}
    with torch.no_grad():
        for n in range(count):
            sample = dataset.raw_example(n)
            item = dataset[n]
            future_row = int(item['auxiliary_future_source_row'])
            recorded_future = np.asarray(dataset._file()['pixels'][future_row])
            observation = make_joint_observation(sample['image'], sample['state'])
            out = get_action(cfg, model, stats, observation, PROMPT, seed=args.seed, randomize_seed=False,
                             num_denoising_steps_action=args.denoising_steps,
                             generate_future_state_and_value_in_parallel=True, batch_size=1)
            predicted = np.asarray(out['future_image_predictions']['future_image'])
            if predicted.shape != (224, 224, 3) or predicted.dtype != np.uint8:
                raise ValueError(f'Unexpected decoded future frame {predicted.shape} {predicted.dtype}')
            actions = absolute_joint_actions(out['actions'], sample['cartesian_position'])
            true_value = float((item['value_function_return'] + 1) / 2)
            diff = np.abs(predicted.astype(np.int16) - recorded_future.astype(np.int16)).astype(np.uint8)
            strip = np.concatenate([sample['image'], predicted, recorded_future, diff], axis=1)
            name = f'{n:03d}_ep{sample["episode_index"]}_row{sample["source_observation_index"]}'
            Image.fromarray(strip).save(args.output_dir / f'{name}.png')
            record = {
                'index': n, 'episode': sample['episode_index'], 'observation_row': sample['source_observation_index'],
                'future_row': future_row, 'rows_ahead': future_row - sample['source_observation_index'],
                'predicted_vs_recorded_l1': l1(predicted, recorded_future), 'predicted_vs_recorded_psnr': psnr(predicted, recorded_future),
                'copy_current_l1': l1(sample['image'], recorded_future), 'copy_current_psnr': psnr(sample['image'], recorded_future),
                'value_predicted': float(out['value_prediction']), 'value_true': true_value,
                'first_target_xyz_mm': float(np.linalg.norm(actions[0, :3] - sample['actions_abs'][0, :3]) * 1000),
                'png': f'{name}.png',
            }
            report['samples'].append(record)
            print(json.dumps(record), flush=True)
    dataset.close()
    keys = ('predicted_vs_recorded_l1', 'predicted_vs_recorded_psnr', 'copy_current_l1', 'copy_current_psnr',
            'first_target_xyz_mm', 'rows_ahead')
    finite = lambda key: [s[key] for s in report['samples'] if np.isfinite(s[key])]
    report['summary'] = {key: float(np.mean(finite(key))) for key in keys}
    report['summary']['value_abs_error'] = float(np.mean([abs(s['value_predicted'] - s['value_true']) for s in report['samples']]))
    report['summary']['fraction_better_than_copy_l1'] = float(np.mean(
        [s['predicted_vs_recorded_l1'] < s['copy_current_l1'] for s in report['samples']]))
    report['finished_at'] = time.time()
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
