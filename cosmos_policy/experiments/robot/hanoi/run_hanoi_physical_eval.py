"""Held-out evaluation for joint_v3 or waypoint_v4 exports on one H100 or H200.

Physical metrics decode sampled actions to absolute XYZ + jaw intent, exactly as
the serving adapter does. Beyond the joint_v3 evaluation this reports, for the
committed first target: p95 XYZ error, the fraction within a tolerance, jaw
intent accuracy, and the hit rate (within tolerance AND correct jaw intent),
because waypoint_v4 destinations are a discrete set and mean error hides
all-or-nothing outcomes. These are offline imitation metrics, not task success.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from cosmos_policy.experiments.robot.hanoi.run_hanoi_joint_eval import physical_metrics, summarize

ACCEPTED_GPUS = ('H100', 'H200')
DEFAULT_TOLERANCE_MM = 5.0


def first_target_metrics(predicted, target, tolerance_mm):
    """Per-sample outcome of the one target the executor commits."""
    error_mm = float(np.linalg.norm(predicted[0, :3] - target[0, :3]) * 1000)
    jaw_correct = bool(predicted[0, 3] == target[0, 3])
    return {'first_jaw_correct': jaw_correct, 'first_within_tolerance': error_mm <= tolerance_mm,
            'first_hit': jaw_correct and error_mm <= tolerance_mm}


def summarize_physical(samples, tolerance_mm):
    result = summarize(samples)
    if samples:
        errors = np.array([s['first_xyz_mm'] for s in samples])
        result['first_xyz_mm_p95'] = float(np.percentile(errors, 95))
        result['first_xyz_mm_median'] = float(np.median(errors))
        result['first_within_tolerance_fraction'] = float(np.mean([s['first_within_tolerance'] for s in samples]))
        result['first_jaw_accuracy'] = float(np.mean([s['first_jaw_correct'] for s in samples]))
        result['first_hit_rate'] = float(np.mean([s['first_hit'] for s in samples]))
    result['hit_tolerance_mm'] = tolerance_mm
    result['samples'] = len(samples)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings.pkl')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--samples', type=int, default=0, help='0 means complete split')
    parser.add_argument('--mode', choices=['loss', 'actions', 'both'], default='both')
    parser.add_argument('--selection', type=Path, help='Locked validation choice required for test evaluation')
    parser.add_argument('--tolerance-mm', type=float, default=DEFAULT_TOLERANCE_MM)
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
    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_joint'
    import torch
    from torch.utils.data import DataLoader, Subset
    from cosmos_policy._src.imaginaire.utils import misc
    from cosmos_policy._src.imaginaire.utils.callback import LowPrecisionCallback
    from cosmos_policy.datasets.hanoi_joint_data import CONTRACT as JOINT_CONTRACT
    from cosmos_policy.datasets.hanoi_joint_dataset import HanoiJointDataset
    from cosmos_policy.datasets.hanoi_waypoint_data import CONTRACT as WAYPOINT_CONTRACT
    from cosmos_policy.datasets.hanoi_waypoint_dataset import HanoiWaypointDataset
    from cosmos_policy.experiments.robot.hanoi.joint_policy import HanoiJointPolicy, predict_joint_actions
    from cosmos_policy.experiments.robot.hanoi.waypoint_policy import inference_config_for, load_policy
    from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE
    from cosmos_policy.utils.hanoi_training import HanoiTrainingMonitor

    gpu = torch.cuda.get_device_name() if torch.cuda.device_count() else 'none'
    if torch.cuda.device_count() != 1 or not any(tag in gpu for tag in ACCEPTED_GPUS):
        raise RuntimeError(f'Evaluate on one H100 or H200 (Hopper, bf16), not {gpu!r} x{torch.cuda.device_count()}')
    contract = json.loads((args.metadata / 'metadata.json').read_text())['contract']
    dataset_class = {JOINT_CONTRACT: HanoiJointDataset, WAYPOINT_CONTRACT: HanoiWaypointDataset}[contract]
    cfg = inference_config_for(contract, args.checkpoint, args.metadata / 'dataset_statistics.json', args.embeddings)
    model, stats, _ = load_policy(cfg, contract)
    dataset = dataset_class(str(args.metadata), args.embeddings, split=args.split, representative_order=False)
    selected = np.arange(len(dataset))
    if args.samples:
        if args.samples < 1:
            raise ValueError('Sample count must be nonnegative')
        rng = np.random.default_rng(195)
        groups = [rng.permutation(np.flatnonzero(dataset.arrays['episode_indices'] == ep))
                  for ep in np.unique(dataset.arrays['episode_indices'])]
        selected = np.array([g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)])[:args.samples]
    report = {'checkpoint': str(args.checkpoint.resolve()), 'split': args.split, 'num_samples': len(selected),
              'mode': args.mode, 'contract': contract, 'metadata': str(args.metadata.resolve()),
              'gpu': gpu, 'torch': torch.__version__,
              'noise_seed': 195, 'inference_seed': 1, 'inference_steps': 5,
              'padding_in_loss': True, 'padding_in_accuracy': False, 'hit_tolerance_mm': args.tolerance_mm,
              'selected_archive_indices': selected.tolist(), 'started_at': time.time()}
    model.eval()
    with torch.no_grad():
        if args.mode in ('loss', 'both'):
            inference_sde = model.sde
            model.sde = HybridEDMSDE(p_mean=np.log(4), p_std=1.2, sigma_max=200, sigma_min=.01,
                                     hybrid_sigma_distribution=True, uniform_lower=1, uniform_upper=85)
            callback = LowPrecisionCallback(config=None, trainer=None, update_iter=1)
            callback.on_train_start(model)
            records = []
            loader = DataLoader(Subset(dataset, selected.tolist()), batch_size=2, num_workers=2, pin_memory=True)
            for batch in loader:
                batch = misc.to(batch, device='cuda')
                callback.on_validation_step_start(model, batch)
                output, loss = model.validation_step(batch, 0)
                records.append((len(batch['actions']), HanoiTrainingMonitor._metrics(output, loss)))
            report['denoising_metrics'] = HanoiTrainingMonitor._average(records)
            model.sde = inference_sde
        if args.mode in ('actions', 'both'):
            samples = []
            # The actual serving adapter shares the model; its coordinate conversion
            # and input path are checked against offline inference with identical seeded noise.
            serving = object.__new__(HanoiJointPolicy)
            serving.cfg, serving.model, serving.stats = cfg, model, stats
            for n, index in enumerate(selected):
                sample = dataset.raw_example(int(index))
                predicted = predict_joint_actions(cfg, model, stats, sample['image'], sample['state'], sample['cartesian_position'], seed=1)
                if n == 0:
                    served = serving.infer({'observation/image': sample['image'], 'observation/state': sample['state'],
                                            'observation/cartesian_position': sample['cartesian_position']}, seed=1)
                    np.testing.assert_array_equal(predicted, served['actions'])
                    assert served['commit_count'] == 1 and served['reference_rate_hz'] is None
                    report['serving_parity_passed'] = True
                first_row = int(sample['source_action_indices'][0])
                previous_jaw = dataset._file()['action_abs'][first_row - 1, 3]
                jaw = sample['actions_abs'][0, 3]
                event = ('open' if jaw == 1 else 'close') if previous_jaw != jaw else None
                metric = physical_metrics(predicted, sample['actions_abs'], sample['actions_is_pad'], event)
                metric.update(first_target_metrics(predicted, sample['actions_abs'], args.tolerance_mm))
                samples.append({'episode': sample['episode_index'], 'raw_observation_row': sample['source_observation_index'], **metric})
                if (n + 1) % 100 == 0:
                    print(f'Inference: {n + 1}/{len(selected)}', flush=True)
            report['samples'], report['physical_metrics'] = samples, summarize_physical(samples, args.tolerance_mm)
            report['physical_metrics_by_episode'] = {
                int(ep): summarize_physical([s for s in samples if s['episode'] == ep], args.tolerance_mm)
                for ep in sorted({s['episode'] for s in samples})}
    dataset.close()
    report['finished_at'] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('samples', 'selected_archive_indices', 'physical_metrics_by_episode')}, indent=2))


if __name__ == '__main__':
    main()
