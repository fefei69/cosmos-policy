"""Fixed-noise Cosmos loss and physical metrics on explicit sparse targets."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def physical_metrics(predicted, target, padding, first_event=None):
    predicted, target, padding = np.asarray(predicted), np.asarray(target), np.asarray(padding)
    if predicted.shape != (8, 4) or target.shape != (8, 4) or padding.shape != (8,) or padding[0]:
        raise ValueError('Expected an eight-target chunk with a real first target')
    valid = ~padding
    errors = np.linalg.norm(predicted[:, :3] - target[:, :3], axis=-1) * 1000
    truth, guess = target[valid, 3] >= .5, predicted[valid, 3] >= .5
    return {'first_xyz_mm': float(errors[0]), 'valid_horizon_xyz_mm': float(errors[valid].mean()),
            'last_valid_xyz_mm': float(errors[np.flatnonzero(valid)[-1]]), 'first_event': first_event,
            'jaw_tn': int((~truth & ~guess).sum()), 'jaw_fp': int((~truth & guess).sum()),
            'jaw_fn': int((truth & ~guess).sum()), 'jaw_tp': int((truth & guess).sum()),
            'valid_targets': int(valid.sum())}


def summarize(samples):
    result = {key: float(np.mean([s[key] for s in samples]))
              for key in ('first_xyz_mm', 'valid_horizon_xyz_mm', 'last_valid_xyz_mm')}
    for key in ('jaw_tn', 'jaw_fp', 'jaw_fn', 'jaw_tp', 'valid_targets'):
        result[key] = sum(s[key] for s in samples)
    closed, opened = result['jaw_tn'] + result['jaw_fp'], result['jaw_tp'] + result['jaw_fn']
    result['jaw_closed_support'], result['jaw_open_support'] = closed, opened
    result['jaw_accuracy'] = (result['jaw_tn'] + result['jaw_tp']) / result['valid_targets']
    result['jaw_balanced_accuracy'] = ((result['jaw_tn'] / closed + result['jaw_tp'] / opened) / 2
                                       if closed and opened else None)
    for event in ('open', 'close'):
        selected = [s for s in samples if s['first_event'] == event]
        result[f'{event}_first_target_support'] = len(selected)
        result[f'{event}_first_xyz_mm'] = float(np.mean([s['first_xyz_mm'] for s in selected])) if selected else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, default=Path('data/hanoi_cosmos/joint_sparse_v3'))
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings.pkl')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--samples', type=int, default=0, help='0 means complete split')
    parser.add_argument('--mode', choices=['loss', 'actions', 'both'], default='both')
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
    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_joint'
    import torch
    from torch.utils.data import DataLoader, Subset
    from cosmos_policy._src.imaginaire.utils import misc
    from cosmos_policy._src.imaginaire.utils.callback import LowPrecisionCallback
    from cosmos_policy.datasets.hanoi_joint_dataset import HanoiJointDataset
    from cosmos_policy.experiments.robot.hanoi.joint_policy import HanoiJointInferenceConfig, HanoiJointPolicy, load_joint_policy, predict_joint_actions
    from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE
    from cosmos_policy.utils.hanoi_training import HanoiTrainingMonitor

    if torch.cuda.device_count() != 1 or 'H100' not in torch.cuda.get_device_name():
        raise RuntimeError('Evaluate on one H100, matching this Cosmos training configuration')
    cfg = HanoiJointInferenceConfig(str(args.checkpoint), str(args.metadata / 'dataset_statistics.json'), args.embeddings)
    model, stats, _ = load_joint_policy(cfg)
    dataset = HanoiJointDataset(str(args.metadata), args.embeddings, split=args.split, representative_order=False)
    selected = np.arange(len(dataset))
    if args.samples:
        if args.samples < 1:
            raise ValueError('Sample count must be nonnegative')
        rng = np.random.default_rng(195)
        groups = [rng.permutation(np.flatnonzero(dataset.arrays['episode_indices'] == ep))
                  for ep in np.unique(dataset.arrays['episode_indices'])]
        selected = np.array([g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)])[:args.samples]
    report = {'checkpoint': str(args.checkpoint.resolve()), 'split': args.split, 'num_samples': len(selected),
              'mode': args.mode, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'optimizer_updates': 0, 'noise_seed': 195, 'inference_seed': 1, 'inference_steps': 5,
              'padding_in_loss': True, 'padding_in_accuracy': False,
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
            # Share the actual serving adapter and verify its coordinate conversion
            # and input path against offline inference with identical seeded noise.
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
                samples.append({'episode': sample['episode_index'], 'raw_observation_row': sample['source_observation_index'], **metric})
                if (n + 1) % 100 == 0:
                    print(f'Inference: {n + 1}/{len(selected)}', flush=True)
            report['samples'], report['physical_metrics'] = samples, summarize(samples)
            report['physical_metrics_by_episode'] = {int(ep): summarize([s for s in samples if s['episode'] == ep])
                                                     for ep in sorted({s['episode'] for s in samples})}
    dataset.close()
    report['finished_at'] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples', 'selected_archive_indices')}, indent=2))


if __name__ == '__main__':
    main()
