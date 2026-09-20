"""Split per-slot chunk error of a hanoi_dense_v5 export into off-path and along-path parts.

For every sampled validation observation the predicted 16 poses are compared
with the commanded reference trajectory that actually followed: the off-path
part is the distance from each predicted pose to the nearest point of that
trajectory (a spatial error the arm would really make), and the along-path part
is how far ahead or behind along the trajectory the prediction sits (a timing
error, which asynchronous re-planning corrects). Reported per slot, for all
rows and split by observation motion.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from cosmos_policy.datasets.hanoi_dense_data import FRAMESKIP


def nearest_on_polyline(points, q):
    """Distance from q to the polyline `points` and the fractional segment index of the nearest point."""
    a, b = points[:-1], points[1:]
    ab = b - a
    t = np.clip(np.einsum('ij,ij->i', q - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12), 0, 1)
    proj = a + t[:, None] * ab
    d = np.linalg.norm(proj - q, axis=1)
    k = int(np.argmin(d))
    return float(d[k]), k + float(t[k])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings.pkl')
    parser.add_argument('--stride', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    horizon = int(json.loads((args.metadata / 'metadata.json').read_text())['horizon'])
    window_rows = FRAMESKIP * horizon + 30  # trajectory context after the observation: the chunk plus one second
    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_dense'
    os.environ.setdefault('HANOI_DENSE_HORIZON', str(horizon))
    import h5py
    import torch
    from torch.utils.data import DataLoader, Subset
    from cosmos_policy.datasets.hanoi_dense_dataset import HanoiDenseDataset
    from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence, unnormalize_actions
    from cosmos_policy.experiments.robot.hanoi.dense_policy import HanoiDenseInferenceConfig, load_dense_policy
    from cosmos_policy.experiments.robot.hanoi.run_hanoi_dense_eval import select_rows

    cfg = HanoiDenseInferenceConfig(str(args.checkpoint), str(args.metadata / 'dataset_statistics.json'), args.embeddings,
                                    num_denoising_steps_action=args.steps, chunk_size=horizon)
    model, stats, _, _ = load_dense_policy(cfg)
    dataset = HanoiDenseDataset(str(args.metadata), args.embeddings, split='val', representative_order=False)
    selected, _ = select_rows(dataset, args.stride)
    arrays = dataset.arrays
    with h5py.File(dataset.source, 'r') as h:
        reference = h['reference_pose'][:, :3].astype(np.float64)
    loader = DataLoader(Subset(dataset, selected.tolist()), batch_size=args.batch_size, num_workers=4, pin_memory=True)
    records = []
    with torch.no_grad():
        for b, batch in enumerate(loader):
            n = int(batch['video'].shape[0])
            data_batch = {
                'dataset_name': 'video_data', 'video': batch['video'].to(dtype=torch.uint8).cuda(),
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
            latent, _ = model.generate_samples_from_batch(data_batch, n_sample=n, num_steps=args.steps, seed=195 + b,
                                                          is_negative_prompt=False, use_variance_scale=False,
                                                          return_orig_clean_latent_frames=True)
            actions = unnormalize_actions(extract_action_chunk_from_latent_sequence(
                latent, (horizon, 4), data_batch['action_latent_idx']).float().cpu().numpy(), stats)
            for k in range(n):
                i = int(batch['__key__'][k])
                pad = arrays['actions_is_pad'][i]
                if pad.all():
                    continue
                row = int(arrays['source_observation_indices'][i]); end = int(arrays['source_episode_bounds'][i, 1])
                path = reference[row:min(row + window_rows, end)]
                target = arrays['actions'][i][:, :3].astype(np.float64)
                predicted = actions[k][:, :3].astype(np.float64)
                total = np.linalg.norm(predicted - target, axis=1) * 1000
                off_path, timing_rows = np.full(horizon, np.nan), np.full(horizon, np.nan)
                for j in range(horizon):
                    if pad[j]:
                        continue
                    d, where = nearest_on_polyline(path, predicted[j])
                    off_path[j] = d * 1000
                    timing_rows[j] = where - FRAMESKIP * (j + 1)  # + means the prediction sits further along the path than the label
                records.append({'row': row, 'stationary': bool(arrays['stationary'][i]), 'total_mm': total.tolist(),
                                'off_path_mm': off_path.tolist(), 'timing_rows': timing_rows.tolist()})
    dataset.close()

    def table(subset):
        if not subset:
            return {'samples': 0}
        total = np.array([r['total_mm'] for r in subset]); off = np.array([r['off_path_mm'] for r in subset]); tim = np.array([r['timing_rows'] for r in subset])
        return {'samples': len(subset),
                'total_mm_per_slot': [float(np.nanmean(total[:, j])) for j in range(horizon)],
                'off_path_mm_per_slot': [float(np.nanmean(off[:, j])) for j in range(horizon)],
                'off_path_mm_p95_per_slot': [float(np.nanpercentile(off[:, j], 95)) for j in range(horizon)],
                'abs_timing_rows_per_slot': [float(np.nanmean(np.abs(tim[:, j]))) for j in range(horizon)],
                'signed_timing_rows_per_slot': [float(np.nanmean(tim[:, j])) for j in range(horizon)]}
    report = {'checkpoint': str(args.checkpoint.resolve()), 'stride': args.stride, 'steps': args.steps, 'horizon': horizon, 'window_rows': window_rows,
              'definition': 'off_path = distance to the nearest point of the commanded reference trajectory that followed the observation; '
                            'timing_rows = position of that nearest point along the trajectory minus the label row, in 30 Hz rows',
              'all': table(records), 'stationary': table([r for r in records if r['stationary']]),
              'moving': table([r for r in records if not r['stationary']])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + '\n')
    for name in ('all', 'stationary', 'moving'):
        t = report[name]
        if t['samples']:
            shown = [j for j in (0, 3, 7, 15, 31) if j < horizon]
            print(name, t['samples'], 'rows | slot', [j + 1 for j in shown], 'total', [round(t['total_mm_per_slot'][j], 2) for j in shown],
                  '| off-path', [round(t['off_path_mm_per_slot'][j], 2) for j in shown],
                  '| |timing| rows', [round(t['abs_timing_rows_per_slot'][j], 1) for j in shown])


if __name__ == '__main__':
    main()
