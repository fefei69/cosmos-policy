"""Cut one episode out of the raw Hanoi recording into a self-contained HDF5.

Every per-row dataset of the source is copied for that episode's rows (pixels
stay gzip-compressed per frame), the file attributes are copied, and the
waypoint_v4 and joint_v3 label rows for the episode are attached under
``labels/<name>`` with indices re-based to the episode (plus the original
absolute rows). Intended for local dry runs: replaying observations through
the policy, or driving the robot along the recorded path.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

DEFAULT_SOURCE = Path('/scratch/cw5167/datasets/hanoi_wm_roundtrip_20260915_223424_AAAA_to_CCCC.h5')
LABEL_SETS = {'waypoint_v4': Path('data/hanoi_cosmos/waypoint_v4'), 'joint_v3': Path('data/hanoi_cosmos/joint_sparse_v3')}
NOTES = {
    'rows': 'one row per 30 Hz collector tick; the episode is rows 0..ep_len-1 of this file',
    'pixels': 'RGB uint8 224x224, already cropped with the contract rectangle; the policy input image',
    'joint_positions': 'six measured joint angles, radians, Trossen driver order 0-5; policy state[0:6]',
    'proprio': 'columns: measured XYZ (m) [0:3], velocities [3:6] (not a policy input), jaw stroke (m) [6] = policy state[6], legacy commanded gripper [7]',
    'action_abs': 'post-action reference: destination XYZ (m) and jaw intent (0 closed / 1 open) of the command active after this row',
    'leg_idx': 'motion leg counter; waypoint_v4 targets are leg endpoints plus gripper events',
    'command_monotonic_ns - image_receipt_monotonic_ns': 'image age at the command; training admitted rows with age in [0, 50 ms]',
    'labels/<set>': 'per-observation supervision rows for this episode: observation_row (local), targets (8 local rows), actions (8 x [xyz, jaw]), actions_is_pad, plus *_absolute for the source file',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, default=40, help='40-44 are validation, 45-49 test, 0-39 training')
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path(f'data/hanoi_cosmos/exports_local/hanoi_episode_{args.episode:03d}.h5')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source, 'r') as src:
        lo = int(src['ep_offset'][args.episode])
        length = int(src['ep_len'][args.episode])
        hi = lo + length
        with h5py.File(output, 'w') as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs['extracted_episode'] = args.episode
            dst.attrs['source_rows'] = [lo, hi]
            dst.attrs['source_file'] = str(args.source)
            dst.attrs['notes_json'] = json.dumps(NOTES)
            for name, dataset in src.items():
                if name in ('ep_offset', 'ep_len'):
                    continue
                if dataset.shape[0] != src['pixels'].shape[0]:
                    dst.create_dataset(name, data=dataset[()])
                    continue
                if name == 'pixels':
                    out = dst.create_dataset(name, shape=(length, 224, 224, 3), dtype='uint8',
                                             chunks=(1, 224, 224, 3), compression='gzip')
                    for start in range(0, length, 256):
                        out[start:min(start + 256, length)] = dataset[lo + start:min(lo + start + 256, hi)]
                else:
                    dst.create_dataset(name, data=dataset[lo:hi])
            dst.create_dataset('ep_offset', data=np.array([0], np.int64))
            dst.create_dataset('ep_len', data=np.array([length], np.int32))
            labels = dst.create_group('labels')
            for name, root in LABEL_SETS.items():
                if not (root / 'metadata.json').exists():
                    continue
                for split in ('train', 'val', 'test'):
                    with np.load(root / f'{split}.npz', allow_pickle=False) as archive:
                        mask = archive['episode_indices'] == args.episode
                        if not mask.any():
                            continue
                        group = labels.create_group(name)
                        group.attrs['split'] = split
                        group.attrs['contract'] = json.loads((root / 'metadata.json').read_text())['contract']
                        obs, tgt = archive['source_observation_indices'][mask], archive['source_action_indices'][mask]
                        group.create_dataset('observation_row', data=obs - lo)
                        group.create_dataset('observation_row_absolute', data=obs)
                        group.create_dataset('targets', data=tgt - lo)
                        group.create_dataset('targets_absolute', data=tgt)
                        for key in ('states', 'cartesian_positions', 'actions', 'actions_is_pad'):
                            group.create_dataset(key, data=archive[key][mask])
                        break
    size = output.stat().st_size / 1e6
    print(json.dumps({'output': str(output), 'episode': args.episode, 'rows': length, 'source_rows': [lo, hi], 'megabytes': round(size, 1)}))


if __name__ == '__main__':
    main()
