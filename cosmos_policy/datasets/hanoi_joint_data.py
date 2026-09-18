"""Import the audited joint-state/sparse-Cartesian handover into Cosmos."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

PROMPT = "Move all four rings from peg A to peg C following Tower of Hanoi rules."
CONTRACT = "hanoi_joint_sparse_v3_cosmos_v1"
DEFAULT_HANDOFF = Path('/scratch/cw5167/workspace/openpi/docs/hanoi_cosmos_training_handoff.json')
DEFAULT_METADATA = Path('data/hanoi_cosmos/joint_sparse_v3')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def relative_actions(actions, xyz):
    result = np.asarray(actions, np.float32).copy()
    xyz = np.asarray(xyz, np.float32)
    if result.shape[-2:] != (8, 4) or xyz.shape != result.shape[:-2] + (3,):
        raise ValueError('Expected eight Cartesian/jaw targets and a separate XYZ anchor')
    result[..., :3] -= xyz[..., None, :]
    return result


def validate_archive(arrays, handle, split):
    """Audit every numeric label against raw rows, including terminal repeats."""
    expected = {'train': (5031, range(40)), 'val': (611, range(40, 45)), 'test': (612, range(45, 50))}
    count, episodes = expected[split]
    rows, targets = arrays['source_observation_indices'], arrays['source_action_indices']
    pads, ep = arrays['actions_is_pad'], arrays['episode_indices']
    if not bool(arrays['validated'].item()) or rows.shape != (count,) or targets.shape != (count, 8):
        raise ValueError(f'Wrong/unvalidated {split} archive')
    if pads.shape != targets.shape or pads.dtype != np.bool_ or set(ep) != set(episodes):
        raise ValueError('Invalid padding or episode split')
    if arrays['states'].shape != (count, 7) or arrays['actions'].shape != (count, 8, 4):
        raise ValueError('Expected seven measured state values and eight four-value targets')
    if len(np.unique(rows)) != count or np.any(np.diff(rows) <= 0):
        raise ValueError('Observation rows must be unique and ordered')
    offsets, lengths = handle['ep_offset'][:], handle['ep_len'][:]
    lo, hi = offsets[ep], offsets[ep] + lengths[ep]
    np.testing.assert_array_equal(arrays['source_episode_bounds'], np.stack([lo, hi], -1))
    if np.any((rows < lo) | (rows >= hi)) or np.any((targets < lo[:, None]) | (targets >= hi[:, None])):
        raise ValueError('Observation/target crosses source episode boundary')
    if np.any(targets[:, 0] < rows) or np.any(np.diff(targets, axis=1) < 0) or pads[:, 0].any():
        raise ValueError('Target order or first-target padding is invalid')
    expected_pad = np.zeros_like(pads)
    expected_pad[:, 1:] = np.diff(targets, axis=1) == 0
    np.testing.assert_array_equal(pads, expected_pad)
    if np.any(targets[pads] != np.broadcast_to((hi - 1)[:, None], targets.shape)[pads]):
        raise ValueError('Padding must repeat the final episode target')
    joints, proprio, raw_actions = handle['joint_positions'][:], handle['proprio'][:], handle['action_abs'][:]
    np.testing.assert_array_equal(arrays['states'], np.concatenate([joints[rows], proprio[rows, 6:7]], -1))
    np.testing.assert_array_equal(arrays['cartesian_positions'], proprio[rows, :3])
    np.testing.assert_array_equal(arrays['actions'], raw_actions[targets])
    for key in ('states', 'cartesian_positions', 'actions'):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f'Non-finite {key}')
    age = handle['command_monotonic_ns'][:][rows] - handle['image_receipt_monotonic_ns'][:][rows]
    if np.any((age < 0) | (age > 50_000_000)):
        raise ValueError('Admitted observation is not fresh')
    if not np.isin(arrays['actions'][..., 3], [0, 1]).all():
        raise ValueError('Jaw labels must be absolute binary intent')
    # Auxiliary future observation is explicitly a raw snapshot after the final
    # selected command index (clamped at terminal). It is NOT an arrival label,
    # and never enters the three conditioning slots used for action prediction.
    future_rows = np.minimum(targets[:, -1] + 1, hi - 1)
    future_states = np.concatenate([joints[future_rows], proprio[future_rows, 6:7]], -1)
    if not np.isfinite(future_states).all():
        raise ValueError('Non-finite auxiliary future state')
    return future_rows, future_states


def fit_statistics(arrays, future_states):
    """Native Cosmos min/max, fit using training examples/auxiliary labels only."""
    statistics = {}
    values = {'actions': relative_actions(arrays['actions'], arrays['cartesian_positions']).reshape(-1, 4),
              'proprio': np.concatenate([arrays['states'], future_states], axis=0)}
    for name, data in values.items():
        lower, upper = data.min(0), data.max(0)
        constant = upper <= lower
        lower, upper = lower.copy(), upper.copy()
        lower[constant] -= 1e-6
        upper[constant] += 1e-6
        statistics[name + '_min'] = lower.tolist()
        statistics[name + '_max'] = upper.tolist()
    return statistics


def prepare(handoff=DEFAULT_HANDOFF, output=DEFAULT_METADATA):
    handoff, output = Path(handoff), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to replace an existing dataset identity: {output}')
    specification = json.loads(handoff.read_text())
    source = Path(specification['source']['hdf5'])
    if specification['prompt'] != PROMPT:
        raise ValueError('Handover prompt changed')
    if source.stat().st_size != specification['source']['hdf5_bytes']:
        raise ValueError('Raw file size differs from handover')
    print('Checking raw HDF5 SHA256...', flush=True)
    source_hash = sha256(source)
    if source_hash != specification['source']['hdf5_sha256']:
        raise ValueError('Raw HDF5 hash differs from handover')
    prepared = specification['prepared_data']
    for path, expected in [(specification['source']['manifest'], specification['source']['manifest_sha256']),
                           (prepared['audit'], prepared['audit_sha256'])]:
        if sha256(path) != expected:
            raise ValueError(f'Handover artifact changed: {path}')
    splits, auxiliary = {}, {}
    with h5py.File(source, 'r') as handle:
        if int(handle.attrs['schema_version']) != 4 or bool(handle.attrs['dry_run']):
            raise ValueError('Expected real schema-v4 recollection')
        if handle.attrs['direction'] != 'AAAA_to_CCCC' or handle.attrs['action_abs_alignment'] != 'post_action_reference':
            raise ValueError('Wrong direction/action alignment')
        if tuple(handle['pixels'].shape) != (360050, 224, 224, 3):
            raise ValueError('Wrong recollection image shape')
        for split in ('train', 'val', 'test'):
            item = prepared['archives'][split]
            if sha256(item['path']) != item['sha256']:
                raise ValueError(f'{split} index hash differs from handover')
            splits[split] = read_archive(item['path'])
            auxiliary[split] = validate_archive(splits[split], handle, split)
            print(f'{split}: all {len(splits[split]["states"])} states/contexts/targets match raw source', flush=True)
    stats = fit_statistics(splits['train'], auxiliary['train'][1])
    output.mkdir(parents=True)
    try:
        for split in splits:
            shutil.copyfile(prepared['archives'][split]['path'], output / f'{split}.npz')
        (output / 'dataset_statistics.json').write_text(json.dumps(stats, indent=2) + '\n')
        metadata = {
            'contract': CONTRACT, 'format_version': 1, 'prompt': PROMPT,
            'raw_path': str(source.resolve()), 'raw_sha256': source_hash,
            'raw_size_bytes': source.stat().st_size, 'raw_mtime_ns': source.stat().st_mtime_ns,
            'handoff_path': str(handoff.resolve()), 'handoff_sha256': sha256(handoff),
            'splits': {split: {'sha256': sha256(output / f'{split}.npz'), 'samples': len(z['states']),
                               'episodes': sorted(map(int, set(z['episode_indices']))),
                               'padding_slots': int(z['actions_is_pad'].sum())} for split, z in splits.items()},
            'statistics_sha256': sha256(output / 'dataset_statistics.json'),
            'normalization': 'Cosmos min/max; training states plus training auxiliary states; all training action slots including terminal repeats; no clipping',
            'state': 'six measured joint angles radians, measured gripper stroke metres; no XYZ or velocity',
            'actions': '8 x [XYZ relative to one measured observation XYZ, absolute jaw intent]',
            'future_auxiliary': 'raw min(last selected target row + 1, episode end - 1), image and measured joint/jaw labels only; not verified arrival; not conditioning input',
            'value_auxiliary': 'discounted terminal success from auxiliary raw row, gamma=0.9995 per raw tick',
            'image_preprocessing': 'stored RGB224, no second crop, no random augmentation; native Wan VAE synthetic latent-slot packing',
            'deployment': specification['contract'],
        }
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    except BaseException:
        # Keep evidence instead of exposing a seemingly complete preparation.
        (output / 'PREPARATION_FAILED').touch()
        raise
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument('--output', type=Path, default=DEFAULT_METADATA)
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.handoff, arguments.output), indent=2))
