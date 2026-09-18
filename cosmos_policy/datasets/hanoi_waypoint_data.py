"""Import the OpenPI waypoint_v4 (recorded leg endpoint) labels into Cosmos.

The joint_v3 labels were Ramer-Douglas-Peucker path points, so near-identical
situations received different "next destination" labels. waypoint_v4 labels are
the recording's own motion-leg endpoints plus gripper events. The observation
contract, image crop, episode splits, horizon (8) and execution prefix (1) are
unchanged; only the target rows differ. This importer never touches the joint_v3
dataset or its modules' behaviour.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from cosmos_policy.datasets.hanoi_joint_data import PROMPT, fit_statistics, read_archive, sha256

CONTRACT = 'hanoi_waypoint_v4_cosmos_v1'
TARGET_EXTRACTION = 'recorded_leg_endpoints_and_gripper_events_merge_arrive_then_grip'
LABEL_ONLY_CONTRACT_FIELDS = {'version', 'target_extraction', 'intermediate_motion_noise',
                              'recorded_path_deviation_budget_m'}
EPISODES = {'train': range(40), 'val': range(40, 45), 'test': range(45, 50)}
DEFAULT_AUDIT = Path('/scratch/cw5167/workspace/openpi/data/hanoi/waypoint_v4/audit.json')
DEFAULT_HANDOFF = Path('/scratch/cw5167/workspace/openpi/docs/hanoi_cosmos_training_handoff.json')
DEFAULT_ISSUE_DOC = Path('/scratch/cw5167/workspace/openpi/docs/hanoi_label_issue_for_cosmos_20260917.md')
DEFAULT_METADATA = Path('data/hanoi_cosmos/waypoint_v4')


def validate_archive(arrays, handle, split, count):
    """The joint_v3 audit of every numeric label against raw rows, with the count from the audit."""
    episodes = EPISODES[split]
    rows, targets = arrays['source_observation_indices'], arrays['source_action_indices']
    pads, ep = arrays['actions_is_pad'], arrays['episode_indices']
    if count < 1 or not bool(arrays['validated'].item()) or rows.shape != (count,) or targets.shape != (count, 8):
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
    # Auxiliary future observation: raw snapshot after the final selected
    # command index (clamped at terminal). Not an arrival label; never conditioning.
    future_rows = np.minimum(targets[:, -1] + 1, hi - 1)
    future_states = np.concatenate([joints[future_rows], proprio[future_rows, 6:7]], -1)
    if not np.isfinite(future_states).all():
        raise ValueError('Non-finite auxiliary future state')
    return future_rows, future_states


def check_audit(record, audit_sha256, verification, specification):
    """Reject anything but consistent v4 labels over the audited joint_v3 recording."""
    consistency = record['label_consistency']
    if not consistency['passed'] or consistency['fraction_over_10_mm'] != 0 or consistency['split'] != 'train':
        raise ValueError('waypoint_v4 labels did not pass the similar-situation consistency audit')
    if not record['admission']['passed']:
        raise ValueError('waypoint_v4 labels were not admitted for training')
    source = specification['source']
    if record['source'] != source['hdf5'] or record['source_sha256'] != source['hdf5_sha256']:
        raise ValueError('waypoint_v4 labels come from a different raw recording than the handover')
    if verification['audit_sha256'] != audit_sha256:
        raise ValueError('Audit content differs from its verification record')
    contract = record['contract']
    if contract.get('version') != 4 or contract.get('target_extraction') != TARGET_EXTRACTION:
        raise ValueError('Expected the recorded-leg-endpoint label contract, version 4')
    changed = {key for key in set(contract) | set(specification['contract'])
               if contract.get(key) != specification['contract'].get(key)}
    if changed - LABEL_ONLY_CONTRACT_FIELDS:
        raise ValueError(f'Observation/action contract changed beyond label extraction: {sorted(changed)}')
    if specification['prompt'] != PROMPT:
        raise ValueError('Handover prompt changed')


def prepare(audit=DEFAULT_AUDIT, handoff=DEFAULT_HANDOFF, output=DEFAULT_METADATA, issue_doc=DEFAULT_ISSUE_DOC):
    audit, handoff, output, issue_doc = Path(audit), Path(handoff), Path(output), Path(issue_doc)
    if output.exists():
        raise FileExistsError(f'Refusing to replace an existing dataset identity: {output}')
    record = json.loads(audit.read_text())
    audit_sha256 = sha256(audit)
    verification = json.loads((audit.parent / 'verification.json').read_text())
    specification = json.loads(handoff.read_text())
    check_audit(record, audit_sha256, verification, specification)
    source = Path(specification['source']['hdf5'])
    if source.stat().st_size != specification['source']['hdf5_bytes']:
        raise ValueError('Raw file size differs from handover')
    if sha256(specification['source']['manifest']) != specification['source']['manifest_sha256']:
        raise ValueError('Raw manifest changed')
    print('Checking raw HDF5 SHA256...', flush=True)
    source_hash = sha256(source)
    if source_hash != specification['source']['hdf5_sha256']:
        raise ValueError('Raw HDF5 hash differs from handover')
    archives = {}
    for split in ('train', 'val', 'test'):
        item = record['archives'][f'aaaa_to_cccc_{split}']
        path = audit.parent / 'indices' / f'aaaa_to_cccc_{split}.npz'
        if sha256(path) != item['sha256']:
            raise ValueError(f'{split} archive hash differs from the audit')
        archives[split] = {'path': path, 'sha256': item['sha256'], 'samples': int(item['samples'])}
    splits, auxiliary = {}, {}
    with h5py.File(source, 'r') as handle:
        if int(handle.attrs['schema_version']) != 4 or bool(handle.attrs['dry_run']):
            raise ValueError('Expected real schema-v4 recollection')
        if handle.attrs['direction'] != 'AAAA_to_CCCC' or handle.attrs['action_abs_alignment'] != 'post_action_reference':
            raise ValueError('Wrong direction/action alignment')
        if tuple(handle['pixels'].shape) != (360050, 224, 224, 3):
            raise ValueError('Wrong recollection image shape')
        for split, item in archives.items():
            splits[split] = read_archive(item['path'])
            auxiliary[split] = validate_archive(splits[split], handle, split, item['samples'])
            print(f'{split}: all {item["samples"]} states/contexts/targets match raw source', flush=True)
    stats = fit_statistics(splits['train'], auxiliary['train'][1])
    output.mkdir(parents=True)
    try:
        for split, item in archives.items():
            shutil.copyfile(item['path'], output / f'{split}.npz')
        (output / 'dataset_statistics.json').write_text(json.dumps(stats, indent=2) + '\n')
        metadata = {
            'contract': CONTRACT, 'format_version': 1, 'prompt': PROMPT,
            'raw_path': str(source.resolve()), 'raw_sha256': source_hash,
            'raw_size_bytes': source.stat().st_size, 'raw_mtime_ns': source.stat().st_mtime_ns,
            'handoff_path': str(handoff.resolve()), 'handoff_sha256': sha256(handoff),
            'label_audit_path': str(audit.resolve()), 'label_audit_sha256': audit_sha256,
            'label_issue_doc': str(issue_doc.resolve()) if issue_doc.exists() else None,
            'label_consistency': record['label_consistency'],
            'label_extraction': TARGET_EXTRACTION, 'label_settings': record['settings'],
            'splits': {split: {'sha256': sha256(output / f'{split}.npz'), 'samples': len(z['states']),
                               'episodes': sorted(map(int, set(z['episode_indices']))),
                               'padding_slots': int(z['actions_is_pad'].sum())} for split, z in splits.items()},
            'statistics_sha256': sha256(output / 'dataset_statistics.json'),
            'normalization': 'Cosmos min/max; training states plus training auxiliary states; all training action slots including terminal repeats; no clipping',
            'state': 'six measured joint angles radians, measured gripper stroke metres; no XYZ or velocity',
            'actions': '8 x [XYZ relative to one measured observation XYZ, absolute jaw intent]; recorded leg endpoints and gripper events',
            'future_auxiliary': 'raw min(last selected target row + 1, episode end - 1), image and measured joint/jaw labels only; not verified arrival; not conditioning input',
            'value_auxiliary': 'discounted terminal success from auxiliary raw row, gamma=0.9995 per raw tick',
            'image_preprocessing': 'stored RGB224, no second crop, no random augmentation; native Wan VAE synthetic latent-slot packing',
            'deployment': record['contract'],
        }
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    except BaseException:
        (output / 'PREPARATION_FAILED').touch()
        raise
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--handoff', type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument('--output', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--issue-doc', type=Path, default=DEFAULT_ISSUE_DOC)
    arguments = parser.parse_args()
    result = prepare(arguments.audit, arguments.handoff, arguments.output, arguments.issue_doc)
    print(json.dumps({key: value for key, value in result.items() if key != 'deployment'}, indent=2))
