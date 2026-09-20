"""Build the dense 10 Hz Hanoi dataset (contract hanoi_dense_v5) from the raw recording.

Every row of every episode that is neither a stale nor a repeated image is an
observation. Its label is the chunk of commanded reference poses that follows:
for j = 1..H the row t + 3 j, XYZ from ``reference_pose`` and jaw intent from
``action_abs`` at that row, absolute base-frame metres. Rows past the episode
end repeat the last row and are marked padded. H is 16 (decision 2) or 32 for
the chunk-length comparison run (``--horizon 32``, written to ``dense_v5_h32``). No rows are curated, dropped,
re-weighted or re-labelled; the row distribution is the data.

The build is deterministic from the raw file and these rules, so it is the same
dataset whichever pipeline builds it. When the OpenPI archive exists it is
cross-checked field by field and the result is recorded, never copied over.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from cosmos_policy.datasets.hanoi_joint_data import PROMPT, read_archive, sha256

CONTRACT = 'hanoi_dense_v5_cosmos_v1'
HORIZON = 16
HORIZONS = (16, 32)  # decision 2 and its listed alternative (the chunk-32 comparison run)
FRAMESKIP = 3
REFERENCE_RATE_HZ = 10
EXECUTION_PREFIX = 8
STATIONARY_SPEED_M_PER_S = 0.002
RAW_RATE_HZ = 30
EPISODES = {'train': range(40), 'val': range(40, 45), 'test': range(45, 50)}
DEFAULT_RAW = Path('/scratch/cw5167/datasets/hanoi_wm_roundtrip_20260915_223424_AAAA_to_CCCC.h5')
DEFAULT_METADATA = Path('data/hanoi_cosmos/dense_v5')
DEFAULT_OPENPI_ARCHIVE = Path('/scratch/cw5167/workspace/openpi/data/hanoi/dense_v5_pi05')  # pi0.5 build (30 slots): indices/aaaa_to_cccc_<split>.npz
STATE_COLUMNS = ['joint_0_rad', 'joint_1_rad', 'joint_2_rad', 'joint_3_rad', 'joint_4_rad', 'joint_5_rad', 'jaw_stroke_m']
ACTION_COLUMNS = ['reference_x_m', 'reference_y_m', 'reference_z_m', 'jaw_open_intent']

def deployment_contract(horizon=HORIZON):
    """Section 6 deployment contract for a prepared horizon; the execution prefix (decision 12) does not change with it."""
    if horizon not in HORIZONS:
        raise ValueError(f'Horizon must be one of {HORIZONS}')
    return {
        'version': 5,
        'robot': 'trossen_wxai_single',
        'reference_rate_hz': REFERENCE_RATE_HZ,
        'action_horizon': int(horizon),
        'execution_prefix': EXECUTION_PREFIX,
        'state': STATE_COLUMNS,
        'actions': ACTION_COLUMNS,
        'internal_xyz_encoding': 'absolute',
        'frame': 'commissioned_base_tool_frame',
        'orientation_rpy_rad': [0.0, math.pi / 4, 0.0],
        'rgb_topic': '/camera/camera/color/image_raw',
        'rgb_crop_xywh': [151, 90, 360, 360],
        'max_image_age_s': 0.05,
        'jaw_open_stroke_m': 0.034,
        'jaw_open_duration_s': 1.0,
        'jaw_close_effort_n': -20.0,
        'jaw_close_duration_s': 1.2,
        'jaw_close_settle_s': 0.2,
        'training_observation_alignment': 'every non-stale row; chunk starts three rows after the observation',
        'training_deployment_timing': 'moving observations in training; asynchronous chunk execution at deployment',
        'recording': 'hanoi_wm_roundtrip_20260915_223424',
    }


DEPLOYMENT_CONTRACT = deployment_contract()


def default_metadata(horizon=HORIZON):
    """Prepared dataset directory: dense_v5 for the guide's horizon, dense_v5_h<H> for a comparison horizon."""
    if horizon not in HORIZONS:
        raise ValueError(f'Horizon must be one of {HORIZONS}')
    return DEFAULT_METADATA if horizon == HORIZON else DEFAULT_METADATA.with_name(f'{DEFAULT_METADATA.name}_h{horizon}')


def finite_difference_speed(xyz, offsets, lengths):
    """Measured speed (m/s) from the 30 Hz measured XYZ; the first row of an episode copies the second."""
    speed = np.zeros(len(xyz), np.float32)
    for o, l in zip(offsets, lengths):
        d = np.linalg.norm(np.diff(xyz[o:o + l], axis=0), axis=1) * RAW_RATE_HZ
        speed[o + 1:o + l] = d
        speed[o] = d[0]
    return speed


def build_split(handle, episodes, horizon=HORIZON, frameskip=FRAMESKIP):
    if horizon not in HORIZONS or frameskip != FRAMESKIP:
        raise ValueError(f'Chunk geometry must be a horizon in {HORIZONS} at frameskip {FRAMESKIP}')
    offsets, lengths = handle['ep_offset'][:], handle['ep_len'][:]
    if not bool((handle['episode_success'][:] == 1).all()):
        raise ValueError('Every committed episode must be successful')
    stale, repeated = handle['image_stale'][:] > 0, handle['image_repeated'][:] > 0
    joints, proprio = handle['joint_positions'][:], handle['proprio'][:]
    reference, action_abs = handle['reference_pose'][:], handle['action_abs'][:]
    speed = finite_difference_speed(proprio[:, :3], offsets, lengths)
    rows, targets, pads, ep_index, bounds = [], [], [], [], []
    for e in episodes:
        lo, hi = int(offsets[e]), int(offsets[e] + lengths[e])
        t = np.arange(lo, hi)[~(stale[lo:hi] | repeated[lo:hi])]
        r = t[:, None] + frameskip * np.arange(1, horizon + 1)[None, :]
        pad = r > hi - 1
        rows.append(t); targets.append(np.minimum(r, hi - 1)); pads.append(pad)
        ep_index.append(np.full(len(t), e, np.int64)); bounds.append(np.tile([lo, hi], (len(t), 1)))
    rows, targets, pads = np.concatenate(rows), np.concatenate(targets), np.concatenate(pads)
    states = np.concatenate([joints[rows], proprio[rows, 6:7]], -1).astype(np.float32)
    actions = np.concatenate([reference[targets][..., :3], action_abs[targets][..., 3:4]], -1).astype(np.float32)
    for name, value in (('states', states), ('actions', actions)):
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite {name}')
    if not np.isin(actions[..., 3], [0, 1]).all():
        raise ValueError('Jaw labels must be absolute binary intent')
    return {
        'source_observation_indices': rows.astype(np.int64), 'states': states, 'actions': actions,
        'actions_is_pad': pads, 'source_action_indices': targets.astype(np.int64),
        'episode_indices': np.concatenate(ep_index), 'source_episode_bounds': np.concatenate(bounds).astype(np.int64),
        'cartesian_positions': proprio[rows, :3].astype(np.float32),  # audit and evaluation only, never a model input
        'measured_speed_m_per_s': speed[rows], 'stationary': speed[rows] < STATIONARY_SPEED_M_PER_S,
        'state_columns': np.array(STATE_COLUMNS), 'action_columns': np.array(ACTION_COLUMNS), 'validated': np.array(True),
    }


def fit_statistics(arrays):
    """Training-split normalisation: Cosmos min/max plus mean/std, actions over valid chunk slots only."""
    valid = arrays['actions'][~arrays['actions_is_pad']]
    statistics = {}
    for name, data in (('proprio', arrays['states']), ('actions', valid)):
        data = data.astype(np.float64)  # axis-0 reductions over millions of float32 rows accumulate visible error
        lower, upper = data.min(0), data.max(0)
        constant = upper <= lower
        lower, upper = lower.copy(), upper.copy()
        lower[constant] -= 1e-6
        upper[constant] += 1e-6
        statistics[f'{name}_min'], statistics[f'{name}_max'] = lower.tolist(), upper.tolist()
        statistics[f'{name}_mean'], statistics[f'{name}_std'] = data.mean(0).tolist(), data.std(0).tolist()
    statistics['actions_valid_slots'] = int(len(valid))
    return statistics


def audit(handle):
    """Section 3 of the dense guide: alignment, jaw flips, stationary fraction, stale/repeated counts."""
    offsets, lengths = handle['ep_offset'][:], handle['ep_len'][:]
    reference, action_abs, proprio = handle['reference_pose'][:], handle['action_abs'][:], handle['proprio'][:]
    gripper = handle['gripper_command_issued'][:] > 0
    stale, repeated = handle['image_stale'][:] > 0, handle['image_repeated'][:] > 0
    next_residual, same_residual = [], []
    for o, l in zip(offsets, lengths):
        next_residual.append(np.linalg.norm(action_abs[o:o + l - 1, :3] - reference[o + 1:o + l, :3], axis=1) * 1000)
        same_residual.append(np.linalg.norm(action_abs[o:o + l, :3] - reference[o:o + l, :3], axis=1) * 1000)
    next_residual, same_residual = np.concatenate(next_residual), np.concatenate(same_residual)
    flips = np.concatenate([np.flatnonzero(np.diff(action_abs[o:o + l, 3]) != 0) + 1 + o for o, l in zip(offsets, lengths)])
    commands = np.flatnonzero(gripper)
    speed = finite_difference_speed(proprio[:, :3], offsets, lengths)
    sdk_speed = np.linalg.norm(proprio[:, 3:6], axis=1)
    per_episode = [{'episode': int(i), 'stale': int(stale[o:o + l].sum()), 'repeated': int(repeated[o:o + l].sum()),
                    'observations': int((~(stale[o:o + l] | repeated[o:o + l])).sum()),
                    'stationary_fraction': float((speed[o:o + l] < STATIONARY_SPEED_M_PER_S).mean())}
                   for i, (o, l) in enumerate(zip(offsets, lengths))]
    return {
        'action_abs_vs_next_reference_mm': {'mean': float(next_residual.mean()), 'p50': float(np.median(next_residual)),
                                            'p95': float(np.percentile(next_residual, 95)), 'max': float(next_residual.max()),
                                            'fraction_over_1mm': float((next_residual > 1).mean())},
        'action_abs_vs_same_row_reference_mm': {'mean': float(same_residual.mean()), 'max': float(same_residual.max())},
        'alignment_note': 'action_abs[t] equals reference_pose[t] exactly (attribute action_abs_alignment=post_action_reference); '
                          'the residual against row t+1 is one 30 Hz tick of commanded motion, not a misalignment',
        'jaw': {'flips': int(len(flips)), 'gripper_commands': int(len(commands)), 'flips_on_command_rows': int(np.isin(flips, commands).sum())},
        'stationary': {'definition': f'finite-difference speed of measured XYZ at {RAW_RATE_HZ} Hz below {STATIONARY_SPEED_M_PER_S} m/s',
                       'fraction_all_rows': float((speed < STATIONARY_SPEED_M_PER_S).mean()),
                       'fraction_sdk_velocity_field': float((sdk_speed < STATIONARY_SPEED_M_PER_S).mean())},
        'images': {'stale': int(stale.sum()), 'repeated': int(repeated.sum()), 'excluded': int((stale | repeated).sum()),
                   'observations': int((~(stale | repeated)).sum())},
        'per_episode': per_episode,
    }


def cross_check_openpi(source, splits):
    """Compare the OpenPI dense archive, if present, with this build; record, never adopt."""
    source = Path(source)
    if not source.exists():
        return {'path': str(source), 'status': 'absent'}
    report = {'path': str(source), 'status': 'checked', 'splits': {}}
    for split, ours in splits.items():
        candidates = sorted(source.rglob(f'*{split}*.npz'))
        if not candidates:
            report['splits'][split] = {'status': 'missing'}
            continue
        path = candidates[0]
        theirs = read_archive(path)
        entry = {'file': str(path), 'sha256': sha256(path), 'keys': sorted(theirs)}
        try:
            k = min(theirs['actions'].shape[1], ours['actions'].shape[1])  # the slots both chunk lengths share
            checks = {
                'rows': np.array_equal(theirs['source_observation_indices'], ours['source_observation_indices']),
                'states': np.array_equal(theirs['states'], ours['states']),
                f'actions_first_{k}_slots': np.array_equal(theirs['actions'][:, :k], ours['actions'][:, :k]),
                f'pads_first_{k}_slots': np.array_equal(theirs['actions_is_pad'][:, :k], ours['actions_is_pad'][:, :k]),
                f'source_action_indices_first_{k}_slots': np.array_equal(theirs['source_action_indices'][:, :k], ours['source_action_indices'][:, :k]),
            }
            entry.update({'their_horizon': int(theirs['actions'].shape[1]), 'our_horizon': int(ours['actions'].shape[1]), 'compared_slots': int(k)})
            entry['checks'] = {k: bool(v) for k, v in checks.items()}
            entry['status'] = 'match' if all(checks.values()) else 'mismatch'
        except (KeyError, ValueError, IndexError) as error:
            entry['status'] = f'schema_difference: {error!r}'
        report['splits'][split] = entry
    return report


def prepare(raw=DEFAULT_RAW, output=None, openpi_archive=DEFAULT_OPENPI_ARCHIVE, horizon=HORIZON):
    raw, output = Path(raw), Path(default_metadata(horizon) if output is None else output)
    if output.exists():
        raise FileExistsError(f'Refusing to replace an existing dataset identity: {output}')
    print('Checking raw HDF5 SHA256...', flush=True)
    raw_hash = sha256(raw)
    splits, audit_report = {}, None
    with h5py.File(raw, 'r') as handle:
        if int(handle.attrs['schema_version']) != 4 or bool(handle.attrs['dry_run']):
            raise ValueError('Expected real schema-v4 recollection')
        if handle.attrs['direction'] != 'AAAA_to_CCCC' or handle.attrs['action_abs_alignment'] != 'post_action_reference':
            raise ValueError('Wrong direction/action alignment')
        if tuple(handle['pixels'].shape) != (360050, 224, 224, 3):
            raise ValueError('Wrong recollection image shape')
        audit_report = audit(handle)
        for split, episodes in EPISODES.items():
            splits[split] = build_split(handle, episodes, horizon)
            print(f'{split}: {len(splits[split]["states"])} observations from episodes {episodes.start}-{episodes.stop - 1}', flush=True)
    statistics = fit_statistics(splits['train'])
    cross_check = cross_check_openpi(openpi_archive, splits)
    output.mkdir(parents=True)
    try:
        for split, arrays in splits.items():
            np.savez(output / f'{split}.npz', **arrays)
        (output / 'dataset_statistics.json').write_text(json.dumps(statistics, indent=2) + '\n')
        metadata = {
            'contract': CONTRACT, 'format_version': 1, 'prompt': PROMPT,
            'raw_path': str(raw.resolve()), 'raw_sha256': raw_hash,
            'raw_size_bytes': raw.stat().st_size, 'raw_mtime_ns': raw.stat().st_mtime_ns,
            'horizon': int(horizon), 'frameskip': FRAMESKIP, 'reference_rate_hz': REFERENCE_RATE_HZ,
            'observation_rule': 'every row of a successful episode whose image is neither stale nor repeated',
            'label_rule': 'slot j is row t + 3 j: XYZ = reference_pose[row, 0:3], jaw = action_abs[row, 3]; rows past the episode end repeat the last row and are padded',
            'stationary_definition': audit_report['stationary']['definition'],
            'splits': {split: {'sha256': sha256(output / f'{split}.npz'), 'samples': int(len(z['states'])),
                               'episodes': sorted(map(int, set(z['episode_indices']))),
                               'padded_slots': int(z['actions_is_pad'].sum()),
                               'stationary_fraction': float(z['stationary'].mean())} for split, z in splits.items()},
            'statistics_sha256': sha256(output / 'dataset_statistics.json'),
            'normalization': 'Cosmos min/max over the training split (states; actions over valid chunk slots); mean/std recorded; no clipping',
            'state': 'six measured joint angles radians, measured gripper stroke metres; no velocity, no XYZ',
            'actions': f'{horizon} x [absolute reference XYZ metres, jaw intent] at 10 Hz',
            'future_auxiliary': f'raw row min(t + {FRAMESKIP * horizon}, episode end - 1) (the end of the chunk), image and measured joint/jaw labels only; not verified arrival; not conditioning input',
            'value_auxiliary': 'discounted terminal success from the auxiliary raw row, gamma=0.9995 per raw tick',
            'image_preprocessing': 'stored RGB224 deployment crop, never re-cropped, no augmentation; native Wan VAE synthetic latent-slot packing',
            'audit': audit_report, 'openpi_cross_check': cross_check, 'deployment': deployment_contract(horizon),
        }
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    except BaseException:
        (output / 'PREPARATION_FAILED').touch()
        raise
    return metadata


def cross_check_existing(output=DEFAULT_METADATA, openpi_archive=DEFAULT_OPENPI_ARCHIVE):
    """Re-run the OpenPI cross-check against an existing build; written beside metadata.json, which stays untouched."""
    output = Path(output)
    splits = {split: read_archive(output / f'{split}.npz') for split in EPISODES}
    report = cross_check_openpi(openpi_archive, splits)
    (output / 'openpi_cross_check.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--horizon', type=int, default=HORIZON, choices=HORIZONS, help='Chunk length: 16 (decision 2) or 32 (comparison run)')
    parser.add_argument('--output', type=Path, default=None, help='Default: data/hanoi_cosmos/dense_v5 for horizon 16, dense_v5_h<H> otherwise')
    parser.add_argument('--openpi-archive', type=Path, default=DEFAULT_OPENPI_ARCHIVE)
    parser.add_argument('--cross-check-only', action='store_true', help='Only compare an existing build with the OpenPI archive')
    arguments = parser.parse_args()
    output = arguments.output or default_metadata(arguments.horizon)
    if arguments.cross_check_only:
        print(json.dumps(cross_check_existing(output, arguments.openpi_archive), indent=2))
        raise SystemExit(0)
    result = prepare(arguments.raw, output, arguments.openpi_archive, arguments.horizon)
    print(json.dumps({key: value for key, value in result.items() if key not in ('deployment', 'audit')}, indent=2))
    print(json.dumps({key: value for key, value in result['audit'].items() if key != 'per_episode'}, indent=2))
