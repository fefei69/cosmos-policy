"""One-move expert dataset (contract hanoi_expert6_h1): the first move of each of the six tower tasks.

Six cases, ring 1 off a full stack (AAAA to BAAA or CAAA, BBBB to ABBB or CBBB,
CCCC to ACCC or BCCC), taken from the six expert recordings of the six-task
run with that run's episode split (episodes 0-7 train, 8 validation, 9 test):
48 / 6 / 6 one-move episodes. An episode is the whole first move, motion
stages open to retreat. The labels are the play rules with a horizon of one
move: the goal of every row is the board after the move, conditioned on
through that board's sentence, and the dense v5 chunk is cut at the move's
last row and padded with it, so the policy learns to stop when the goal is
reached. Rows are the six-task observation rows (image neither stale nor
repeated; telemetry is finite throughout these recordings).

Every case is demonstrated eight times, so this is not a composition test: it
measures whether the goal-sentence interface works on expert data, the
execution control for the play run. The same rows exist in the V-JEPA side's
``expert6_first_move`` archive; ``prepare`` compares the two row for row and
records the result, never adopting the other archive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cosmos_policy.datasets.hanoi_dense_data import FRAMESKIP, HORIZON, HORIZONS, REFERENCE_RATE_HZ, STATIONARY_SPEED_M_PER_S, fit_statistics
from cosmos_policy.datasets.hanoi_joint_data import read_archive, sha256
from cosmos_policy.datasets.hanoi_multitask_data import EPISODE_SPLIT, TASKS, check_raw
from cosmos_policy.datasets.hanoi_play_data import (
    BOARD_INDEX, BOARDS, DEPLOYMENT_CONTRACT_V7, PROMPT_TEMPLATE, PROMPTS, PROMPTS_SHA256, board_string, build_span, concatenate,
    label_audit, load_columns,
)

CONTRACT_H1 = 'hanoi_expert6_h1_cosmos_v1'
DEFAULT_METADATA = Path('data/hanoi_cosmos/expert6_h1')
DEFAULT_REFERENCE = Path('/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/data/expert6_first_move')
H1_FILE_ROLES = tuple(task.direction for task in TASKS)  # file index = task index
MOVE_STAGES = (1, 9)  # open .. retreat
EPISODE_ID = 'task index * 10000 + expert episode * 100 + move index (0)'
COMPARED = ('source_observation_indices', 'file_indices', 'states', 'actions', 'actions_is_pad', 'source_action_indices',
            'goal_end_rows', 'goal_board_indices', 'board_indices', 'motion_stages')


def first_move_span(columns, episode):
    """Raw rows [lo, hi) of move 0 of an expert episode, stages open to retreat; contiguous by construction of the recorder."""
    lo, hi = int(columns['ep_offset'][episode]), int(columns['ep_offset'][episode] + columns['ep_len'][episode])
    stage, move = columns['motion_stage'][lo:hi], columns['move_idx'][lo:hi]
    rows = np.arange(lo, hi)[(move == 0) & (stage >= MOVE_STAGES[0]) & (stage <= MOVE_STAGES[1])]
    if not len(rows) or not (np.diff(rows) == 1).all():
        raise ValueError(f'Episode {episode}: the first move is not one contiguous block of rows')
    return int(rows[0]), int(rows[-1]) + 1


def build_split(columns_by_task, split, horizon=HORIZON):
    parts, cases = [], {}
    rng = np.random.default_rng(0)  # unused: a one-move span has a single goal
    for task in TASKS:
        columns = columns_by_task[task.index]
        for episode in EPISODE_SPLIT[split]:
            lo, hi = first_move_span(columns, episode)
            usable = np.arange(lo, hi)[columns['usable'][lo:hi]]
            before, after = board_string(columns['board'][int(usable[0])]), board_string(columns['board'][int(usable[-1])])
            if before != task.start * 4 or sum(a != b for a, b in zip(before, after)) != 1 or after[0] == before[0]:
                raise ValueError(f'{task.direction} episode {episode}: first move is not ring 1 off the full stack ({before} to {after})')
            part = build_span(columns, task.index, lo, hi, task.index * 10000 + episode * 100, rng, horizon=horizon, cap=1,
                              single_goal_board=after, kind='clip')
            cases[f'{before}>{after}'] = cases.get(f'{before}>{after}', 0) + 1
            parts.append(part)
    arrays = concatenate(parts)
    arrays['task_indices'] = arrays['file_indices'].copy()
    return arrays, cases


def cross_check(reference, splits):
    """Compare with the V-JEPA side's expert6_first_move archive, if present; record, never adopt."""
    reference = Path(reference)
    if not (reference / 'metadata.json').exists():
        return {'path': str(reference), 'status': 'absent'}
    report = {'path': str(reference), 'status': 'checked', 'splits': {}}
    for split, ours in splits.items():
        path = reference / f'{split}.npz'
        theirs = read_archive(path)
        entry = {'file': str(path), 'sha256': sha256(path)}
        try:
            order_ours = np.lexsort((ours['source_observation_indices'], ours['file_indices']))
            order_theirs = np.lexsort((theirs['source_observation_indices'], theirs['file_indices']))
            checks = {key: bool(np.array_equal(ours[key][order_ours], theirs[key][order_theirs])) for key in COMPARED}
            checks['prompts'] = list(theirs['prompts']) == list(PROMPTS)
            entry.update({'rows_ours': int(len(ours['states'])), 'rows_theirs': int(len(theirs['states'])), 'checks': checks,
                          'status': 'match' if all(checks.values()) else 'mismatch'})
        except (KeyError, ValueError, IndexError) as error:
            entry['status'] = f'schema_difference: {error!r}'
        report['splits'][split] = entry
    return report


def prepare(output=DEFAULT_METADATA, reference=DEFAULT_REFERENCE, horizon=HORIZON, tasks=TASKS):
    output = Path(output)
    if horizon not in HORIZONS:
        raise ValueError(f'Horizon must be one of {HORIZONS}')
    if output.exists():
        raise FileExistsError(f'Refusing to replace an existing dataset identity: {output}')
    files, columns_by_task = [], {}
    for task in tasks:
        print(f'Hashing {task.file}...', flush=True)
        stat = task.path.stat()
        with h5py.File(task.path, 'r') as handle:
            check_raw(handle, task)
            columns_by_task[task.index] = load_columns(handle)
        files.append({'index': task.index, 'role': task.direction, 'path': str(task.path.resolve()), 'sha256': sha256(task.path),
                      'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'episodes': 10})
    splits, cases = {}, {}
    for split in EPISODE_SPLIT:
        splits[split], cases[split] = build_split(columns_by_task, split, horizon)
        print(f'{split}: {len(splits[split]["states"])} rows, cases {cases[split]}', flush=True)
    statistics = fit_statistics(splits['train'])
    comparison = cross_check(reference, splits)
    output.mkdir(parents=True)
    try:
        for split, arrays in splits.items():
            np.savez(output / f'{split}.npz', **arrays)
        (output / 'dataset_statistics.json').write_text(json.dumps(statistics, indent=2) + '\n')
        metadata = {
            'contract': CONTRACT_H1, 'format_version': 1,
            'horizon': int(horizon), 'frameskip': FRAMESKIP, 'reference_rate_hz': REFERENCE_RATE_HZ,
            'horizon_cap_moves': 1, 'goal_seed': 0,
            'prompt_template': PROMPT_TEMPLATE, 'prompts_sha256': PROMPTS_SHA256, 'prompts': list(PROMPTS), 'boards': list(BOARDS),
            'files': files,
            'episode_split': {split: list(map(int, episodes)) for split, episodes in EPISODE_SPLIT.items()},
            'global_episode_id': EPISODE_ID,
            'cases': cases,
            'case_sentences': {case: PROMPTS[BOARD_INDEX[case.split('>')[1]]] for case in sorted(cases['train'])},
            'observation_rule': 'every row of the first move (stages open to retreat) whose image is neither stale nor repeated and whose telemetry is finite',
            'goal_rule': 'the goal of every row is the board after the move (the board at the move\'s last usable row); one move, no sampling',
            'label_rule': ('slot j is row t + 3 j: XYZ = reference_pose[row, 0:3], jaw = action_abs[row, 3]; slots past the last row of the move '
                           'repeat that row and are padded (trained as hold targets, excluded from metrics)'),
            'conditioning': 'the goal board sentence (T5-11B embedding, one of the 81 play sentences) is the only goal signal; no goal image, no task id',
            'stationary_definition': f'finite-difference speed of measured XYZ at 30 Hz below {STATIONARY_SPEED_M_PER_S} m/s',
            'splits': {split: {'sha256': sha256(output / f'{split}.npz'), 'samples': int(len(z['states'])),
                               'episodes': sorted(map(int, set(z['episode_indices']))), 'rows_by_kind': {'walk': 0, 'crop': 0, 'clip': int(len(z['states']))},
                               'moves': int(len(set(z['episode_indices']))),
                               'stationary_fraction': float(z['stationary'].mean()), 'audit': label_audit(z)} for split, z in splits.items()},
            'statistics_sha256': sha256(output / 'dataset_statistics.json'),
            'normalization': 'Cosmos min/max over the training split (states; actions over valid chunk slots); mean/std recorded; no clipping',
            'state': 'six measured joint angles radians, measured gripper stroke metres; no velocity, no XYZ',
            'actions': f'{horizon} x [absolute reference XYZ metres, jaw intent] at 10 Hz',
            'future_auxiliary': f'raw row min(t + {FRAMESKIP * horizon}, last row of the move) of the same file, image and measured joint/jaw labels only; not conditioning input',
            'value_auxiliary': 'discounted goal arrival from the auxiliary raw row, gamma=0.9995 per raw tick, horizon = the move end',
            'image_preprocessing': 'stored RGB224 deployment crop, never re-cropped, no augmentation; native Wan VAE synthetic latent-slot packing',
            'reference_cross_check': comparison,
            'deployment': {**DEPLOYMENT_CONTRACT_V7, 'recording': 'first move of the six expert recordings (hanoi_wm_roundtrip_20260925_171442, 20260926_011840)'},
        }
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    except BaseException:
        (output / 'PREPARATION_FAILED').touch()
        raise
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--horizon', type=int, default=HORIZON, choices=HORIZONS)
    arguments = parser.parse_args()
    result = prepare(arguments.output, arguments.reference, arguments.horizon)
    print(json.dumps({key: value for key, value in result.items() if key not in ('deployment', 'prompts', 'boards')}, indent=2))
