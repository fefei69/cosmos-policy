"""Six-task dense 10 Hz Hanoi dataset (contract hanoi_multitask_v6) from the September 25/26 recordings.

Six directed tower moves (A->C, C->A, A->B, B->A, B->C, C->B), ten
successful episodes each, one HDF5 file per task. The label rule is the dense
v5 one (every non-stale, non-repeated row is an observation; slot j of the
chunk is row t + 3 j of the commanded reference, absolute XYZ plus jaw
intent, padded past the episode end). New here: each row carries its task
index, and the task's fixed instruction is the conditioning prompt. Episodes
0-7 of every file train, episode 8 validates, episode 9 tests. Nothing is
curated, dropped or re-weighted.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from cosmos_policy.datasets.hanoi_dense_data import (
    DEPLOYMENT_CONTRACT, FRAMESKIP, HORIZON, HORIZONS, REFERENCE_RATE_HZ, STATE_COLUMNS, ACTION_COLUMNS,
    audit, build_split, fit_statistics,
)
from cosmos_policy.datasets.hanoi_joint_data import sha256

CONTRACT = 'hanoi_multitask_v6_cosmos_v1'
RAW_ROOT = Path('/scratch/cw5167/datasets')
DEFAULT_METADATA = Path('data/hanoi_cosmos/multitask_v6')
DEFAULT_EMBEDDINGS = Path('data/hanoi_cosmos/t5_embeddings_multitask.pkl')
EPISODE_SPLIT = {'train': tuple(range(8)), 'val': (8,), 'test': (9,)}  # per file; ten episodes each
PEG_Y = {'A': -0.057, 'B': 0.015, 'C': 0.085}  # commanded release y per peg, metres (identical in all six files)
PEG_POSITION = {'A': 'the left peg', 'B': 'the middle peg', 'C': 'the right peg'}


def prompt_for(start: str, goal: str) -> str:
    """Peg letters early, the goal stated twice with its position, so the task occupies several token positions."""
    return (f'Move all four rings from peg {start} to peg {goal} following Tower of Hanoi rules. '
            f'The goal is peg {goal}, {PEG_POSITION[goal]}.')


@dataclass(frozen=True)
class Task:
    index: int
    direction: str
    start: str
    goal: str
    first_move_target: str  # peg the smallest ring goes to on move 0 (from the recordings)
    file: str

    @property
    def prompt(self) -> str:
        return prompt_for(self.start, self.goal)

    @property
    def path(self) -> Path:
        return RAW_ROOT / self.file


TASKS = (
    Task(0, 'AAAA_to_CCCC', 'A', 'C', 'B', 'hanoi_wm_roundtrip_20260925_171442_AAAA_to_CCCC.h5'),
    Task(1, 'CCCC_to_AAAA', 'C', 'A', 'B', 'hanoi_wm_roundtrip_20260925_171442_CCCC_to_AAAA.h5'),
    Task(2, 'AAAA_to_BBBB', 'A', 'B', 'C', 'hanoi_wm_roundtrip_20260926_011840_AAAA_to_BBBB.h5'),
    Task(3, 'BBBB_to_AAAA', 'B', 'A', 'C', 'hanoi_wm_roundtrip_20260926_011840_BBBB_to_AAAA.h5'),
    Task(4, 'BBBB_to_CCCC', 'B', 'C', 'A', 'hanoi_wm_roundtrip_20260926_011840_BBBB_to_CCCC.h5'),
    Task(5, 'CCCC_to_BBBB', 'C', 'B', 'A', 'hanoi_wm_roundtrip_20260926_011840_CCCC_to_BBBB.h5'),
)
PROMPTS = tuple(task.prompt for task in TASKS)
TASK_BY_PROMPT = {task.prompt: task for task in TASKS}
TASK_BY_DIRECTION = {task.direction: task for task in TASKS}
SAME_START_PAIRS = tuple((a.index, b.index) for a in TASKS for b in TASKS if a.index < b.index and a.start == b.start)
REVERSE_OF = {task.index: TASK_BY_DIRECTION[f'{task.goal * 4}_to_{task.start * 4}'].index for task in TASKS}
GLOBAL_EPISODE_STRIDE = 100  # global episode id = task index * 100 + episode index within its file

DEPLOYMENT_CONTRACT_V6 = {
    **DEPLOYMENT_CONTRACT,
    'version': 6,
    'tasks': [{'index': t.index, 'direction': t.direction, 'start_peg': t.start, 'goal_peg': t.goal, 'prompt': t.prompt} for t in TASKS],
    'prompt_rule': 'the request must carry one of the six prompts verbatim; there is no default task',
    'recording': 'hanoi_wm_roundtrip_20260925_171442 and 20260926_011840 (velocity-noise-v1 motion)',
}


def check_raw(handle, task: Task):
    """The file must be the real schema-v4 recording of this task with ten successful episodes."""
    attrs = handle.attrs
    if int(attrs['schema_version']) != 4 or bool(attrs['dry_run']):
        raise ValueError(f'{task.file}: expected real schema-v4 recollection')
    if attrs['direction'] != task.direction or attrs['action_abs_alignment'] != 'post_action_reference':
        raise ValueError(f'{task.file}: wrong direction/action alignment')
    lengths = handle['ep_len'][:]
    if len(lengths) != 10 or int(lengths.sum()) != handle['pixels'].shape[0] or tuple(handle['pixels'].shape[1:]) != (224, 224, 3):
        raise ValueError(f'{task.file}: expected ten episodes of RGB224 rows')
    if not bool((handle['episode_success'][:] == 1).all()):
        raise ValueError(f'{task.file}: every episode must be successful')
    goal = np.unique(handle['goal_board'][:], axis=0)
    if goal.shape != (1, 4) or set(goal[0].tolist()) != {'ABC'.index(task.goal)}:
        raise ValueError(f'{task.file}: goal board differs from the task table')


def build_task_split(handle, task: Task, episodes, horizon=HORIZON):
    arrays = build_split(handle, list(episodes), horizon)
    n = len(arrays['states'])
    arrays['task_indices'] = np.full(n, task.index, np.int64)
    arrays['file_indices'] = np.full(n, task.index, np.int64)  # one file per task; kept separate for clarity
    arrays['episode_indices'] = arrays['episode_indices'] + GLOBAL_EPISODE_STRIDE * task.index
    return arrays


def concatenate(parts):
    keys = parts[0].keys()
    out = {}
    for key in keys:
        values = [p[key] for p in parts]
        if values[0].ndim == 0 or key in ('state_columns', 'action_columns', 'validated'):
            for v in values[1:]:
                if not np.array_equal(v, values[0]):
                    raise ValueError(f'Inconsistent {key} across files')
            out[key] = values[0]
        else:
            out[key] = np.concatenate(values)
    return out


def prepare(output=DEFAULT_METADATA, horizon=HORIZON):
    output = Path(output)
    if horizon not in HORIZONS:
        raise ValueError(f'Horizon must be one of {HORIZONS}')
    if output.exists():
        raise FileExistsError(f'Refusing to replace an existing dataset identity: {output}')
    files, audits, parts = [], {}, {split: [] for split in EPISODE_SPLIT}
    for task in TASKS:
        print(f'Hashing {task.file}...', flush=True)
        digest = sha256(task.path)
        with h5py.File(task.path, 'r') as handle:
            check_raw(handle, task)
            audits[task.direction] = audit(handle)
            for split, episodes in EPISODE_SPLIT.items():
                arrays = build_task_split(handle, task, episodes, horizon)
                parts[split].append(arrays)
                print(f'  {task.direction} {split}: {len(arrays["states"])} observations from episodes {list(episodes)}', flush=True)
        stat = task.path.stat()
        files.append({'task_index': task.index, 'direction': task.direction, 'path': str(task.path.resolve()), 'sha256': digest,
                      'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'episodes': 10})
    splits = {split: concatenate(parts[split]) for split in parts}
    statistics = fit_statistics(splits['train'])
    output.mkdir(parents=True)
    try:
        for split, arrays in splits.items():
            np.savez(output / f'{split}.npz', **arrays)
        (output / 'dataset_statistics.json').write_text(json.dumps(statistics, indent=2) + '\n')
        metadata = {
            'contract': CONTRACT, 'format_version': 1,
            'tasks': [{'index': t.index, 'direction': t.direction, 'start_peg': t.start, 'goal_peg': t.goal,
                       'first_move_target_peg': t.first_move_target, 'prompt': t.prompt, 'file': t.file} for t in TASKS],
            'files': files,
            'horizon': int(horizon), 'frameskip': FRAMESKIP, 'reference_rate_hz': REFERENCE_RATE_HZ,
            'episode_split': {split: list(map(int, episodes)) for split, episodes in EPISODE_SPLIT.items()},
            'global_episode_id': f'task index times {GLOBAL_EPISODE_STRIDE} plus the episode index within its file',
            'observation_rule': 'every row of a successful episode whose image is neither stale nor repeated',
            'label_rule': 'slot j is row t + 3 j: XYZ = reference_pose[row, 0:3], jaw = action_abs[row, 3]; rows past the episode end repeat the last row and are padded',
            'conditioning': 'the task prompt (T5-11B embedding) is the only task signal; no task id or goal image',
            'stationary_definition': next(iter(audits.values()))['stationary']['definition'],
            'splits': {split: {'sha256': sha256(output / f'{split}.npz'), 'samples': int(len(z['states'])),
                               'episodes': sorted(map(int, set(z['episode_indices']))),
                               'per_task_samples': {t.direction: int((z['task_indices'] == t.index).sum()) for t in TASKS},
                               'padded_slots': int(z['actions_is_pad'].sum()),
                               'stationary_fraction': float(z['stationary'].mean())} for split, z in splits.items()},
            'statistics_sha256': sha256(output / 'dataset_statistics.json'),
            'normalization': 'Cosmos min/max over the combined training split (states; actions over valid chunk slots); mean/std recorded; no clipping',
            'state': 'six measured joint angles radians, measured gripper stroke metres; no velocity, no XYZ',
            'actions': f'{horizon} x [absolute reference XYZ metres, jaw intent] at 10 Hz',
            'future_auxiliary': f'raw row min(t + {FRAMESKIP * horizon}, episode end - 1) of the same file (the end of the chunk), image and measured joint/jaw labels only; not conditioning input',
            'value_auxiliary': 'discounted terminal success from the auxiliary raw row, gamma=0.9995 per raw tick',
            'image_preprocessing': 'stored RGB224 deployment crop, never re-cropped, no augmentation; native Wan VAE synthetic latent-slot packing',
            'audit': audits, 'deployment': DEPLOYMENT_CONTRACT_V6,
        }
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    except BaseException:
        (output / 'PREPARATION_FAILED').touch()
        raise
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--horizon', type=int, default=HORIZON, choices=HORIZONS)
    arguments = parser.parse_args()
    result = prepare(arguments.output, arguments.horizon)
    print(json.dumps({key: value for key, value in result.items() if key not in ('deployment', 'audit')}, indent=2))
    for direction, report in result['audit'].items():
        print(direction, json.dumps({key: value for key, value in report.items() if key != 'per_episode'}))
