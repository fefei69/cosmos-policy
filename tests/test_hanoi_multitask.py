"""hanoi_multitask_v6 regressions: task table, multi-file build, per-task prompts, probe matching, policy prompt rule."""
import json
import pickle

import h5py
import numpy as np
import pytest
import torch

from cosmos_policy.datasets.hanoi_dense_data import HORIZON
from cosmos_policy.datasets.hanoi_multitask_data import (
    CONTRACT, GLOBAL_EPISODE_STRIDE, PEG_Y, PROMPTS, REVERSE_OF, SAME_START_PAIRS, TASKS, TASK_BY_PROMPT,
    build_task_split, check_raw, concatenate, prompt_for,
)
from cosmos_policy.datasets.hanoi_multitask_dataset import HanoiMultitaskDataset
from cosmos_policy.datasets.hanoi_joint_data import sha256
from cosmos_policy.experiments.robot.hanoi.multitask_policy import HanoiMultitaskInferenceConfig, resolve_task, validate_checkpoint_contract
from cosmos_policy.experiments.robot.hanoi.run_hanoi_multitask_eval import chunk_error_mm, first_move_rows, matched_state_pairs
from tests.test_hanoi_dense import ROWS, make_raw


def test_task_table_is_consistent():
    assert len(TASKS) == 6 and len(set(PROMPTS)) == 6 and [t.index for t in TASKS] == list(range(6))
    assert {t.direction for t in TASKS} == {'AAAA_to_CCCC', 'CCCC_to_AAAA', 'AAAA_to_BBBB', 'BBBB_to_AAAA', 'BBBB_to_CCCC', 'CCCC_to_BBBB'}
    for t in TASKS:
        assert t.direction == f'{t.start * 4}_to_{t.goal * 4}' and t.first_move_target not in (t.start,)
        assert t.prompt == prompt_for(t.start, t.goal) and f'peg {t.start} to peg {t.goal}' in t.prompt and t.prompt.count(f'peg {t.goal}') == 2
        assert TASK_BY_PROMPT[t.prompt] is t and TASKS[REVERSE_OF[t.index]].start == t.goal and TASKS[REVERSE_OF[t.index]].goal == t.start
    assert SAME_START_PAIRS == ((0, 2), (1, 5), (3, 4)) and PEG_Y['A'] < PEG_Y['B'] < PEG_Y['C']


def make_task_raw(path, task, episodes=10):
    """A synthetic ten-episode recording of one task with the attributes and board/move columns the builder checks."""
    make_raw(path, episodes)
    n = ROWS * episodes
    with h5py.File(path, 'a') as h:
        h.attrs['direction'] = task.direction
        for key in ('goal_board', 'move_idx', 'held_disk', 'motion_stage'):
            if key in h:
                del h[key]
        h['goal_board'] = np.full((n, 4), 'ABC'.index(task.goal), np.int8)
        t = np.arange(n) % ROWS
        h['move_idx'] = np.where(t < 60, 0, 1).astype(np.int64)
        h['held_disk'] = np.where((t >= 20) & (t < 60), 1, 0).astype(np.int8)
        h['motion_stage'] = np.where((t >= 20) & (t < 40), 5, np.where((t >= 40) & (t < 60), 6, 1)).astype(np.int8)
    return path


def test_check_raw_and_task_split(tmp_path):
    task = TASKS[0]
    with h5py.File(make_task_raw(tmp_path / 'a.h5', task), 'r') as h:
        check_raw(h, task)
        arrays = build_task_split(h, task, (8,))
        with pytest.raises(ValueError, match='direction'):
            check_raw(h, TASKS[1])
    assert arrays['actions'].shape == (ROWS - 2, HORIZON, 4)
    assert (arrays['task_indices'] == 0).all() and (arrays['file_indices'] == 0).all()
    assert set(arrays['episode_indices'].tolist()) == {8 + GLOBAL_EPISODE_STRIDE * 0}
    with h5py.File(make_task_raw(tmp_path / 'b.h5', TASKS[3]), 'r') as h:
        other = build_task_split(h, TASKS[3], (8,))
    both = concatenate([arrays, other])
    assert len(both['states']) == 2 * (ROWS - 2) and set(both['episode_indices'].tolist()) == {8, 308}
    assert np.array_equal(both['state_columns'], arrays['state_columns'])


@pytest.fixture
def prepared(tmp_path):
    files, parts = [], []
    for task in TASKS:
        raw = make_task_raw(tmp_path / f'{task.direction}.h5', task)
        with h5py.File(raw, 'r') as h:
            parts.append(build_task_split(h, task, (8,)))
        files.append({'task_index': task.index, 'direction': task.direction, 'path': str(raw), 'sha256': 'x',
                      'size_bytes': raw.stat().st_size, 'mtime_ns': raw.stat().st_mtime_ns, 'episodes': 10})
    arrays = concatenate(parts)
    root = tmp_path / 'multitask'; root.mkdir()
    np.savez(root / 'val.npz', **arrays)
    stats = {'proprio_min': (arrays['states'].min(0) - 1e-3).tolist(), 'proprio_max': (arrays['states'].max(0) + 1e-3).tolist(),
             'actions_min': (arrays['actions'].reshape(-1, 4).min(0) - 1e-3).tolist(), 'actions_max': (arrays['actions'].reshape(-1, 4).max(0) + 1e-3).tolist()}
    (root / 'dataset_statistics.json').write_text(json.dumps(stats))
    metadata = {'contract': CONTRACT, 'horizon': HORIZON, 'frameskip': 3,
                'tasks': [{'index': t.index, 'direction': t.direction, 'prompt': t.prompt} for t in TASKS], 'files': files,
                'splits': {'val': {'sha256': sha256(root / 'val.npz'), 'samples': len(arrays['states'])}},
                'statistics_sha256': sha256(root / 'dataset_statistics.json')}
    (root / 'metadata.json').write_text(json.dumps(metadata))
    embeddings = tmp_path / 'embeddings.pkl'
    with embeddings.open('wb') as f:
        pickle.dump({t.prompt: torch.full((1, 512, 1024), float(t.index + 1), dtype=torch.bfloat16) for t in TASKS}, f)
    return root, embeddings, arrays


def test_multitask_dataset_reads_each_file_with_its_prompt(prepared):
    root, embeddings, arrays = prepared
    ds = HanoiMultitaskDataset(root, embeddings, split='val')
    assert len(ds) == 6 * (ROWS - 2) and ds.horizon == HORIZON
    for task in TASKS:
        i = int(np.flatnonzero(arrays['task_indices'] == task.index)[0])
        item = ds[i]
        assert item['task_index'] == task.index and float(item['t5_text_embeddings'][0, 0]) == task.index + 1
        assert item['actions'].shape == (HORIZON, 4) and item['video'].shape == (3, 25, 224, 224)
        assert item['episode_index'] == 8 + GLOBAL_EPISODE_STRIDE * task.index
        raw = ds.raw_example(i)
        assert raw['prompt'] == task.prompt and raw['image'].shape == (224, 224, 3)
    assert len(ds._handles) == 6
    ds.close()
    assert ds._handles == {}
    bad = pickle.load(embeddings.open('rb')); bad[TASKS[1].prompt] = bad[TASKS[0].prompt]
    with (root.parent / 'dup.pkl').open('wb') as f:
        pickle.dump(bad, f)
    with pytest.raises(ValueError, match='identical'):
        HanoiMultitaskDataset(root, root.parent / 'dup.pkl', split='val')


def test_probe_matching_uses_first_move_rows(prepared):
    root, embeddings, arrays = prepared
    ds = HanoiMultitaskDataset(root, embeddings, split='val')
    rows = first_move_rows(ds, 0)
    assert len(rows) == 40 - 0  # rows 20..59 of the episode, minus none (stale rows 5 and 7 are outside)
    # Synthetic tasks share identical trajectories, so matched states exist but labels never differ: no decisions.
    assert matched_state_pairs(ds, 0, 2) == []
    a = np.zeros((HORIZON, 4), np.float32); b = a.copy(); b[:, 0] = 0.002
    assert chunk_error_mm(a, b, np.zeros(HORIZON, bool)) == pytest.approx(2.0)
    ds.close()


def test_policy_requires_a_verbatim_prompt(tmp_path):
    assert resolve_task(TASKS[4].prompt).direction == 'BBBB_to_CCCC'
    for bad in ('Move all four rings from peg A to peg C following Tower of Hanoi rules.', TASKS[0].prompt.lower(), '', None):
        with pytest.raises(ValueError, match='verbatim'):
            resolve_task(bad)
    cfg = HanoiMultitaskInferenceConfig('ckpt.pt', 'stats.json', 'emb.pkl')
    assert cfg.chunk_size == HORIZON and cfg.config_file.endswith('hanoi_multitask_config.py') and cfg.config.endswith('multitask__inference')
    run = tmp_path / 'run'; (run / 'exports').mkdir(parents=True)
    stats = tmp_path / 'stats.json'; stats.write_text('{}')
    emb = tmp_path / 'emb.pkl'; emb.write_bytes(b'x')
    export = run / 'exports' / 'iter_000002000.pt'
    (run / 'joint_contract.json').write_text(json.dumps({'contract': CONTRACT, 'statistics_sha256': sha256(stats), 'prompts': list(PROMPTS),
                                                         'embeddings_sha256': sha256(emb), 'horizon': HORIZON}))
    assert validate_checkpoint_contract(export, stats, emb)['contract'] == CONTRACT
    (run / 'joint_contract.json').write_text(json.dumps({'contract': CONTRACT, 'statistics_sha256': sha256(stats), 'prompts': list(PROMPTS[:5]),
                                                         'embeddings_sha256': sha256(emb), 'horizon': HORIZON}))
    with pytest.raises(ValueError, match='prompts'):
        validate_checkpoint_contract(export, stats, emb)
