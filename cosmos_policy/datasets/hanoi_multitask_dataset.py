"""Cosmos latent-slot samples over the six-task dense labels (contract hanoi_multitask_v6).

Same seven-slot packing and dense-v5 label geometry as HanoiDenseDataset.
Differences: rows come from six recordings (one per task), each read through
its own file handle, and the conditioning prompt is the row's task prompt, so
``t5_text_embeddings`` varies per sample. Episode ids are global (task index
times 100 plus the episode within its file).
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

from cosmos_policy.datasets.hanoi_data import normalize
from cosmos_policy.datasets.hanoi_dense_data import FRAMESKIP, HORIZONS
from cosmos_policy.datasets.hanoi_dense_dataset import HanoiDenseDataset
from cosmos_policy.datasets.hanoi_joint_data import read_archive, sha256
from cosmos_policy.datasets.hanoi_multitask_data import CONTRACT, DEFAULT_EMBEDDINGS, DEFAULT_METADATA, TASKS


class HanoiMultitaskDataset(HanoiDenseDataset):
    contract = CONTRACT

    def __init__(self, metadata_dir=str(DEFAULT_METADATA), t5_text_embeddings_path=str(DEFAULT_EMBEDDINGS),
                 split='train', representative_order=False, gamma=0.9995):
        if split not in ('train', 'val', 'test') or not 0 < gamma <= 1:
            raise ValueError('Invalid split or discount')
        if representative_order and split != 'val':
            raise ValueError('Representative ordering is only for validation')
        root = Path(metadata_dir)
        self.metadata = json.loads((root / 'metadata.json').read_text())
        if self.metadata['contract'] != CONTRACT or (root / 'PREPARATION_FAILED').exists():
            raise ValueError('Wrong/incomplete multitask dataset contract')
        self.horizon = int(self.metadata['horizon'])
        if self.horizon not in HORIZONS or self.metadata['frameskip'] != FRAMESKIP:
            raise ValueError('Prepared chunk geometry differs from this contract')
        self.future_rows = FRAMESKIP * self.horizon
        prepared_tasks = self.metadata['tasks']
        if [(t['index'], t['direction'], t['prompt']) for t in prepared_tasks] != [(t.index, t.direction, t.prompt) for t in TASKS]:
            raise ValueError('Prepared task table (directions or prompts) differs from this code')
        files = sorted(self.metadata['files'], key=lambda f: f['task_index'])
        if [f['task_index'] for f in files] != list(range(len(TASKS))):
            raise ValueError('Prepared file table does not cover the six tasks once each')
        self.sources = []
        for record in files:
            source = Path(record['path'])
            identity = source.stat()
            if (identity.st_size, identity.st_mtime_ns) != (record['size_bytes'], record['mtime_ns']):
                raise ValueError(f'Raw recording changed after preparation: {source}')
            self.sources.append(source)
        path = root / f'{split}.npz'
        if sha256(path) != self.metadata['splits'][split]['sha256']:
            raise ValueError('Dense supervision changed after preparation')
        if sha256(root / 'dataset_statistics.json') != self.metadata['statistics_sha256']:
            raise ValueError('Training normalization changed after preparation')
        self.arrays = read_archive(path)
        self.stats = json.loads((root / 'dataset_statistics.json').read_text())
        self.split, self.gamma = split, gamma
        n = self.metadata['splits'][split]['samples']
        if self.arrays['states'].shape != (n, 7) or self.arrays['actions'].shape != (n, self.horizon, 4):
            raise ValueError('Archive shapes differ from the prepared contract')
        if self.arrays['task_indices'].shape != (n,) or not np.array_equal(self.arrays['task_indices'], self.arrays['file_indices']):
            raise ValueError('Task and file indices must be present and aligned')
        if self.arrays['task_indices'].min() < 0 or self.arrays['task_indices'].max() >= len(self.sources):
            raise ValueError('Task index out of range')
        with open(t5_text_embeddings_path, 'rb') as stream:
            cache = pickle.load(stream)
        self.embeddings = []
        for task in TASKS:
            if task.prompt not in cache:
                raise ValueError(f'No cached embedding for the prompt of {task.direction}')
            embedding = torch.as_tensor(cache[task.prompt]).detach().cpu()
            if embedding.shape == (1, 512, 1024):
                embedding = embedding[0]
            if embedding.shape != (512, 1024) or not torch.isfinite(embedding).all():
                raise ValueError(f'Invalid cached embedding for {task.direction}')
            self.embeddings.append(embedding.to(torch.bfloat16))
        for a in range(len(TASKS)):
            for b in range(a + 1, len(TASKS)):
                if torch.equal(self.embeddings[a], self.embeddings[b]):
                    raise ValueError('Two tasks share an identical prompt embedding')
        self.order = np.arange(n)
        if representative_order:
            # Interleave episodes so any prefix of the order covers every held-out episode of every task.
            rng = np.random.default_rng(195)
            groups = [rng.permutation(np.flatnonzero(self.arrays['episode_indices'] == ep)).tolist()
                      for ep in np.unique(self.arrays['episode_indices'])]
            self.order = np.array([group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)])
        self.resume_data_order_identity = {
            'contract': CONTRACT, 'metadata_sha256': sha256(root / 'metadata.json'),
            'indices_sha256': sha256(path), 'statistics': self.stats, 'split': split,
            'representative_order': representative_order, 'gamma': gamma,
            'embeddings_sha256': sha256(t5_text_embeddings_path),
        }
        self._handles, self._pid = {}, None

    # One read handle per recording per process (DataLoader workers fork after construction).
    def _file(self, file_index=0):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        handle = self._handles.get(int(file_index))
        if handle is None:
            handle = self._handles[int(file_index)] = h5py.File(self.sources[int(file_index)], 'r')
        return handle

    def close(self):
        for handle in getattr(self, '_handles', {}).values():
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - closing a handle inherited from a parent process
                pass
        self._handles = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handles'], state['_pid'] = {}, None
        return state

    def raw_example(self, index):
        i = int(self.order[index])
        z = self.arrays
        row = int(z['source_observation_indices'][i])
        task = int(z['task_indices'][i])
        return {'image': self._file(task)['pixels'][row], 'state': z['states'][i].copy(),
                'cartesian_position': z['cartesian_positions'][i].copy(), 'actions_abs': z['actions'][i].copy(),
                'actions_is_pad': z['actions_is_pad'][i].copy(), 'source_action_indices': z['source_action_indices'][i].copy(),
                'source_observation_index': row, 'episode_index': int(z['episode_indices'][i]),
                'stationary': bool(z['stationary'][i]), 'task_index': task, 'prompt': TASKS[task].prompt, 'archive_index': i}

    def __getitem__(self, index):
        sample = self.raw_example(index)
        i, task = sample['archive_index'], sample['task_index']
        handle = self._file(task)
        end = int(self.arrays['source_episode_bounds'][i, 1])
        future_row = min(sample['source_observation_index'] + self.future_rows, end - 1)
        future_state = np.r_[handle['joint_positions'][future_row], handle['proprio'][future_row, 6]].astype(np.float32)
        current, future = sample['image'], handle['pixels'][future_row]
        blank = np.zeros_like(current)
        frames = [blank[None]] + [np.repeat(frame[None], 4, 0) for frame in (blank, current, blank, blank, future, blank)]
        actions = normalize(sample['actions_abs'], self.stats, 'actions')
        value = np.float32(2 * self.gamma ** (end - 1 - future_row) - 1)
        result = {
            'video': torch.from_numpy(np.concatenate(frames).transpose(3, 0, 1, 2).copy()),
            'actions': actions,
            'proprio': normalize(sample['state'], self.stats, 'proprio'),
            'future_proprio': normalize(future_state, self.stats, 'proprio'),
            't5_text_embeddings': self.embeddings[task], 't5_text_mask': torch.ones(512, dtype=torch.int64),
            'fps': 16, 'padding_mask': torch.zeros(1, 224, 224), 'image_size': torch.full((4,), 224.0),
            '__key__': i, 'rollout_data_mask': 0, 'rollout_data_success_mask': 0,
            'world_model_sample_mask': 0, 'value_function_sample_mask': 0, 'global_rollout_idx': -1,
            'current_proprio_latent_idx': 1, 'current_image_latent_idx': 2, 'action_latent_idx': 3,
            'future_proprio_latent_idx': 4, 'future_image_latent_idx': 5, 'value_latent_idx': 6,
            'value_function_return': value,
            # Audit/evaluation metadata is never injected into conditioning.
            'actions_is_pad': sample['actions_is_pad'], 'source_observation_index': sample['source_observation_index'],
            'source_action_indices': sample['source_action_indices'], 'auxiliary_future_source_row': future_row,
            'episode_index': sample['episode_index'], 'stationary': sample['stationary'], 'task_index': task,
        }
        for key in ('current_wrist_image_latent_idx', 'current_wrist_image2_latent_idx', 'current_image2_latent_idx',
                    'future_wrist_image_latent_idx', 'future_wrist_image2_latent_idx', 'future_image2_latent_idx'):
            result[key] = -1
        return result
