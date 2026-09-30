"""Cosmos latent-slot samples over the play labels (contract hanoi_play_k5).

Same seven-slot packing and dense-v5 chunk geometry as HanoiDenseDataset.
Differences: rows come from two recordings (the play walks, the expert clips),
each read through its own handle, and the conditioning prompt is the row's
goal-board sentence, so ``t5_text_embeddings`` is one of 81 cached embeddings.
The goal, the chunk cut and every audit column were fixed at preparation time
and live in the archive; nothing is sampled here.
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
from cosmos_policy.datasets.hanoi_play_data import BOARDS, CONTRACT, DEFAULT_EMBEDDINGS, DEFAULT_METADATA, FILES, PROMPTS, PROMPTS_SHA256

ARCHIVE_COLUMNS = ('source_observation_indices', 'states', 'actions', 'actions_is_pad', 'source_action_indices', 'episode_indices',
                   'source_episode_bounds', 'segment_bounds', 'file_indices', 'segment_kinds', 'move_indices', 'motion_stages',
                   'board_indices', 'goal_board_indices', 'next_board_indices', 'goal_end_rows', 'goal_moves_ahead',
                   'goal_graph_distance', 'cartesian_positions', 'measured_speed_m_per_s', 'stationary')


def load_prompt_embeddings(path, dtype=torch.bfloat16):
    """The 81 goal sentences' embeddings, indexed like BOARDS; every one present, finite and distinct."""
    with open(path, 'rb') as stream:
        cache = pickle.load(stream)
    embeddings = []
    for board, prompt in zip(BOARDS, PROMPTS):
        if prompt not in cache:
            raise ValueError(f'No cached embedding for the goal sentence of {board}')
        embedding = torch.as_tensor(cache[prompt]).detach().cpu()
        if embedding.shape == (1, 512, 1024):
            embedding = embedding[0]
        if embedding.shape != (512, 1024) or not torch.isfinite(embedding).all():
            raise ValueError(f'Invalid cached embedding for {board}')
        embeddings.append(embedding.to(dtype))
    flat = torch.stack([e.float().flatten() for e in embeddings])
    if len(torch.unique(flat, dim=0)) != len(BOARDS):
        raise ValueError('Two goal boards share an identical prompt embedding')
    return embeddings


class HanoiPlayDataset(HanoiDenseDataset):
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
            raise ValueError('Wrong/incomplete play dataset contract')
        self.horizon = int(self.metadata['horizon'])
        if self.horizon not in HORIZONS or self.metadata['frameskip'] != FRAMESKIP:
            raise ValueError('Prepared chunk geometry differs from this contract')
        self.future_rows = FRAMESKIP * self.horizon
        if self.metadata['prompts_sha256'] != PROMPTS_SHA256 or self.metadata['prompts'] != list(PROMPTS):
            raise ValueError('Prepared goal sentences differ from this code')
        files = sorted(self.metadata['files'], key=lambda f: f['index'])
        if [(f['index'], f['role']) for f in files] != list(enumerate(FILES)):
            raise ValueError('Prepared file table does not cover the play and expert recordings once each')
        self.sources = []
        for record in files:
            source = Path(record['path'])
            identity = source.stat()
            if (identity.st_size, identity.st_mtime_ns) != (record['size_bytes'], record['mtime_ns']):
                raise ValueError(f'Raw recording changed after preparation: {source}')
            self.sources.append(source)
        path = root / f'{split}.npz'
        if sha256(path) != self.metadata['splits'][split]['sha256']:
            raise ValueError('Play supervision changed after preparation')
        if sha256(root / 'dataset_statistics.json') != self.metadata['statistics_sha256']:
            raise ValueError('Training normalization changed after preparation')
        self.arrays = read_archive(path)
        self.stats = json.loads((root / 'dataset_statistics.json').read_text())
        self.split, self.gamma = split, gamma
        n = self.metadata['splits'][split]['samples']
        if self.arrays['states'].shape != (n, 7) or self.arrays['actions'].shape != (n, self.horizon, 4):
            raise ValueError('Archive shapes differ from the prepared contract')
        for key in ARCHIVE_COLUMNS:
            if key not in self.arrays or len(self.arrays[key]) != n:
                raise ValueError(f'Archive lacks the audit column {key}')
        if self.arrays['file_indices'].min() < 0 or self.arrays['file_indices'].max() >= len(self.sources):
            raise ValueError('File index out of range')
        if self.arrays['goal_board_indices'].min() < 0 or self.arrays['goal_board_indices'].max() >= len(BOARDS):
            raise ValueError('Goal board index out of range')
        self.embeddings = load_prompt_embeddings(t5_text_embeddings_path)
        self.order = np.arange(n)
        if representative_order:
            # Interleave walks so any prefix of the order covers every held-out walk.
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
        file_index = int(z['file_indices'][i])
        goal = int(z['goal_board_indices'][i])
        return {'image': self._file(file_index)['pixels'][row], 'state': z['states'][i].copy(),
                'cartesian_position': z['cartesian_positions'][i].copy(), 'actions_abs': z['actions'][i].copy(),
                'actions_is_pad': z['actions_is_pad'][i].copy(), 'source_action_indices': z['source_action_indices'][i].copy(),
                'source_observation_index': row, 'episode_index': int(z['episode_indices'][i]),
                'stationary': bool(z['stationary'][i]), 'file_index': file_index, 'board': BOARDS[int(z['board_indices'][i])],
                'goal_board': BOARDS[goal], 'goal_board_index': goal, 'prompt': PROMPTS[goal],
                'motion_stage': int(z['motion_stages'][i]), 'move_index': int(z['move_indices'][i]),
                'goal_moves_ahead': int(z['goal_moves_ahead'][i]), 'goal_graph_distance': int(z['goal_graph_distance'][i]),
                'archive_index': i}

    def __getitem__(self, index):
        sample = self.raw_example(index)
        i, goal = sample['archive_index'], sample['goal_board_index']
        handle = self._file(sample['file_index'])
        end = int(self.arrays['source_episode_bounds'][i, 1])  # one past the goal move's last row
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
            't5_text_embeddings': self.embeddings[goal], 't5_text_mask': torch.ones(512, dtype=torch.int64),
            'fps': 16, 'padding_mask': torch.zeros(1, 224, 224), 'image_size': torch.full((4,), 224.0),
            '__key__': i, 'rollout_data_mask': 0, 'rollout_data_success_mask': 0,
            'world_model_sample_mask': 0, 'value_function_sample_mask': 0, 'global_rollout_idx': -1,
            'current_proprio_latent_idx': 1, 'current_image_latent_idx': 2, 'action_latent_idx': 3,
            'future_proprio_latent_idx': 4, 'future_image_latent_idx': 5, 'value_latent_idx': 6,
            'value_function_return': value,
            # Audit/evaluation metadata is never injected into conditioning.
            'actions_is_pad': sample['actions_is_pad'], 'source_observation_index': sample['source_observation_index'],
            'source_action_indices': sample['source_action_indices'], 'auxiliary_future_source_row': future_row,
            'episode_index': sample['episode_index'], 'stationary': sample['stationary'], 'goal_board_index': goal,
            'motion_stage': sample['motion_stage'], 'goal_moves_ahead': sample['goal_moves_ahead'],
        }
        for key in ('current_wrist_image_latent_idx', 'current_wrist_image2_latent_idx', 'current_image2_latent_idx',
                    'future_wrist_image_latent_idx', 'future_wrist_image2_latent_idx', 'future_image2_latent_idx'):
            result[key] = -1
        return result
