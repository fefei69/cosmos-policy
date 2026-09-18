"""Cosmos latent-slot samples backed by the handover's explicit sparse indices."""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_policy.datasets.hanoi_data import normalize
from cosmos_policy.datasets.hanoi_joint_data import CONTRACT, DEFAULT_METADATA, PROMPT, read_archive, relative_actions, sha256


class HanoiJointDataset(Dataset):
    resume_data_order = True

    def __init__(self, metadata_dir=str(DEFAULT_METADATA),
                 t5_text_embeddings_path='data/hanoi_cosmos/t5_embeddings.pkl', split='train',
                 representative_order=False, gamma=0.9995):
        if split not in ('train', 'val', 'test') or not 0 < gamma <= 1:
            raise ValueError('Invalid split or discount')
        if representative_order and split != 'val':
            raise ValueError('Representative ordering is only for validation')
        root = Path(metadata_dir)
        self.metadata = json.loads((root / 'metadata.json').read_text())
        if self.metadata['contract'] != CONTRACT or (root / 'PREPARATION_FAILED').exists():
            raise ValueError('Wrong/incomplete joint sparse dataset contract')
        self.source = Path(self.metadata['raw_path'])
        identity = self.source.stat()
        if (identity.st_size, identity.st_mtime_ns) != (self.metadata['raw_size_bytes'], self.metadata['raw_mtime_ns']):
            raise ValueError('Raw recollection changed after preparation')
        path = root / f'{split}.npz'
        if sha256(path) != self.metadata['splits'][split]['sha256']:
            raise ValueError('Sparse supervision changed after preparation')
        if sha256(root / 'dataset_statistics.json') != self.metadata['statistics_sha256']:
            raise ValueError('Training normalization changed after preparation')
        self.arrays = read_archive(path)
        self.stats = json.loads((root / 'dataset_statistics.json').read_text())
        self.split, self.gamma = split, gamma
        if self.arrays['states'].shape != (self.metadata['splits'][split]['samples'], 7):
            raise ValueError('Joint state shape differs from prepared contract')
        with open(t5_text_embeddings_path, 'rb') as stream:
            cache = pickle.load(stream)
        embedding = torch.as_tensor(cache[PROMPT]).detach().cpu()
        if embedding.shape == (1, 512, 1024):
            embedding = embedding[0]
        if embedding.shape != (512, 1024) or not torch.isfinite(embedding).all():
            raise ValueError('Invalid cached forward instruction embedding')
        self.embedding = embedding.to(torch.bfloat16)
        self.order = np.arange(len(self.arrays['states']))
        if representative_order:
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
        self._handle, self._pid = None, None

    def __len__(self):
        return len(self.order)

    def _file(self):
        if self._pid != os.getpid() or self._handle is None:
            self.close()
            self._handle = h5py.File(self.source, 'r')
            self._pid = os.getpid()
        return self._handle

    def close(self):
        if getattr(self, '_handle', None) is not None:
            self._handle.close()
        self._handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handle'], state['_pid'] = None, None
        return state

    def __del__(self):
        self.close()

    def raw_example(self, index):
        i = int(self.order[index])
        z = self.arrays
        row = int(z['source_observation_indices'][i])
        return {'image': self._file()['pixels'][row], 'state': z['states'][i].copy(),
                'cartesian_position': z['cartesian_positions'][i].copy(), 'actions_abs': z['actions'][i].copy(),
                'actions_is_pad': z['actions_is_pad'][i].copy(), 'source_action_indices': z['source_action_indices'][i].copy(),
                'source_observation_index': row, 'episode_index': int(z['episode_indices'][i]), 'archive_index': i}

    def __getitem__(self, index):
        sample = self.raw_example(index)
        i, handle = sample['archive_index'], self._file()
        end = int(self.arrays['source_episode_bounds'][i, 1])
        future_row = min(int(sample['source_action_indices'][-1]) + 1, end - 1)
        future_state = np.r_[handle['joint_positions'][future_row], handle['proprio'][future_row, 6]].astype(np.float32)
        current, future = sample['image'], handle['pixels'][future_row]
        blank = np.zeros_like(current)
        frames = [blank[None]] + [np.repeat(frame[None], 4, 0) for frame in (blank, current, blank, blank, future, blank)]
        actions = normalize(relative_actions(sample['actions_abs'], sample['cartesian_position']), self.stats, 'actions')
        value = np.float32(2 * self.gamma ** (end - 1 - future_row) - 1)
        result = {
            'video': torch.from_numpy(np.concatenate(frames).transpose(3, 0, 1, 2).copy()),
            'actions': actions,
            'proprio': normalize(sample['state'], self.stats, 'proprio'),
            'future_proprio': normalize(future_state, self.stats, 'proprio'),
            't5_text_embeddings': self.embedding, 't5_text_mask': torch.ones(512, dtype=torch.int64),
            'fps': 16, 'padding_mask': torch.zeros(1, 224, 224), 'image_size': torch.full((4,), 224.0),
            '__key__': i, 'rollout_data_mask': 0, 'rollout_data_success_mask': 0,
            'world_model_sample_mask': 0, 'value_function_sample_mask': 0, 'global_rollout_idx': -1,
            'current_proprio_latent_idx': 1, 'current_image_latent_idx': 2, 'action_latent_idx': 3,
            'future_proprio_latent_idx': 4, 'future_image_latent_idx': 5, 'value_latent_idx': 6,
            'value_function_return': value,
            # Audit/evaluation metadata is never injected into conditioning.
            'actions_is_pad': sample['actions_is_pad'], 'source_observation_index': sample['source_observation_index'],
            'source_action_indices': sample['source_action_indices'], 'auxiliary_future_source_row': future_row,
            'episode_index': sample['episode_index'],
        }
        for key in ('current_wrist_image_latent_idx', 'current_wrist_image2_latent_idx', 'current_image2_latent_idx',
                    'future_wrist_image_latent_idx', 'future_wrist_image2_latent_idx', 'future_image2_latent_idx'):
            result[key] = -1
        return result
