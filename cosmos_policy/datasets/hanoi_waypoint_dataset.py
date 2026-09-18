"""Cosmos latent-slot samples over the consistent waypoint_v4 labels.

Sample construction is inherited unchanged from the joint_v3 dataset: one RGB224
frame, seven measured joint/gripper values, eight relative XYZ + jaw targets, the
auxiliary future frame/state and the discounted value. Only the prepared dataset
contract differs, so this class owns its constructor and nothing else.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from cosmos_policy.datasets.hanoi_joint_data import PROMPT, read_archive, sha256
from cosmos_policy.datasets.hanoi_joint_dataset import HanoiJointDataset
from cosmos_policy.datasets.hanoi_waypoint_data import CONTRACT, DEFAULT_METADATA


class HanoiWaypointDataset(HanoiJointDataset):
    contract = CONTRACT

    def __init__(self, metadata_dir=str(DEFAULT_METADATA),
                 t5_text_embeddings_path='data/hanoi_cosmos/t5_embeddings.pkl', split='train',
                 representative_order=False, gamma=0.9995):
        # HanoiJointDataset.__init__ pins the joint_v3 contract, so it is not called.
        if split not in ('train', 'val', 'test') or not 0 < gamma <= 1:
            raise ValueError('Invalid split or discount')
        if representative_order and split != 'val':
            raise ValueError('Representative ordering is only for validation')
        root = Path(metadata_dir)
        self.metadata = json.loads((root / 'metadata.json').read_text())
        if self.metadata['contract'] != CONTRACT or (root / 'PREPARATION_FAILED').exists():
            raise ValueError('Wrong/incomplete waypoint dataset contract')
        if self.metadata['label_consistency']['fraction_over_10_mm'] != 0:
            raise ValueError('Refusing labels that failed the similar-situation consistency audit')
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
