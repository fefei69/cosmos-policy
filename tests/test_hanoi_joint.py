"""Joint/sparse regressions: coordinate context, padding, and future-input isolation."""
import json
import pickle

import h5py
import numpy as np
import pytest
import torch

from cosmos_policy.datasets.hanoi_data import normalize
from cosmos_policy.datasets.hanoi_joint_data import CONTRACT, PROMPT, fit_statistics, relative_actions, sha256
from cosmos_policy.datasets.hanoi_joint_dataset import HanoiJointDataset
from cosmos_policy.experiments.robot.hanoi.joint_policy import (
    HanoiJointInferenceConfig, HanoiJointPolicy, absolute_joint_actions, make_joint_observation,
    validate_checkpoint_contract, validate_joint_config,
)
from cosmos_policy.experiments.robot.hanoi.run_hanoi_joint_eval import physical_metrics, summarize


@pytest.fixture
def fixture(tmp_path):
    raw = tmp_path / 'source.h5'
    pixels = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    with h5py.File(raw, 'w') as h:
        h['pixels'] = np.stack([pixels] * 24)
        h['joint_positions'] = np.arange(144, dtype=np.float32).reshape(24, 6) / 100
        state = np.zeros((24, 8), np.float32)
        state[:, :3] = [.2, -.1, .3]
        state[:, 3:6] = 123456  # Velocity must not enter the learned state.
        state[:, 6] = .03
        state[:, 7] = 999999  # Legacy commanded gripper must not enter it either.
        h['proprio'] = state
    targets = np.array([[2, 4, 6, 8, 10, 12, 14, 16], [23] * 8])
    absolute = np.full((2, 8, 4), .25, np.float32)
    absolute[:, :, 3] = np.arange(8) % 2
    absolute[1] = absolute[1, 0]
    arrays = {'states': np.array([[.1] * 6 + [.03], [.2] * 6 + [.04]], np.float32),
              'cartesian_positions': np.array([[.2, -.1, .3], [.21, -.1, .3]], np.float32),
              'actions': absolute, 'source_observation_indices': np.array([0, 20]),
              'source_action_indices': targets, 'source_episode_bounds': np.array([[0, 24], [0, 24]]),
              'episode_indices': np.zeros(2, np.int64), 'actions_is_pad': np.array([[False] * 8, [False] + [True] * 7])}
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    np.savez(metadata / 'train.npz', **arrays)
    stats = fit_statistics(arrays, arrays['states'])
    (metadata / 'dataset_statistics.json').write_text(json.dumps(stats))
    m = {'contract': CONTRACT, 'raw_path': str(raw), 'raw_size_bytes': raw.stat().st_size,
         'raw_mtime_ns': raw.stat().st_mtime_ns,
         'splits': {'train': {'sha256': sha256(metadata / 'train.npz'), 'samples': 2}},
         'statistics_sha256': sha256(metadata / 'dataset_statistics.json')}
    (metadata / 'metadata.json').write_text(json.dumps(m))
    embeddings = tmp_path / 'embeddings.pkl'
    with embeddings.open('wb') as f:
        pickle.dump({PROMPT: torch.zeros(512, 1024, dtype=torch.bfloat16)}, f)
    return raw, metadata, embeddings, arrays, stats


def test_explicit_sparse_rows_and_joint_state(fixture):
    _, meta, embeddings, arrays, stats = fixture
    ds = HanoiJointDataset(meta, embeddings)
    item = ds[0]
    np.testing.assert_array_equal(item['proprio'], normalize(arrays['states'][0], stats, 'proprio'))
    np.testing.assert_allclose(item['actions'], normalize(relative_actions(arrays['actions'][0], arrays['cartesian_positions'][0]), stats, 'actions'))
    assert item['proprio'].shape == (7,) and item['actions'].shape == (8, 4)
    np.testing.assert_array_equal(item['source_action_indices'], [2, 4, 6, 8, 10, 12, 14, 16])
    assert item['auxiliary_future_source_row'] == 17  # Not obs row + 8.
    assert item['video'].shape == (3, 25, 224, 224)
    assert 'cartesian_position' not in item
    last = ds[1]
    assert last['auxiliary_future_source_row'] == 23
    assert last['actions_is_pad'].sum() == 7
    np.testing.assert_array_equal(last['actions'], np.repeat(last['actions'][:1], 8, axis=0))
    ds.close()


def test_future_labels_cannot_change_current_conditioning(fixture):
    raw, meta, embeddings, _, _ = fixture
    ds = HanoiJointDataset(meta, embeddings)
    before = ds[0]
    ds.close()
    with h5py.File(raw, 'r+') as h:
        h['pixels'][17] = 0
        h['joint_positions'][17] = 999
        h['proprio'][:, 3:6] = float('nan')
        h['proprio'][:, 7] = float('nan')
    after = ds[0]
    torch.testing.assert_close(before['video'][:, :9], after['video'][:, :9])
    np.testing.assert_array_equal(before['proprio'], after['proprio'])
    np.testing.assert_array_equal(before['actions'], after['actions'])
    assert not np.array_equal(before['future_proprio'], after['future_proprio'])
    ds.close()


def test_reject_changed_normalization_and_supervision(fixture):
    _, meta, embeddings, arrays, _ = fixture
    arrays['states'][0, 0] = 999
    np.savez(meta / 'train.npz', **arrays)
    with pytest.raises(ValueError, match='supervision changed'):
        HanoiJointDataset(meta, embeddings)


def test_common_cartesian_anchor_is_independent_of_joint_angles():
    xyz = np.array([.2, -.1, .3], np.float32)
    targets = np.zeros((8, 4), np.float32)
    targets[:, :3] = xyz + np.arange(8, dtype=np.float32)[:, None] * .001
    targets[:, 3] = np.arange(8) % 2
    relative = relative_actions(targets, xyz)
    np.testing.assert_allclose(absolute_joint_actions(relative, xyz), targets, atol=1e-7)
    changed = absolute_joint_actions(relative, xyz + [.1, .2, -.1])
    np.testing.assert_allclose(changed[:, :3] - targets[:, :3], np.tile([.1, .2, -.1], (8, 1)), atol=1e-7)
    obs = make_joint_observation(np.zeros((224, 224, 3), np.uint8), [1, 2, 3, 4, 5, 6, .03])
    assert set(obs) == {'primary_image', 'proprio'}
    with pytest.raises(ValueError):
        make_joint_observation(obs['primary_image'], [1, 2, 3, 4])


def test_padding_excluded_from_all_physical_metrics():
    target = np.zeros((8, 4), np.float32)
    target[0, 3] = 1
    predicted = target.copy()
    predicted[0, 0] = .003
    predicted[1:, :] = 999
    metrics = physical_metrics(predicted, target, np.array([False] + [True] * 7), 'open')
    assert metrics['first_xyz_mm'] == pytest.approx(3)
    assert metrics['valid_horizon_xyz_mm'] == pytest.approx(3)
    assert metrics['last_valid_xyz_mm'] == pytest.approx(3)
    assert metrics['valid_targets'] == 1 and metrics['jaw_tp'] == 1
    result = summarize([metrics])
    assert result['jaw_accuracy'] == 1 and result['jaw_balanced_accuracy'] is None
    assert result['open_first_target_support'] == 1


def test_checkpoint_contract_rejects_previous_dense_run(tmp_path):
    checkpoint = tmp_path / 'checkpoints/iter_000028000'
    checkpoint.mkdir(parents=True)
    stats = tmp_path / 'stats.json'
    stats.write_text('{}')
    with pytest.raises(FileNotFoundError):
        validate_checkpoint_contract(checkpoint, stats)
    (tmp_path / 'joint_contract.json').write_text(json.dumps({'contract': 'old_dense_xyz', 'statistics_sha256': sha256(stats)}))
    with pytest.raises(ValueError, match='different observation/action contract'):
        validate_checkpoint_contract(checkpoint, stats)


def test_joint_config_cannot_silently_use_dense_horizon():
    cfg = HanoiJointInferenceConfig('x', 'y', 'z')
    validate_joint_config(cfg)
    cfg.chunk_size = 63
    with pytest.raises(ValueError, match='chunk_size'):
        validate_joint_config(cfg)
