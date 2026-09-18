"""waypoint_v4 regressions: label audit gates, dataset contract, sync-free metrics, selection."""
import json
import pickle

import h5py
import numpy as np
import pytest
import torch

from cosmos_policy.datasets.hanoi_joint_data import CONTRACT as JOINT_CONTRACT, PROMPT, fit_statistics, sha256
from cosmos_policy.datasets.hanoi_joint_dataset import HanoiJointDataset
from cosmos_policy.datasets.hanoi_waypoint_data import CONTRACT, TARGET_EXTRACTION, check_audit, validate_archive
from cosmos_policy.datasets.hanoi_waypoint_dataset import HanoiWaypointDataset
from cosmos_policy.experiments.robot.hanoi.run_hanoi_physical_eval import first_target_metrics, summarize_physical
from cosmos_policy.experiments.robot.hanoi.waypoint_policy import (
    HanoiWaypointInferenceConfig, inference_config_for, validate_checkpoint_contract,
)
from cosmos_policy.utils.hanoi_waypoint_training import records_to_floats


def make_arrays():
    targets = np.array([[2, 4, 6, 8, 10, 12, 14, 16], [23] * 8])
    absolute = np.full((2, 8, 4), .25, np.float32)
    absolute[:, :, 3] = np.arange(8) % 2
    absolute[1] = absolute[1, 0]
    return {'states': np.array([[.1] * 6 + [.03], [.2] * 6 + [.04]], np.float32),
            'cartesian_positions': np.array([[.2, -.1, .3], [.21, -.1, .3]], np.float32),
            'actions': absolute, 'source_observation_indices': np.array([0, 20]),
            'source_action_indices': targets, 'source_episode_bounds': np.array([[0, 24], [0, 24]]),
            'episode_indices': np.zeros(2, np.int64), 'actions_is_pad': np.array([[False] * 8, [False] + [True] * 7])}


@pytest.fixture
def fixture(tmp_path):
    raw = tmp_path / 'source.h5'
    pixels = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    with h5py.File(raw, 'w') as h:
        h['pixels'] = np.stack([pixels] * 24)
        h['joint_positions'] = np.arange(144, dtype=np.float32).reshape(24, 6) / 100
        state = np.zeros((24, 8), np.float32)
        state[:, :3] = [.2, -.1, .3]
        state[:, 6] = .03
        h['proprio'] = state
    arrays = make_arrays()
    embeddings = tmp_path / 'embeddings.pkl'
    with embeddings.open('wb') as f:
        pickle.dump({PROMPT: torch.zeros(512, 1024, dtype=torch.bfloat16)}, f)
    roots = {}
    for contract in (JOINT_CONTRACT, CONTRACT):
        metadata = tmp_path / contract
        metadata.mkdir()
        np.savez(metadata / 'train.npz', **arrays)
        stats = fit_statistics(arrays, arrays['states'])
        (metadata / 'dataset_statistics.json').write_text(json.dumps(stats))
        m = {'contract': contract, 'raw_path': str(raw), 'raw_size_bytes': raw.stat().st_size,
             'raw_mtime_ns': raw.stat().st_mtime_ns,
             'splits': {'train': {'sha256': sha256(metadata / 'train.npz'), 'samples': 2}},
             'statistics_sha256': sha256(metadata / 'dataset_statistics.json'),
             'label_consistency': {'passed': True, 'fraction_over_10_mm': 0.0}}
        (metadata / 'metadata.json').write_text(json.dumps(m))
        roots[contract] = metadata
    return roots, embeddings


def test_waypoint_samples_match_joint_samples_for_identical_arrays(fixture):
    roots, embeddings = fixture
    joint = HanoiJointDataset(roots[JOINT_CONTRACT], embeddings)
    waypoint = HanoiWaypointDataset(roots[CONTRACT], embeddings)
    assert len(joint) == len(waypoint) == 2
    for index in range(2):
        a, b = joint[index], waypoint[index]
        assert a.keys() == b.keys()
        for key in a:
            if isinstance(a[key], torch.Tensor):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            elif isinstance(a[key], np.ndarray):
                np.testing.assert_array_equal(a[key], b[key])
            else:
                assert a[key] == b[key], key
    assert waypoint.resume_data_order_identity['contract'] == CONTRACT
    assert joint.resume_data_order_identity['contract'] == JOINT_CONTRACT
    joint.close()
    waypoint.close()


def test_contracts_are_not_interchangeable(fixture):
    roots, embeddings = fixture
    with pytest.raises(ValueError, match='contract'):
        HanoiWaypointDataset(roots[JOINT_CONTRACT], embeddings)
    with pytest.raises(ValueError, match='contract'):
        HanoiJointDataset(roots[CONTRACT], embeddings)


def test_waypoint_dataset_refuses_inconsistent_labels(fixture):
    roots, embeddings = fixture
    metadata = roots[CONTRACT] / 'metadata.json'
    record = json.loads(metadata.read_text())
    record['label_consistency']['fraction_over_10_mm'] = 0.146
    metadata.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='consistency'):
        HanoiWaypointDataset(roots[CONTRACT], embeddings)


def make_audit():
    contract = {'version': 4, 'target_extraction': TARGET_EXTRACTION, 'action_horizon': 8, 'execution_prefix': 1,
                'intermediate_motion_noise': 'present', 'recorded_path_deviation_budget_m': 0.0025}
    record = {'label_consistency': {'passed': True, 'fraction_over_10_mm': 0.0, 'split': 'train'},
              'admission': {'passed': True}, 'source': '/raw.h5', 'source_sha256': 'abc', 'contract': contract}
    specification = {'source': {'hdf5': '/raw.h5', 'hdf5_sha256': 'abc'}, 'prompt': PROMPT,
                     'contract': {'version': 3, 'action_horizon': 8, 'execution_prefix': 1}}
    return record, specification


def test_audit_gates():
    record, specification = make_audit()
    check_audit(record, 'h', {'audit_sha256': 'h'}, specification)
    bad = json.loads(json.dumps(record))
    bad['label_consistency']['fraction_over_10_mm'] = 0.01
    with pytest.raises(ValueError, match='consistency'):
        check_audit(bad, 'h', {'audit_sha256': 'h'}, specification)
    with pytest.raises(ValueError, match='verification'):
        check_audit(record, 'other', {'audit_sha256': 'h'}, specification)
    bad = json.loads(json.dumps(record))
    bad['contract']['action_horizon'] = 6
    with pytest.raises(ValueError, match='beyond label extraction'):
        check_audit(bad, 'h', {'audit_sha256': 'h'}, specification)
    bad = json.loads(json.dumps(record))
    bad['source_sha256'] = 'zzz'
    with pytest.raises(ValueError, match='different raw recording'):
        check_audit(bad, 'h', {'audit_sha256': 'h'}, specification)


def test_validate_archive_uses_audited_count(tmp_path):
    raw = tmp_path / 'raw.h5'
    arrays = make_arrays()
    arrays['validated'] = np.array(True)
    arrays['episode_indices'] = np.array([0, 40])
    arrays['source_episode_bounds'] = np.array([[0, 24], [0, 24]])
    with h5py.File(raw, 'w') as h:
        h['ep_offset'] = np.zeros(50, np.int64)
        h['ep_len'] = np.full(50, 24, np.int64)
        h['joint_positions'] = np.tile(np.array([.1] * 6, np.float32), (24, 1))
        proprio = np.zeros((24, 8), np.float32)
        proprio[:, :3] = [.2, -.1, .3]
        proprio[:, 6] = .03
        h['proprio'] = proprio
        actions = np.full((24, 4), .25, np.float32)
        actions[:, 3] = np.arange(24) % 2
        h['action_abs'] = actions
        h['command_monotonic_ns'] = np.full(24, 10**6, np.int64)
        h['image_receipt_monotonic_ns'] = np.zeros(24, np.int64)
    # This synthetic archive is not internally consistent for a real split; the
    # count/episode gate must fire before any numeric comparison does.
    with h5py.File(raw, 'r') as h:
        with pytest.raises(ValueError, match='Wrong/unvalidated'):
            validate_archive(arrays, h, 'train', 3)
        with pytest.raises(ValueError, match='episode split'):
            validate_archive(arrays, h, 'train', 2)


def test_records_to_floats_matches_per_microbatch_conversion():
    pending = [(2, {'loss': torch.tensor(0.5), 'demo_sample_action_l1_loss': torch.tensor(0.25)}),
               (2, {'loss': torch.tensor(1.0), 'demo_sample_action_l1_loss': torch.tensor(0.75),
                    'demo_sample_value_l1_loss': torch.tensor(0.125)})]
    assert records_to_floats(pending) == [(2, {'loss': 0.5, 'demo_sample_action_l1_loss': 0.25}),
                                          (2, {'loss': 1.0, 'demo_sample_action_l1_loss': 0.75, 'demo_sample_value_l1_loss': 0.125})]
    assert records_to_floats([]) == []
    with pytest.raises(FloatingPointError, match='Non-finite'):
        records_to_floats([(2, {'loss': torch.tensor(float('nan')), 'demo_sample_action_l1_loss': torch.tensor(0.1)})])
    with pytest.raises(FloatingPointError, match='action loss'):
        records_to_floats([(2, {'loss': torch.tensor(0.1), 'demo_sample_action_l1_loss': torch.tensor(float('inf'))})])


def test_selection_and_first_target_metrics():
    from examples.hanoi.run_waypoint import select_checkpoint
    from pathlib import Path
    def report(rate, xyz):
        return {'physical_metrics': {'first_hit_rate': rate, 'first_xyz_mm': xyz}}
    reports = [(report(0.8, 4.0), Path('iter_000001000.pt')), (report(0.9, 6.0), Path('iter_000002000.pt')),
               (report(0.9, 5.0), Path('iter_000004000.pt')), (report(0.9, 5.0), Path('iter_000003000.pt'))]
    assert select_checkpoint(reports)[1].name == 'iter_000003000.pt'
    predicted = np.array([[.2, -.1, .3, 1]] + [[0, 0, 0, 0]] * 7, np.float32)
    target = predicted.copy()
    target[0, :3] += [.004, 0, 0]
    assert first_target_metrics(predicted, target, 5.0) == {'first_jaw_correct': True, 'first_within_tolerance': True, 'first_hit': True}
    target[0, 3] = 0
    assert first_target_metrics(predicted, target, 5.0)['first_hit'] is False
    assert first_target_metrics(predicted, target, 3.0)['first_within_tolerance'] is False
    samples = [{'first_xyz_mm': 4.0, 'valid_horizon_xyz_mm': 4.0, 'last_valid_xyz_mm': 4.0, 'first_event': None,
                'jaw_tn': 4, 'jaw_fp': 0, 'jaw_fn': 0, 'jaw_tp': 4, 'valid_targets': 8,
                'first_jaw_correct': True, 'first_within_tolerance': True, 'first_hit': True},
               {'first_xyz_mm': 40.0, 'valid_horizon_xyz_mm': 40.0, 'last_valid_xyz_mm': 40.0, 'first_event': 'close',
                'jaw_tn': 4, 'jaw_fp': 0, 'jaw_fn': 0, 'jaw_tp': 4, 'valid_targets': 8,
                'first_jaw_correct': True, 'first_within_tolerance': False, 'first_hit': False}]
    summary = summarize_physical(samples, 5.0)
    assert summary['first_hit_rate'] == 0.5 and summary['first_within_tolerance_fraction'] == 0.5
    assert summary['first_jaw_accuracy'] == 1.0 and summary['first_xyz_mm'] == 22.0 and summary['hit_tolerance_mm'] == 5.0
    assert summary['close_first_xyz_mm'] == 40.0 and summary['samples'] == 2


def test_inference_config_and_contract_validation(tmp_path):
    cfg = inference_config_for(CONTRACT, 'ckpt.pt', 'stats.json', 'emb.pkl')
    assert isinstance(cfg, HanoiWaypointInferenceConfig)
    assert cfg.config == 'cosmos_predict2_2b_hanoi_waypoint__inference' and cfg.config_file.endswith('hanoi_waypoint_config.py')
    assert inference_config_for(JOINT_CONTRACT, 'ckpt.pt', 'stats.json', 'emb.pkl').config == 'cosmos_predict2_2b_hanoi_joint__inference'
    with pytest.raises(ValueError, match='Unknown'):
        inference_config_for('other', 'ckpt.pt', 'stats.json', 'emb.pkl')
    run = tmp_path / 'run'
    (run / 'exports').mkdir(parents=True)
    stats = tmp_path / 'stats.json'
    stats.write_text('{}')
    (run / 'joint_contract.json').write_text(json.dumps({'contract': CONTRACT, 'statistics_sha256': sha256(stats)}))
    checkpoint = run / 'exports' / 'iter_000001000.pt'
    assert validate_checkpoint_contract(checkpoint, stats, CONTRACT)['contract'] == CONTRACT
    with pytest.raises(ValueError, match='different observation/action contract'):
        validate_checkpoint_contract(checkpoint, stats, JOINT_CONTRACT)
