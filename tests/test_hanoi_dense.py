"""hanoi_dense_v5 regressions: builder rules, statistics, dataset contract, loaders, selection, metrics."""
import json
import pickle

import h5py
import numpy as np
import pytest
import torch

from cosmos_policy.datasets.hanoi_dense_data import (
    CONTRACT, DEPLOYMENT_CONTRACT, FRAMESKIP, HORIZON, STATIONARY_SPEED_M_PER_S, audit, build_split,
    cross_check_openpi, finite_difference_speed, fit_statistics,
)
from cosmos_policy.datasets.hanoi_dense_dataset import FUTURE_ROWS, HanoiDenseDataset
from cosmos_policy.datasets.hanoi_joint_data import PROMPT, sha256
from cosmos_policy.experiments.robot.hanoi.dense_policy import (
    HanoiDenseInferenceConfig, threshold_jaw, validate_checkpoint_contract, validate_dense_config,
)
from cosmos_policy.experiments.robot.hanoi.run_hanoi_dense_eval import flip_slot, per_sample_metrics, summarize
from cosmos_policy.models.hanoi_dense_model import normalise_video_base_state

ROWS = 120  # one synthetic episode of 120 rows at 30 Hz (4 s)


def make_raw(path, episodes=2):
    n = ROWS * episodes
    with h5py.File(path, 'w') as h:
        h.attrs['schema_version'] = 4; h.attrs['dry_run'] = False
        h.attrs['direction'] = 'AAAA_to_CCCC'; h.attrs['action_abs_alignment'] = 'post_action_reference'
        pixels = (np.arange(n, dtype=np.uint32)[:, None, None, None] % 255).astype(np.uint8) * np.ones((1, 224, 224, 3), np.uint8)
        h['pixels'] = pixels
        h['ep_offset'] = np.arange(episodes, dtype=np.int64) * ROWS
        h['ep_len'] = np.full(episodes, ROWS, np.int32)
        h['episode_success'] = np.ones(n, np.int64)
        t = np.arange(n) % ROWS
        xyz = np.stack([0.4 + 0.001 * t, 0.0 * t, 0.1 + 0.0005 * t], 1).astype(np.float32)  # steady motion 2.24 mm/row
        xyz[t < 20] = xyz[19]  # first 20 rows stationary (copies of one pose)
        ref = np.concatenate([xyz, np.tile([0, np.pi / 4, 0], (n, 1))], 1).astype(np.float32)
        h['reference_pose'] = ref
        jaw = (t >= 60).astype(np.float32)  # one flip per episode at row 60
        h['action_abs'] = np.concatenate([xyz, jaw[:, None]], 1).astype(np.float32)
        proprio = np.zeros((n, 8), np.float32); proprio[:, :3] = xyz + 0.0004; proprio[:, 3:6] = 0.05; proprio[:, 6] = 0.02 + 0.0001 * (t % 7)
        h['proprio'] = proprio
        h['joint_positions'] = (0.01 * np.arange(6)[None, :] + 0.001 * t[:, None]).astype(np.float32)
        gripper = np.zeros(n, np.uint8); gripper[t == 60] = 1
        h['gripper_command_issued'] = gripper
        stale = np.zeros(n, np.uint8); stale[t == 5] = 1
        repeated = np.zeros(n, np.uint8); repeated[t == 5] = 1; repeated[t == 7] = 1
        h['image_stale'] = stale; h['image_repeated'] = repeated
        h['cartesian_command_issued'] = np.zeros(n, np.uint8); h['motion_stage'] = np.zeros(n, np.int8)
    return path


def test_build_split_rules(tmp_path):
    with h5py.File(make_raw(tmp_path / 'raw.h5'), 'r') as h:
        arrays = build_split(h, range(1))
    rows = arrays['source_observation_indices']
    assert len(rows) == ROWS - 2 and 5 not in rows and 7 not in rows, 'stale and repeated rows are the only exclusions'
    assert arrays['states'].shape == (ROWS - 2, 7) and arrays['actions'].shape == (ROWS - 2, HORIZON, 4)
    # slot j is row t + 3 j, clamped and padded past the episode end
    t0 = rows[0]
    np.testing.assert_array_equal(arrays['source_action_indices'][0], t0 + FRAMESKIP * np.arange(1, HORIZON + 1))
    last = np.flatnonzero(rows == ROWS - 1)[0]
    assert arrays['actions_is_pad'][last].all() and (arrays['source_action_indices'][last] == ROWS - 1).all()
    with h5py.File(tmp_path / 'raw.h5', 'r') as h:
        ref, aabs = h['reference_pose'][:], h['action_abs'][:]
    np.testing.assert_array_equal(arrays['actions'][0, :, :3], ref[t0 + FRAMESKIP * np.arange(1, HORIZON + 1), :3])
    np.testing.assert_array_equal(arrays['actions'][0, :, 3], aabs[t0 + FRAMESKIP * np.arange(1, HORIZON + 1), 3])
    assert arrays['stationary'][:15].all() and not arrays['stationary'][40]
    assert 'cartesian_positions' in arrays and arrays['validated']


def test_statistics_use_float64_and_valid_slots_only(tmp_path):
    with h5py.File(make_raw(tmp_path / 'raw.h5'), 'r') as h:
        arrays = build_split(h, range(2))
    stats = fit_statistics(arrays)
    for group in ('proprio', 'actions'):
        assert all(lo <= m <= hi for lo, m, hi in zip(stats[f'{group}_min'], stats[f'{group}_mean'], stats[f'{group}_max']))
    valid = arrays['actions'][~arrays['actions_is_pad']]
    np.testing.assert_allclose(np.array(stats['actions_max'])[[0, 2]], valid[:, [0, 2]].max(0), atol=1e-7)
    assert stats['actions_max'][1] == pytest.approx(1e-6) and stats['actions_min'][1] == pytest.approx(-1e-6)  # constant column widened
    assert stats['actions_min'][3] == 0.0 and stats['actions_max'][3] == 1.0 and stats['actions_valid_slots'] == len(valid)


def test_audit_reports_alignment_flips_and_motion(tmp_path):
    with h5py.File(make_raw(tmp_path / 'raw.h5'), 'r') as h:
        report = audit(h)
    assert report['action_abs_vs_same_row_reference_mm']['max'] == 0.0
    assert report['jaw'] == {'flips': 2, 'gripper_commands': 2, 'flips_on_command_rows': 2}
    assert abs(report['stationary']['fraction_all_rows'] - 20 / ROWS) < 1e-6
    assert report['stationary']['fraction_sdk_velocity_field'] == 0.0  # SDK velocity field is never under 2 mm/s here
    assert report['images'] == {'stale': 2, 'repeated': 4, 'excluded': 4, 'observations': 2 * ROWS - 4}


def test_finite_difference_speed_first_row_copies_second():
    xyz = np.array([[0, 0, 0], [0.003, 0, 0], [0.003, 0, 0]], np.float32)
    speed = finite_difference_speed(xyz, [0], [3])
    assert speed[0] == speed[1] == pytest.approx(0.09) and speed[2] == 0 < STATIONARY_SPEED_M_PER_S


def test_cross_check_reports_match_and_absence(tmp_path):
    with h5py.File(make_raw(tmp_path / 'raw.h5'), 'r') as h:
        ours = {'train': build_split(h, range(1))}
    assert cross_check_openpi(tmp_path / 'nowhere', ours)['status'] == 'absent'
    theirs = tmp_path / 'openpi'; theirs.mkdir()
    wide = {k: v for k, v in ours['train'].items()}
    wide['actions'] = np.concatenate([wide['actions'], wide['actions'][:, :14]], 1)  # a 30-slot archive
    wide['actions_is_pad'] = np.concatenate([wide['actions_is_pad'], wide['actions_is_pad'][:, :14]], 1)
    wide['source_action_indices'] = np.concatenate([wide['source_action_indices'], wide['source_action_indices'][:, :14]], 1)
    np.savez(theirs / 'aaaa_to_cccc_train.npz', **wide)
    report = cross_check_openpi(theirs, ours)
    assert report['splits']['train']['status'] == 'match'
    wide['states'] = wide['states'].copy(); wide['states'][0, 0] += 1
    np.savez(theirs / 'aaaa_to_cccc_train.npz', **wide)
    assert cross_check_openpi(theirs, ours)['splits']['train']['status'] == 'mismatch'


@pytest.fixture
def prepared(tmp_path):
    raw = make_raw(tmp_path / 'raw.h5')
    with h5py.File(raw, 'r') as h:
        arrays = build_split(h, range(1))
    root = tmp_path / 'dense'; root.mkdir()
    np.savez(root / 'train.npz', **arrays)
    stats = fit_statistics(arrays)
    (root / 'dataset_statistics.json').write_text(json.dumps(stats))
    metadata = {'contract': CONTRACT, 'horizon': HORIZON, 'frameskip': FRAMESKIP, 'raw_path': str(raw),
                'raw_size_bytes': raw.stat().st_size, 'raw_mtime_ns': raw.stat().st_mtime_ns,
                'splits': {'train': {'sha256': sha256(root / 'train.npz'), 'samples': len(arrays['states'])}},
                'statistics_sha256': sha256(root / 'dataset_statistics.json')}
    (root / 'metadata.json').write_text(json.dumps(metadata))
    embeddings = tmp_path / 'embeddings.pkl'
    with embeddings.open('wb') as f:
        pickle.dump({PROMPT: torch.zeros(512, 1024, dtype=torch.bfloat16)}, f)
    return root, embeddings, arrays, stats


def test_dense_dataset_samples(prepared):
    root, embeddings, arrays, stats = prepared
    ds = HanoiDenseDataset(root, embeddings)
    item = ds[0]
    assert item['video'].shape == (3, 25, 224, 224) and item['actions'].shape == (HORIZON, 4) and item['proprio'].shape == (7,)
    assert item['auxiliary_future_source_row'] == int(arrays['source_observation_indices'][0]) + FUTURE_ROWS
    assert -1 <= item['actions'].min() and item['actions'].max() <= 1
    assert item['stationary'] is True and 'cartesian_position' not in item
    late = ds[len(ds) - 1]
    assert late['auxiliary_future_source_row'] == ROWS - 1 and late['actions_is_pad'].all()
    # The future frame at slot 5 (raw frame 17) is the raw future row's image, the current frame is the observation.
    with h5py.File(arrays and root.parent / 'raw.h5', 'r') as h:
        np.testing.assert_array_equal(item['video'][:, 17].permute(1, 2, 0).numpy(), h['pixels'][item['auxiliary_future_source_row']])
        np.testing.assert_array_equal(item['video'][:, 5].permute(1, 2, 0).numpy(), h['pixels'][int(arrays['source_observation_indices'][0])])
    assert ds.resume_data_order_identity['contract'] == CONTRACT
    ds.close()


def test_dense_dataset_rejects_other_contracts(prepared):
    root, embeddings, _, _ = prepared
    metadata = json.loads((root / 'metadata.json').read_text())
    metadata['contract'] = 'hanoi_waypoint_v4_cosmos_v1'
    (root / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='contract'):
        HanoiDenseDataset(root, embeddings)


def test_inference_config_and_contract(tmp_path):
    cfg = HanoiDenseInferenceConfig('ckpt.pt', 'stats.json', 'emb.pkl')
    assert cfg.chunk_size == HORIZON and cfg.config_file.endswith('hanoi_dense_config.py')
    validate_dense_config(cfg)
    cfg.chunk_size = 8
    with pytest.raises(ValueError, match='chunk_size'):
        validate_dense_config(cfg)
    run = tmp_path / 'run'; (run / 'exports').mkdir(parents=True)
    stats = tmp_path / 'stats.json'; stats.write_text('{}')
    (run / 'joint_contract.json').write_text(json.dumps({'contract': CONTRACT, 'statistics_sha256': sha256(stats)}))
    assert validate_checkpoint_contract(run / 'exports' / 'iter_000001000.pt', stats)['contract'] == CONTRACT
    (run / 'joint_contract.json').write_text(json.dumps({'contract': 'hanoi_waypoint_v4_cosmos_v1', 'statistics_sha256': sha256(stats)}))
    with pytest.raises(ValueError, match='different'):
        validate_checkpoint_contract(run / 'exports' / 'iter_000001000.pt', stats)
    assert DEPLOYMENT_CONTRACT['version'] == 5 and DEPLOYMENT_CONTRACT['execution_prefix'] == 8


def test_threshold_jaw():
    actions = np.zeros((HORIZON, 4), np.float32); actions[:, 3] = np.linspace(0, 1, HORIZON)
    out = threshold_jaw(actions)
    assert set(np.unique(out[:, 3])) == {0.0, 1.0} and out[-1, 3] == 1 and out[0, 3] == 0
    with pytest.raises(ValueError):
        threshold_jaw(np.zeros((8, 4)))


def test_video_base_state_normalisation():
    tensors = {'net.blocks.0.w': torch.zeros(2), 'net_ema.blocks.0.w': torch.ones(2), 'tokenizer.x': torch.zeros(1)}
    assert list(normalise_video_base_state({'model': tensors})) == ['blocks.0.w']
    assert list(normalise_video_base_state({'blocks.0.w': torch.zeros(2), 'ema.blocks.0.w': torch.zeros(2)})) == ['blocks.0.w']
    with pytest.raises(ValueError, match='nonempty'):
        normalise_video_base_state({'model': {}})


def test_selection_rule_and_metrics():
    from pathlib import Path
    from examples.hanoi.run_dense import select_checkpoint
    def report(mean, jaw):
        return {'metrics': {'all': {'xyz_mm': {'mean_valid_slots': mean}, 'jaw': {'accuracy_valid_slots': jaw}}}}
    reports = [(report(1.0, 0.98), Path('iter_000001000.pt')), (report(1.5, 0.995), Path('iter_000002000.pt')),
               (report(1.5, 0.999), Path('iter_000003000.pt')), (report(1.2, 0.99), Path('iter_000004000.pt'))]
    (best, path), floor = select_checkpoint(reports)
    assert path.name == 'iter_000004000.pt' and floor
    (best, path), floor = select_checkpoint(reports[:1])
    assert path.name == 'iter_000001000.pt' and not floor
    target = np.zeros((HORIZON, 4), np.float32); target[8:, 3] = 1
    pad = np.zeros(HORIZON, bool); pad[-2:] = True
    predicted = target.copy(); predicted[0, :3] = [0.002, 0, 0]; predicted[8, 3] = 0  # 2 mm slot-1 error, flip one slot late
    m = per_sample_metrics(predicted, target, pad, current_jaw=0.0)
    assert m['slot1_mm'] == pytest.approx(2.0) and m['true_flip_slot'] == 8 and m['predicted_flip_slot'] == 9
    assert flip_slot(np.array([0, 0, 0]), 0) == -1
    assert per_sample_metrics(target, target, np.ones(HORIZON, bool), 0.0) is None  # all slots padded: no score
    m.update({'target_jaw': target[:, 3], 'stationary': False, 'value_abs_error': 0.1, 'future_l1': None, 'future_psnr_db': None})
    s = summarize([m])
    assert s['jaw']['flip_timing_rows_median'] == FRAMESKIP and s['xyz_mm']['slot1_mean'] == pytest.approx(2.0)
    assert s['xyz_mm']['endpoint_mean'] == 0 and s['value_abs_error']['mean'] == pytest.approx(0.1) and 'future_l1' not in s
