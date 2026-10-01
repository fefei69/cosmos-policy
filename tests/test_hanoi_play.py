"""hanoi_play_k5 regressions: goal sentences, the Hanoi graph, manifest reading, hindsight labels with the horizon cap, dataset, policy prompt rule."""
import json
import pickle

import h5py
import numpy as np
import pytest
import torch

from cosmos_policy.datasets.hanoi_dense_data import FRAMESKIP, HORIZON
from cosmos_policy.datasets.hanoi_play_data import (
    BOARDS, BOARD_BY_PROMPT, BOARD_INDEX, CONTRACT, DECISION_STAGES, DISTANCE, EXPERT_EPISODE_OFFSET, FULL_STACKS,
    HORIZON_CAP_MOVES, PROMPTS, PROMPTS_SHA256, board_string, build_split, label_audit, legal_moves, load_columns,
    prepare, prompt_for_board, read_manifest,
)
from cosmos_policy.datasets.hanoi_play_dataset import HanoiPlayDataset
from cosmos_policy.datasets.hanoi_joint_data import sha256

ROWS_PER_MOVE = 12
STAGES = [1, 2, 2, 3, 4, 5, 6, 6, 7, 8, 9, 9]  # open .. retreat; the board reads the post-move board from the release row on
RELEASE_AT = 9


def random_route(rng, start, moves):
    route = [start]
    for _ in range(moves):
        route.append(str(rng.choice(legal_moves(route[-1]))))
    return route


def make_play_raw(path, walks=125, moves=2, start='AAAA', direction=None, failed=(), seed=3):
    """A synthetic coverage recording: ``walks`` walks of ``moves`` legal moves plus one terminal hold row each."""
    rng = np.random.default_rng(seed)
    rows_per_walk = ROWS_PER_MOVE * moves + 1
    n = rows_per_walk * walks
    board = np.zeros((n, 4), np.int8); move_idx = np.zeros(n, np.int64); stage = np.zeros(n, np.int8); held = np.zeros(n, np.int8)
    success = np.ones(n, np.int64)
    routes = {}
    current = start
    for w in range(walks):
        route = random_route(rng, current, moves); routes[w] = route
        base = w * rows_per_walk
        for m in range(moves):
            for k in range(ROWS_PER_MOVE):
                r = base + m * ROWS_PER_MOVE + k
                b = route[m + 1] if k >= RELEASE_AT else route[m]
                board[r] = ['ABC'.index(p) for p in b]; move_idx[r] = m; stage[r] = STAGES[k]; held[r] = 1 if 4 <= STAGES[k] <= 7 else 0
        last = base + rows_per_walk - 1
        board[last] = ['ABC'.index(p) for p in route[-1]]; move_idx[last] = moves; stage[last] = 12
        if w in failed:
            success[base:base + rows_per_walk] = 0
        current = route[-1]
    t = np.arange(n)
    xyz = np.stack([0.4 + 0.002 * (t % rows_per_walk), 0.001 * (board[:, 0] - 1), 0.1 + 0.0005 * (t % ROWS_PER_MOVE)], 1).astype(np.float32)
    with h5py.File(path, 'w') as h:
        h.attrs['schema_version'] = 4; h.attrs['dry_run'] = False; h.attrs['action_abs_alignment'] = 'post_action_reference'
        h.attrs['collection_mode'] = 'roundtrip' if direction else 'original'
        if direction:
            h.attrs['direction'] = direction
        h['pixels'] = (t[:, None, None, None] % 255).astype(np.uint8) * np.ones((1, 224, 224, 3), np.uint8)
        h['ep_offset'] = np.arange(walks, dtype=np.int64) * rows_per_walk
        h['ep_len'] = np.full(walks, rows_per_walk, np.int32)
        h['episode_success'] = success
        h['reference_pose'] = np.concatenate([xyz, np.tile([0, np.pi / 4, 0], (n, 1))], 1).astype(np.float32)
        jaw = (stage >= 4).astype(np.float32) * (stage <= 7)
        h['action_abs'] = np.concatenate([xyz, jaw[:, None]], 1).astype(np.float32)
        proprio = np.zeros((n, 8), np.float32); proprio[:, :3] = xyz + 0.0004; proprio[:, 6] = 0.02 + 0.001 * jaw
        h['proprio'] = proprio
        h['joint_positions'] = (0.01 * np.arange(6)[None, :] + 0.001 * (t % rows_per_walk)[:, None]).astype(np.float32)
        stale = np.zeros(n, np.uint8); stale[t % ROWS_PER_MOVE == 5] = 1
        h['image_stale'] = stale; h['image_repeated'] = np.zeros(n, np.uint8)
        finite = np.ones(n, np.uint8); finite[t % rows_per_walk == 7] = 0
        h['robot_telemetry_finite'] = finite
        h['board'] = board; h['move_idx'] = move_idx; h['motion_stage'] = stage; h['held_disk'] = held
        h['gripper_command_issued'] = np.zeros(n, np.uint8); h['goal_board'] = np.tile([2, 2, 2, 2], (n, 1)).astype(np.int8)
    return path, routes


def write_manifest(path, play_routes, expert_routes, train_walks, val_walks, test_walks):
    """A manifest with one crop in a non-whole walk, one crop in a whole walk (to be skipped) and one expert clip."""
    crop_walk = max(set(play_routes) - set(train_walks) - set(val_walks) - set(test_walks))
    crops = [{'episode': crop_walk, 'frames': [ROWS_PER_MOVE, 2 * ROWS_PER_MOVE - 1], 'before': play_routes[crop_walk][1], 'after': play_routes[crop_walk][2]},
             {'episode': train_walks[0], 'frames': [0, ROWS_PER_MOVE - 1], 'before': play_routes[train_walks[0]][0], 'after': play_routes[train_walks[0]][1]}]
    clips = [{'expert_episode': 0, 'frames': [0, ROWS_PER_MOVE - 1], 'before': expert_routes[0][0], 'after': expert_routes[0][1]}]
    manifest = {'train': {'old_whole_episodes': train_walks, 'old_one_move_crops': crops, 'expert_repair_clips': clips},
                'heldout': {'old_validation': val_walks, 'old_test': test_walks}, 'not_used': {'unsuccessful': [12]}}
    path.write_text(json.dumps(manifest))
    return path, crops, clips


@pytest.fixture
def synthetic(tmp_path):
    play, routes = make_play_raw(tmp_path / 'play.h5', failed=(12,))
    expert, expert_routes = make_play_raw(tmp_path / 'expert.h5', walks=10, moves=1, direction='AAAA_to_CCCC', seed=5)
    train = [w for w in range(125) if w % 5 != 1 and w != 12][:78]
    rest = [w for w in range(125) if w not in train and w != 12]
    val, test = rest[:10], rest[10:20]
    manifest, crops, clips = write_manifest(tmp_path / 'manifest.json', routes, expert_routes, train, val, test)
    return {'play': play, 'expert': expert, 'manifest': manifest, 'routes': routes, 'expert_routes': expert_routes,
            'train': train, 'val': val, 'test': test, 'crops': crops, 'clips': clips}


def test_goal_sentences_and_graph():
    assert len(PROMPTS) == 81 and len(set(PROMPTS)) == 81 and len(BOARDS) == 81
    assert prompt_for_board('AAAA') == 'Goal: peg A holds rings 1, 2, 3 and 4, peg B is empty, peg C is empty.'
    assert prompt_for_board('CABA') == 'Goal: peg A holds rings 2 and 4, peg B holds ring 3, peg C holds ring 1.'
    assert prompt_for_board('BAAA') == 'Goal: peg A holds rings 2, 3 and 4, peg B holds ring 1, peg C is empty.'
    for board, prompt in zip(BOARDS, PROMPTS):
        assert BOARD_BY_PROMPT[prompt] == board and prompt.startswith('Goal: peg A') and prompt.endswith('.')
    assert FULL_STACKS == ('AAAA', 'BBBB', 'CCCC') and len(PROMPTS_SHA256) == 64
    with pytest.raises(ValueError):
        prompt_for_board('AAAD')
    assert DISTANCE.max() == 15 and np.array_equal(DISTANCE, DISTANCE.T) and (np.diag(DISTANCE) == 0).all()
    assert DISTANCE[BOARD_INDEX['AAAA'], BOARD_INDEX['CCCC']] == 15 and DISTANCE[BOARD_INDEX['AAAA'], BOARD_INDEX['BAAA']] == 1
    assert sorted(legal_moves('AAAA')) == ['BAAA', 'CAAA'] and len(legal_moves('CABA')) == 3
    assert board_string(np.array([1, 0, 0, 2], np.int8)) == 'BAAC'


def test_manifest_reading(synthetic):
    manifest = read_manifest(synthetic['manifest'])
    assert manifest['train_walks'] == synthetic['train'] and manifest['val_walks'] == synthetic['val'] and manifest['test_walks'] == synthetic['test']
    assert len(manifest['crops']) == 1 and manifest['skipped_crops_in_whole_walks'] == [synthetic['train'][0]]
    assert len(manifest['clips']) == 1 and manifest['clips'][0]['episode'] == 0


def test_labels_follow_the_cap_and_the_cut(synthetic):
    manifest = read_manifest(synthetic['manifest'])
    with h5py.File(synthetic['play'], 'r') as p, h5py.File(synthetic['expert'], 'r') as e:
        play, expert = load_columns(p), load_columns(e)
        rows_train, counts = build_split(play, expert, manifest, 'train', np.random.default_rng(1), cap=HORIZON_CAP_MOVES)
        rows_val, val_counts = build_split(play, expert, manifest, 'val', np.random.default_rng(2))
    usable_per_walk = 2 * ROWS_PER_MOVE + 1 - 2 - 1  # two stale rows and one non-finite row per walk
    assert counts == {'walk': 78 * usable_per_walk, 'crop': ROWS_PER_MOVE - 1, 'clip': ROWS_PER_MOVE - 2}  # the clip also loses its non-finite row 7
    assert val_counts == {'walk': 10 * usable_per_walk, 'crop': 0, 'clip': 0}
    z = rows_train
    n = len(z['states'])
    assert z['actions'].shape == (n, HORIZON, 4) and z['goal_moves_ahead'].max() <= HORIZON_CAP_MOVES - 1
    # the goal board is the board at the goal move's last usable row, and the chunk is cut there
    for i in range(0, n, 7):
        end = int(z['goal_end_rows'][i]); row = int(z['source_observation_indices'][i])
        assert end >= row and z['source_episode_bounds'][i, 1] == end + 1
        expected_targets = np.minimum(row + FRAMESKIP * np.arange(1, HORIZON + 1), end)
        np.testing.assert_array_equal(z['source_action_indices'][i], expected_targets)
        np.testing.assert_array_equal(z['actions_is_pad'][i], row + FRAMESKIP * np.arange(1, HORIZON + 1) > end)
        columns = play if z['file_indices'][i] == 0 else expert
        assert BOARDS[int(z['goal_board_indices'][i])] == board_string(columns['board'][end])
        assert z['goal_graph_distance'][i] == DISTANCE[z['board_indices'][i], z['goal_board_indices'][i]]
    # the terminal hold row of each walk: goal = its own board, all slots padded to that pose
    hold = z['motion_stages'] == 12
    expected_hold = sum(bool(play['usable'][int(play['ep_offset'][w] + play['ep_len'][w] - 1)]) for w in manifest['train_walks'])
    assert hold.sum() == expected_hold > 0 and (z['goal_board_indices'][hold] == z['board_indices'][hold]).all() and z['actions_is_pad'][hold].all()
    # a row in the last move whose goal is its own move ends within the walk: pads beyond the walk end as before
    # crops and clips carry their single board change as the goal
    crop = z['segment_kinds'] == 1
    assert crop.sum() == ROWS_PER_MOVE - 1 and {BOARDS[int(g)] for g in z['goal_board_indices'][crop]} == {synthetic['crops'][0]['after']}
    clip = z['segment_kinds'] == 2
    assert clip.sum() == ROWS_PER_MOVE - 2 and (z['file_indices'][clip] == 1).all() and (z['episode_indices'][clip] == EXPERT_EPISODE_OFFSET).all()
    assert {BOARDS[int(g)] for g in z['goal_board_indices'][clip]} == {synthetic['clips'][0]['after']}
    # every row's goal is reachable within the cap along its own walk, so the moves-ahead histogram stops at cap - 1
    audit = label_audit(z)
    assert set(audit['goal_moves_ahead_histogram']) <= set(range(HORIZON_CAP_MOVES)) and audit['rows'] == n
    assert audit['decision_rows'] == int(np.isin(z['motion_stages'], DECISION_STAGES).sum()) > 0
    assert abs(audit['labelled_move_vs_goal']['progress'] + audit['labelled_move_vs_goal']['lateral'] + audit['labelled_move_vs_goal']['regress'] - 1) < 1e-9


def test_prepare_and_dataset(synthetic, tmp_path):
    out = tmp_path / 'play_k5'
    metadata = prepare(out, synthetic['manifest'], play_path=synthetic['play'], expert_path=synthetic['expert'])
    assert metadata['contract'] == CONTRACT and metadata['horizon_cap_moves'] == HORIZON_CAP_MOVES and metadata['prompts_sha256'] == PROMPTS_SHA256
    assert metadata['splits']['train']['rows_by_kind']['crop'] == ROWS_PER_MOVE - 1 and metadata['splits']['test']['rows_by_kind'] == {'walk': 10 * (2 * ROWS_PER_MOVE - 2), 'crop': 0, 'clip': 0}
    assert metadata['files'][0]['role'] == 'play' and metadata['files'][1]['role'] == 'expert' and metadata['manifest']['sha256'] == sha256(synthetic['manifest'])
    with pytest.raises(FileExistsError):
        prepare(out, synthetic['manifest'], play_path=synthetic['play'], expert_path=synthetic['expert'])
    embeddings = tmp_path / 'embeddings.pkl'
    with embeddings.open('wb') as f:
        pickle.dump({prompt: torch.full((1, 512, 1024), float(i + 1), dtype=torch.bfloat16) for i, prompt in enumerate(PROMPTS)}, f)
    ds = HanoiPlayDataset(out, embeddings, split='train')
    assert len(ds) == metadata['splits']['train']['samples'] and ds.horizon == HORIZON
    for i in (0, len(ds) // 2, len(ds) - 1):
        item = ds[i]; raw = ds.raw_example(i)
        goal = int(item['goal_board_index'])
        assert raw['prompt'] == PROMPTS[goal] == prompt_for_board(raw['goal_board']) and float(item['t5_text_embeddings'][0, 0]) == goal + 1
        assert item['actions'].shape == (HORIZON, 4) and item['video'].shape == (3, 25, 224, 224)
        assert item['auxiliary_future_source_row'] <= int(ds.arrays['goal_end_rows'][i])
    assert len(ds._handles) >= 1
    ds.close()
    assert ds._handles == {}
    val = HanoiPlayDataset(out, embeddings, split='val', representative_order=True)
    assert len(val) == metadata['splits']['val']['samples'] and set(val.order.tolist()) == set(range(len(val)))
    bad = pickle.load(embeddings.open('rb')); bad[PROMPTS[1]] = bad[PROMPTS[0]]
    with (tmp_path / 'dup.pkl').open('wb') as f:
        pickle.dump(bad, f)
    with pytest.raises(ValueError, match='identical'):
        HanoiPlayDataset(out, tmp_path / 'dup.pkl', split='val')


def test_policy_requires_a_verbatim_goal_sentence(tmp_path):
    from cosmos_policy.experiments.robot.hanoi.play_policy import HanoiPlayInferenceConfig, resolve_goal, validate_checkpoint_contract
    assert resolve_goal(prompt_for_board('BAAA')) == 'BAAA' and resolve_goal(PROMPTS[BOARD_INDEX['CCCC']]) == 'CCCC'
    for bad in ('Move all four rings from peg A to peg C following Tower of Hanoi rules.', PROMPTS[0].lower(), PROMPTS[0][:-1], '', None):
        with pytest.raises(ValueError, match='verbatim'):
            resolve_goal(bad)
    cfg = HanoiPlayInferenceConfig('ckpt.pt', 'stats.json', 'emb.pkl')
    assert cfg.chunk_size == HORIZON and cfg.config_file.endswith('hanoi_play_config.py') and cfg.config.endswith('play__inference')
    run = tmp_path / 'run'; (run / 'exports').mkdir(parents=True)
    stats = tmp_path / 'stats.json'; stats.write_text('{}')
    emb = tmp_path / 'emb.pkl'; emb.write_bytes(b'x')
    export = run / 'exports' / 'iter_000002000.pt'
    (run / 'joint_contract.json').write_text(json.dumps({'contract': CONTRACT, 'statistics_sha256': sha256(stats), 'prompts_sha256': PROMPTS_SHA256,
                                                         'embeddings_sha256': sha256(emb), 'horizon': HORIZON}))
    assert validate_checkpoint_contract(export, stats, emb)['contract'] == CONTRACT
    (run / 'joint_contract.json').write_text(json.dumps({'contract': CONTRACT, 'statistics_sha256': sha256(stats), 'prompts_sha256': 'other',
                                                         'embeddings_sha256': sha256(emb), 'horizon': HORIZON}))
    with pytest.raises(ValueError, match='goal sentences'):
        validate_checkpoint_contract(export, stats, emb)


def test_probe_pairs_need_same_board_close_state_and_different_goal(synthetic, tmp_path):
    from cosmos_policy.experiments.robot.hanoi.run_hanoi_play_eval import decision_rows, matched_goal_pairs
    manifest = read_manifest(synthetic['manifest'])
    with h5py.File(synthetic['play'], 'r') as p, h5py.File(synthetic['expert'], 'r') as e:
        z, _ = build_split(load_columns(p), load_columns(e), manifest, 'train', np.random.default_rng(1))
    rows = decision_rows(z)
    assert len(rows) and set(z['motion_stages'][rows].tolist()) <= set(DECISION_STAGES)
    pairs = matched_goal_pairs(z, rows, match_mm=1e9, decision_mm=0.0)
    for i, j, gap in pairs:
        assert z['board_indices'][i] == z['board_indices'][j] and z['goal_board_indices'][i] != z['goal_board_indices'][j] and gap > 0
    assert matched_goal_pairs(z, rows, match_mm=0.0, decision_mm=1e9) == []


def test_config_and_launcher_import(monkeypatch):
    import importlib
    import subprocess
    monkeypatch.setenv('COSMOS_POLICY_PLATFORM', 'hanoi_dense')
    monkeypatch.setenv('HANOI_DENSE_HORIZON', str(HORIZON))
    try:
        config = importlib.import_module('cosmos_policy.config.hanoi_play_config')
    except subprocess.CalledProcessError:  # the base config probes the CUDA runtime, absent on login nodes
        pytest.skip('CUDA runtime libraries unavailable on this host')
    assert config.MAX_UPDATES == 32000 and config.SAVE_EVERY == 2000 and config.EXPERIMENT == 'cosmos_predict2_2b_hanoi_play'
    launcher = importlib.import_module('examples.hanoi.run_play')
    assert launcher.run_name_for('video') == 'hanoi_cosmos_play_20260930_video_init'
    assert 'cosmos_policy/datasets/hanoi_play_data.py' in launcher.CODE_PATHS and launcher.STAGE_EVAL['stride'] == 27
