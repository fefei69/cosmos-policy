"""Language-goal hindsight labels over the September 24 play recording (contract hanoi_play_k5).

The data is the CIDM manifest's training set, unchanged: 78 whole coverage
walks, 18 one-move crops and 4 expert clips of ``hanoi_wm_20260924_210743``
and the September 25 AAAA-to-CCCC recording, with the manifest's frame filter
(stale or repeated image, non-finite telemetry). Every surviving row is an
observation. Two label rules are applied on top of the raw rows, both at
preparation time so the archive is the complete audit:

* the goal of a row is the board reached at the end of a move chosen once,
  uniformly with a fixed seed, among the row's own move and the next
  ``HORIZON_CAP_MOVES - 1`` moves of its walk (clipped at the walk end; a
  crop or clip has its single move); the goal is conditioning only through the
  board's sentence, ``prompt_for_board``, one fixed template for all 81 boards;
* the dense v5 chunk (slot j is row t + 3 j, absolute XYZ plus jaw intent) is
  cut at the last row of the goal move: later slots take that row's pose and
  jaw and are marked padded, the walk-end rule, so the policy learns to stop
  when the goal board is reached.

No row is removed, added, re-weighted or relabelled in its actions. Held-out
walks are the manifest's validation and test walks with the same rules.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path

import h5py
import numpy as np

from cosmos_policy.datasets.hanoi_dense_data import (
    ACTION_COLUMNS, DEPLOYMENT_CONTRACT, FRAMESKIP, HORIZON, HORIZONS, REFERENCE_RATE_HZ, STATE_COLUMNS,
    STATIONARY_SPEED_M_PER_S, fit_statistics, finite_difference_speed,
)
from cosmos_policy.datasets.hanoi_joint_data import sha256

CONTRACT = 'hanoi_play_k5_cosmos_v1'
HORIZON_CAP_MOVES = 5
GOAL_SEED = 195
RAW_ROOT = Path('/scratch/cw5167/datasets')
PLAY_FILE = 'hanoi_wm_20260924_210743.h5'
EXPERT_FILE = 'hanoi_wm_roundtrip_20260925_171442_AAAA_to_CCCC.h5'
MANIFEST = RAW_ROOT / 'dataset_manifest_v1/manifest.json'
DEFAULT_METADATA = Path('data/hanoi_cosmos/play_k5')
DEFAULT_EMBEDDINGS = Path('data/hanoi_cosmos/t5_embeddings_play.pkl')
FILES = ('play', 'expert')  # file index 0 and 1
EXPERT_EPISODE_OFFSET = 1000  # global episode id of expert episode e is 1000 + e
PEGS = 'ABC'
RINGS = 4
BOARDS = tuple(''.join(b) for b in itertools.product(PEGS, repeat=RINGS))  # peg per ring, smallest ring first
BOARD_INDEX = {board: i for i, board in enumerate(BOARDS)}
FULL_STACKS = tuple(peg * RINGS for peg in PEGS)
STAGE_NAMES = {0: 'unknown', 1: 'open', 2: 'approach_source', 3: 'descend_source', 4: 'grasp', 5: 'lift', 6: 'transit',
               7: 'insert_target', 8: 'release', 9: 'retreat', 10: 'return_initial_xy', 11: 'return_initial_z', 12: 'hold'}
DECISION_STAGES = (2, 6)  # approach_source (which peg to pick from) and transit (where to place): the goal-sensitive rows
PROMPT_TEMPLATE = ('"Goal: peg A <contents>, peg B <contents>, peg C <contents>." with rings numbered 1 (smallest) to 4; '
                   '"is empty", "holds ring N", or "holds rings a, b and c" in ascending order')


def prompt_for_board(board: str) -> str:
    """One sentence per board, pegs always in the order A, B, C; 25 T5 tokens for every board."""
    if len(board) != RINGS or any(peg not in PEGS for peg in board):
        raise ValueError(f'Not a board: {board!r}')
    clauses = []
    for peg in PEGS:
        rings = [str(i + 1) for i in range(RINGS) if board[i] == peg]
        if not rings:
            clauses.append(f'peg {peg} is empty')
        elif len(rings) == 1:
            clauses.append(f'peg {peg} holds ring {rings[0]}')
        else:
            clauses.append(f'peg {peg} holds rings ' + ', '.join(rings[:-1]) + ' and ' + rings[-1])
    return 'Goal: ' + ', '.join(clauses) + '.'


PROMPTS = tuple(prompt_for_board(board) for board in BOARDS)
BOARD_BY_PROMPT = {prompt: board for board, prompt in zip(BOARDS, PROMPTS)}
PROMPTS_SHA256 = hashlib.sha256('\n'.join(PROMPTS).encode()).hexdigest()


def legal_moves(board: str):
    """Boards reachable by one legal move (only the top ring of a peg moves, never onto a smaller ring)."""
    out = []
    for peg in PEGS:
        tops = [i for i in range(RINGS) if board[i] == peg]
        if not tops:
            continue
        ring = tops[0]
        for target in PEGS:
            if target == peg:
                continue
            target_tops = [i for i in range(RINGS) if board[i] == target]
            if not target_tops or target_tops[0] > ring:
                moved = list(board)
                moved[ring] = target
                out.append(''.join(moved))
    return out


def _distances():
    table = np.zeros((len(BOARDS), len(BOARDS)), np.int8)
    for start in BOARDS:
        seen = {start: 0}
        queue = deque([start])
        while queue:
            board = queue.popleft()
            for other in legal_moves(board):
                if other not in seen:
                    seen[other] = seen[board] + 1
                    queue.append(other)
        for board, d in seen.items():
            table[BOARD_INDEX[start], BOARD_INDEX[board]] = d
    return table


DISTANCE = _distances()  # graph distance between boards; diameter 15


def board_string(row):
    """The recording's ``board`` row (peg index per ring) as a board string."""
    return ''.join(PEGS[int(peg)] for peg in row)


DEPLOYMENT_CONTRACT_V7 = {
    **DEPLOYMENT_CONTRACT,
    'version': 7,
    'prompt_rule': 'the request must carry the sentence of the goal board verbatim (prompt_for_board); there is no default goal',
    'prompt_template': PROMPT_TEMPLATE,
    'prompts_sha256': PROMPTS_SHA256,
    'goal_sentences': len(PROMPTS),
    'recording': 'hanoi_wm_20260924_210743 coverage walks (plus manifest crops and expert clips), language-goal hindsight labels',
}


def read_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text())
    train = manifest['train']
    whole = sorted(int(e) for e in train['old_whole_episodes'])
    crops = [{'episode': int(c['episode']), 'frames': [int(c['frames'][0]), int(c['frames'][1])], 'before': c['before'], 'after': c['after']}
             for c in train['old_one_move_crops'] if int(c['episode']) not in whole]  # crops of whole walks are already covered
    clips = [{'episode': int(c['expert_episode']), 'frames': [int(c['frames'][0]), int(c['frames'][1])], 'before': c['before'], 'after': c['after']}
             for c in train['expert_repair_clips']]
    held = manifest['heldout']
    val, test = sorted(int(e) for e in held['old_validation']), sorted(int(e) for e in held['old_test'])
    overlap = (set(whole) & set(val)) | (set(whole) & set(test)) | (set(val) & set(test))
    if overlap or len(whole) != 78 or len(val) != 10 or len(test) != 10:
        raise ValueError('Manifest splits are not the expected disjoint 78/10/10 walks')
    unsuccessful = {int(e) for e in manifest['not_used']['unsuccessful']}
    if unsuccessful & (set(whole) | set(val) | set(test)) or {c['episode'] for c in crops} & unsuccessful:
        raise ValueError('Manifest uses an unsuccessful walk')
    return {'train_walks': whole, 'val_walks': val, 'test_walks': test, 'crops': crops, 'clips': clips,
            'skipped_crops_in_whole_walks': [int(c['episode']) for c in train['old_one_move_crops'] if int(c['episode']) in whole]}


def check_play_raw(handle):
    attrs = handle.attrs
    if int(attrs['schema_version']) != 4 or bool(attrs['dry_run']) or attrs['collection_mode'] != 'original':
        raise ValueError('Expected the real schema-v4 coverage (play) recording')
    if attrs['action_abs_alignment'] != 'post_action_reference':
        raise ValueError('Wrong action alignment')
    if len(handle['ep_len']) != 125 or tuple(handle['pixels'].shape[1:]) != (224, 224, 3):
        raise ValueError('Expected 125 walks of RGB224 rows')
    for key in ('board', 'move_idx', 'motion_stage', 'held_disk', 'robot_telemetry_finite', 'image_stale', 'image_repeated'):
        if key not in handle:
            raise ValueError(f'Recording lacks {key}')


def check_expert_raw(handle):
    attrs = handle.attrs
    if int(attrs['schema_version']) != 4 or bool(attrs['dry_run']) or attrs['direction'] != 'AAAA_to_CCCC':
        raise ValueError('Expected the real schema-v4 AAAA_to_CCCC recording')
    if attrs['action_abs_alignment'] != 'post_action_reference' or len(handle['ep_len']) != 10:
        raise ValueError('Wrong action alignment or episode count')
    if not bool((handle['episode_success'][:] == 1).all()):
        raise ValueError('Every expert episode must be successful')


def load_columns(handle):
    """The small per-row columns needed for labels (never the pixels)."""
    columns = {key: handle[key][:] for key in ('ep_offset', 'ep_len', 'episode_success', 'image_stale', 'image_repeated',
                                                'robot_telemetry_finite', 'board', 'move_idx', 'motion_stage', 'joint_positions',
                                                'proprio', 'reference_pose', 'action_abs')}
    columns['usable'] = ~((columns['image_stale'] > 0) | (columns['image_repeated'] > 0) | (columns['robot_telemetry_finite'] == 0))
    columns['usable'] &= np.isfinite(columns['proprio']).all(1) & np.isfinite(columns['joint_positions']).all(1)
    return columns


def move_ends(columns, lo, hi):
    """(move index -> last usable row of that move) within rows [lo, hi); the board there is the post-move board."""
    rows = np.arange(lo, hi)[columns['usable'][lo:hi]]
    ends = {}
    for row in rows:
        ends[int(columns['move_idx'][row])] = int(row)  # rows ascend, so the last write wins
    return ends


def build_span(columns, file_index, lo, hi, episode_id, rng, *, horizon=HORIZON, cap=HORIZON_CAP_MOVES,
               single_goal_board=None, kind='walk'):
    """Labelled rows for the usable rows of [lo, hi): one goal per row, chunk cut at the goal move's last row."""
    if horizon not in HORIZONS or cap < 1:
        raise ValueError('Bad chunk geometry or horizon cap')
    usable = columns['usable']
    rows = np.arange(lo, hi)[usable[lo:hi]]
    if not len(rows):
        raise ValueError(f'No usable rows in [{lo}, {hi})')
    ends = move_ends(columns, lo, hi)
    moves = sorted(ends)
    end_of = np.array([ends[m] for m in moves])
    move_of_row = columns['move_idx'][rows].astype(np.int64)
    position = {m: k for k, m in enumerate(moves)}
    goal_end = np.empty(len(rows), np.int64)
    goal_moves_ahead = np.empty(len(rows), np.int64)
    for k, row in enumerate(rows):
        m = position[int(move_of_row[k])]
        if single_goal_board is not None:
            choice = len(moves) - 1  # a crop or clip: its single board change, whatever the stage bookkeeping says
        else:
            choice = m + int(rng.integers(min(cap, len(moves) - m)))
        goal_end[k] = end_of[choice]
        goal_moves_ahead[k] = choice - m
    if single_goal_board is not None:
        reached = board_string(columns['board'][int(end_of[-1])])
        if reached != single_goal_board:
            raise ValueError(f'Segment [{lo}, {hi}) ends on {reached}, manifest says {single_goal_board}')
    targets = rows[:, None] + FRAMESKIP * np.arange(1, horizon + 1)[None, :]
    pads = targets > goal_end[:, None]
    targets = np.minimum(targets, goal_end[:, None])
    states = np.concatenate([columns['joint_positions'][rows], columns['proprio'][rows, 6:7]], -1).astype(np.float32)
    actions = np.concatenate([columns['reference_pose'][targets][..., :3], columns['action_abs'][targets][..., 3:4]], -1).astype(np.float32)
    for name, value in (('states', states), ('actions', actions)):
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite {name}')
    if not np.isin(actions[..., 3], [0, 1]).all():
        raise ValueError('Jaw labels must be absolute binary intent')
    current = np.array([BOARD_INDEX[board_string(b)] for b in columns['board'][rows]], np.int64)
    goal = np.array([BOARD_INDEX[board_string(b)] for b in columns['board'][goal_end]], np.int64)
    next_board = np.array([BOARD_INDEX[board_string(columns['board'][int(end_of[position[int(m)]])])] for m in move_of_row], np.int64)
    speed = finite_difference_speed(columns['proprio'][lo:hi, :3], np.array([0]), np.array([hi - lo]))[rows - lo]
    n = len(rows)
    return {
        'source_observation_indices': rows.astype(np.int64), 'states': states, 'actions': actions,
        'actions_is_pad': pads, 'source_action_indices': targets.astype(np.int64),
        'episode_indices': np.full(n, episode_id, np.int64),
        'source_episode_bounds': np.stack([np.full(n, lo, np.int64), goal_end + 1], 1),  # the chunk's world ends at the goal move
        'segment_bounds': np.tile([lo, hi], (n, 1)).astype(np.int64),
        'file_indices': np.full(n, file_index, np.int64), 'segment_kinds': np.full(n, {'walk': 0, 'crop': 1, 'clip': 2}[kind], np.int64),
        'move_indices': move_of_row, 'motion_stages': columns['motion_stage'][rows].astype(np.int64),
        'board_indices': current, 'goal_board_indices': goal, 'next_board_indices': next_board,
        'goal_end_rows': goal_end, 'goal_moves_ahead': goal_moves_ahead,
        'goal_graph_distance': DISTANCE[current, goal].astype(np.int64),
        'cartesian_positions': columns['proprio'][rows, :3].astype(np.float32),
        'measured_speed_m_per_s': speed.astype(np.float32), 'stationary': speed < STATIONARY_SPEED_M_PER_S,
        'state_columns': np.array(STATE_COLUMNS), 'action_columns': np.array(ACTION_COLUMNS), 'validated': np.array(True),
    }


def concatenate(parts):
    out = {}
    for key in parts[0]:
        values = [p[key] for p in parts]
        if values[0].ndim == 0 or key in ('state_columns', 'action_columns', 'validated'):
            for v in values[1:]:
                if not np.array_equal(v, values[0]):
                    raise ValueError(f'Inconsistent {key} across segments')
            out[key] = values[0]
        else:
            out[key] = np.concatenate(values)
    return out


def walk_bounds(columns, episode):
    if int(columns['episode_success'][int(columns['ep_offset'][episode])]) != 1:
        raise ValueError(f'Walk {episode} was not successful')
    lo = int(columns['ep_offset'][episode])
    return lo, lo + int(columns['ep_len'][episode])


def build_split(play, expert, manifest, split, rng, horizon=HORIZON, cap=HORIZON_CAP_MOVES):
    """Labelled rows of one split; ``rng`` draws the goals (one per row, in archive order)."""
    parts, counts = [], {'walk': 0, 'crop': 0, 'clip': 0}
    for episode in manifest[f'{split}_walks']:
        lo, hi = walk_bounds(play, episode)
        part = build_span(play, 0, lo, hi, episode, rng, horizon=horizon, cap=cap)
        counts['walk'] += len(part['states'])
        parts.append(part)
    if split == 'train':
        for crop in manifest['crops']:
            lo, hi = walk_bounds(play, crop['episode'])
            a, b = crop['frames']
            if not 0 <= a <= b < hi - lo:
                raise ValueError('Crop span outside its walk')
            part = build_span(play, 0, lo + a, lo + b + 1, crop['episode'], rng, horizon=horizon, cap=cap,
                              single_goal_board=crop['after'], kind='crop')
            counts['crop'] += len(part['states'])
            parts.append(part)
        for clip in manifest['clips']:
            lo, hi = walk_bounds(expert, clip['episode'])
            a, b = clip['frames']
            if not 0 <= a <= b < hi - lo:
                raise ValueError('Clip span outside its episode')
            part = build_span(expert, 1, lo + a, lo + b + 1, EXPERT_EPISODE_OFFSET + clip['episode'], rng, horizon=horizon, cap=cap,
                              single_goal_board=clip['after'], kind='clip')
            counts['clip'] += len(part['states'])
            parts.append(part)
    return concatenate(parts), counts


def label_audit(arrays):
    """What the archive's labels look like: goal horizon, graph distance, progress of the labelled move, goal boards, stages."""
    current, goal, following = arrays['board_indices'], arrays['goal_board_indices'], arrays['next_board_indices']
    before, after = DISTANCE[current, goal].astype(int), DISTANCE[following, goal].astype(int)
    reached = current == goal
    moving = ~reached
    kinds = {'goal_reached': float(reached.mean()),
             'progress': float((after[moving] < before[moving]).mean()) if moving.any() else None,
             'lateral': float((after[moving] == before[moving]).mean()) if moving.any() else None,
             'regress': float((after[moving] > before[moving]).mean()) if moving.any() else None}
    stages = {STAGE_NAMES.get(int(s), str(s)): int(c) for s, c in zip(*np.unique(arrays['motion_stages'], return_counts=True))}
    return {
        'rows': int(len(current)),
        'goal_moves_ahead_histogram': {int(k): int(v) for k, v in zip(*np.unique(arrays['goal_moves_ahead'], return_counts=True))},
        'goal_graph_distance_histogram': {int(k): int(v) for k, v in zip(*np.unique(arrays['goal_graph_distance'], return_counts=True))},
        'labelled_move_vs_goal': kinds,
        'per_goal_board_rows': {BOARDS[int(k)]: int(v) for k, v in zip(*np.unique(goal, return_counts=True))},
        'full_stack_goal_rows': {b: int((goal == BOARD_INDEX[b]).sum()) for b in FULL_STACKS},
        'rows_per_stage': stages,
        'decision_rows': int(np.isin(arrays['motion_stages'], DECISION_STAGES).sum()),
        'distinct_board_goal_pairs': int(len(np.unique(current * len(BOARDS) + goal))),
        'padded_slots': int(arrays['actions_is_pad'].sum()),
        'all_hold_chunks': int(arrays['actions_is_pad'].all(1).sum()),
    }


def prepare(output=DEFAULT_METADATA, manifest_path=MANIFEST, horizon=HORIZON, cap=HORIZON_CAP_MOVES, seed=GOAL_SEED,
            play_path=RAW_ROOT / PLAY_FILE, expert_path=RAW_ROOT / EXPERT_FILE):
    output = Path(output)
    if horizon not in HORIZONS:
        raise ValueError(f'Horizon must be one of {HORIZONS}')
    if output.exists():
        raise FileExistsError(f'Refusing to replace an existing dataset identity: {output}')
    manifest = read_manifest(manifest_path)
    files = []
    for index, path in enumerate((Path(play_path), Path(expert_path))):
        print(f'Hashing {path.name}...', flush=True)
        stat = path.stat()
        files.append({'index': index, 'role': FILES[index], 'path': str(path.resolve()), 'sha256': sha256(path),
                      'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    with h5py.File(play_path, 'r') as play_handle, h5py.File(expert_path, 'r') as expert_handle:
        check_play_raw(play_handle)
        check_expert_raw(expert_handle)
        play, expert = load_columns(play_handle), load_columns(expert_handle)
        files[0]['episodes'], files[1]['episodes'] = int(len(play['ep_len'])), int(len(expert['ep_len']))
    splits, counts, audits = {}, {}, {}
    for split in ('train', 'val', 'test'):
        rng = np.random.default_rng([seed, {'train': 0, 'val': 1, 'test': 2}[split]])
        splits[split], counts[split] = build_split(play, expert, manifest, split, rng, horizon, cap)
        audits[split] = label_audit(splits[split])
        print(f'{split}: {len(splits[split]["states"])} rows ({counts[split]})', flush=True)
    statistics = fit_statistics(splits['train'])
    output.mkdir(parents=True)
    try:
        for split, arrays in splits.items():
            np.savez(output / f'{split}.npz', **arrays)
        (output / 'dataset_statistics.json').write_text(json.dumps(statistics, indent=2) + '\n')
        metadata = {
            'contract': CONTRACT, 'format_version': 1,
            'horizon': int(horizon), 'frameskip': FRAMESKIP, 'reference_rate_hz': REFERENCE_RATE_HZ,
            'horizon_cap_moves': int(cap), 'goal_seed': int(seed),
            'prompt_template': PROMPT_TEMPLATE, 'prompts_sha256': PROMPTS_SHA256, 'prompts': list(PROMPTS), 'boards': list(BOARDS),
            'files': files,
            'manifest': {'path': str(Path(manifest_path).resolve()), 'sha256': sha256(manifest_path),
                         'train_walks': manifest['train_walks'], 'val_walks': manifest['val_walks'], 'test_walks': manifest['test_walks'],
                         'crops': manifest['crops'], 'clips': manifest['clips'],
                         'skipped_crops_in_whole_walks': manifest['skipped_crops_in_whole_walks']},
            'global_episode_id': f'play walk index; expert episode index plus {EXPERT_EPISODE_OFFSET}',
            'observation_rule': 'every row of a manifest training walk, crop or clip whose image is neither stale nor repeated and whose telemetry is finite',
            'goal_rule': (f'one goal per row, drawn once (seed {seed}) uniformly over the row\'s own move and the next {cap - 1} moves of its walk, '
                          'clipped at the walk end; the goal board is the board at the last usable row of that move; a crop or clip has its single move'),
            'label_rule': ('slot j is row t + 3 j: XYZ = reference_pose[row, 0:3], jaw = action_abs[row, 3]; slots past the last row of the goal move '
                           'repeat that row and are padded (trained as hold targets, excluded from metrics)'),
            'conditioning': 'the goal board sentence (T5-11B embedding, one of 81) is the only goal signal; no goal image, no task id',
            'stationary_definition': f'finite-difference speed of measured XYZ at 30 Hz below {STATIONARY_SPEED_M_PER_S} m/s',
            'splits': {split: {'sha256': sha256(output / f'{split}.npz'), 'samples': int(len(z['states'])),
                               'episodes': sorted(map(int, set(z['episode_indices']))), 'rows_by_kind': counts[split],
                               'stationary_fraction': float(z['stationary'].mean()), 'audit': audits[split]} for split, z in splits.items()},
            'statistics_sha256': sha256(output / 'dataset_statistics.json'),
            'normalization': 'Cosmos min/max over the training split (states; actions over valid chunk slots); mean/std recorded; no clipping',
            'state': 'six measured joint angles radians, measured gripper stroke metres; no velocity, no XYZ',
            'actions': f'{horizon} x [absolute reference XYZ metres, jaw intent] at 10 Hz',
            'future_auxiliary': f'raw row min(t + {FRAMESKIP * horizon}, last row of the goal move) of the same file, image and measured joint/jaw labels only; not conditioning input',
            'value_auxiliary': 'discounted goal arrival from the auxiliary raw row, gamma=0.9995 per raw tick, horizon = the goal move end',
            'image_preprocessing': 'stored RGB224 deployment crop, never re-cropped, no augmentation; native Wan VAE synthetic latent-slot packing',
            'deployment': DEPLOYMENT_CONTRACT_V7,
        }
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    except BaseException:
        (output / 'PREPARATION_FAILED').touch()
        raise
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--horizon', type=int, default=HORIZON, choices=HORIZONS)
    parser.add_argument('--cap', type=int, default=HORIZON_CAP_MOVES, help='Hindsight horizon cap in moves')
    parser.add_argument('--seed', type=int, default=GOAL_SEED)
    arguments = parser.parse_args()
    result = prepare(arguments.output, arguments.manifest, arguments.horizon, arguments.cap, arguments.seed)
    print(json.dumps({key: value for key, value in result.items() if key not in ('deployment', 'prompts', 'boards')}, indent=2))
