# Handover: train pi0.5 on the play recording with the Cosmos play_k5 labels

Written September 30, 2026, by the Cosmos-side agent for the OpenPI agent.
Goal: a pi0.5 model trained on exactly the rows, goals, action labels and
goal sentences that the Cosmos run `hanoi_cosmos_play_20260930_video_init`
uses, so the two policies are comparable offline and on the arm. Do not write
under the cosmos-policy tree; read from it freely. The Cosmos reference
implementation is `cosmos_policy/datasets/hanoi_play_data.py`; the design and
its reasons are in `docs/hanoi_play_composition_related_work.md`, sections
6.10 to 6.12.

## 1. What is being tested

The play recording contains every single move but none of the six 15-move
tower tasks (two of them are absent entirely, the others only as noisy
fragments). Both policies learn from hindsight-labelled segments of at most
five moves; at test time they are asked for a full-stack goal 15 moves away,
which never occurs in training. The world-model side composes with a planner;
the question is whether imitation composes on its own. Nothing is curated:
every usable row is a training row, and no action is edited.

## 2. Data

Two schema-v4 recordings in `/scratch/cw5167/datasets/`:

| Role | File | SHA-256 (prefix) |
|---|---|---|
| play | `hanoi_wm_20260924_210743.h5` (125 coverage walks of 20 moves, 1.3M rows) | `c958b447` |
| expert | `hanoi_wm_roundtrip_20260925_171442_AAAA_to_CCCC.h5` (10 episodes) | `74fa5a6f` |

and the CIDM manifest `/scratch/cw5167/datasets/dataset_manifest_v1/manifest.json`
(README beside it), which defines the split:

- train: the 78 whole walks in `train.old_whole_episodes`, the 18 one-move
  crops in `train.old_one_move_crops` (frame spans inclusive; the four crops
  that lie inside whole walks 15 and 104 are already covered and are skipped),
  and the 4 expert clips in `train.expert_repair_clips`;
- validation: the 10 walks in `heldout.old_validation`; test: `heldout.old_test`.
  Walk 12 (unsuccessful) is never used. The expert episodes 6 to 9 are not used.

**Frame filter** (the manifest's): drop rows with `image_stale` or
`image_repeated` set or `robot_telemetry_finite` zero. Every surviving row is
an observation. Expected counts:

| Split | Rows | Walk / crop / clip rows |
|---|---|---|
| train | 800,571 | 785,001 / 12,634 / 2,936 |
| validation | 100,342 | walks only |
| test | 100,163 | walks only |

## 3. Labels

**Goal per row.** For a row in move m of a walk, the goal move is drawn once,
uniformly, among moves m to m + 4 (clipped at the walk's last move; the walk's
terminal hold rows have their own board as the goal; a crop or clip has its
single board change). The goal board is the recording's `board` at the last
usable row of the goal move (that row is the `retreat` stage and already
reads the post-move board). The draw is seeded, so **take the goal from the
Cosmos archive rather than re-drawing it**: `data/hanoi_cosmos/play_k5/{train,val,test}.npz`
carry, per row, `file_indices` (0 play, 1 expert), `source_observation_indices`
(row in that file), `goal_board_indices` (index into the 81 boards, peg per
ring with ring 1 the smallest, `itertools.product('ABC', repeat=4)` order),
`goal_end_rows` (last row of the goal move), `move_indices`, `motion_stages`,
`goal_moves_ahead` and `goal_graph_distance`. Using these makes the two
policies see identical (row, goal) pairs. Reading the archive is fine; do not
copy it into your tree as a dataset.

**Action chunk.** The dense v5 rule with a cut: slot j = 1..16 is raw row
t + 3 j, XYZ from `reference_pose[row, 0:3]`, jaw intent from
`action_abs[row, 3]`, absolute base-frame metres at 10 Hz; slots whose row
lies past `goal_end_rows` take that row's pose and jaw and are marked padded
(in Cosmos they train as hold targets and are excluded from metrics). Rows
after the goal move's release therefore carry an all-hold chunk: the policy
learns to stop when the goal board is reached. Keep your execution prefix
and asynchronous re-planning as in your six-task run.

**State**: six measured joint angles plus the measured jaw stroke
(`joint_positions`, `proprio[:, 6]`), no velocity, no XYZ. Image: the stored
224 x 224 RGB, no augmentation. Normalisation over the combined training
split.

Cross-check before training: per split, your rows, states and the first 16
slots (with the cut) must equal the Cosmos archive's `states`, `actions` and
`actions_is_pad`. Record the result.

## 4. Conditioning: the goal sentences

The sentence of the goal board is the only goal signal. No goal image, no
task id, no start board (the image shows it). One fixed template over all 81
boards, pegs always in the order A, B, C, rings numbered 1 (smallest) to 4;
an empty peg reads "is empty", one ring "holds ring N", several "holds rings
a, b and c" in ascending order:

```
Goal: peg A holds rings 1, 2, 3 and 4, peg B is empty, peg C is empty.
Goal: peg A holds rings 2, 3 and 4, peg B holds ring 1, peg C is empty.
Goal: peg A holds rings 2 and 4, peg B holds ring 3, peg C holds ring 1.
```

Generate the strings with `prompt_for_board` in
`cosmos_policy/datasets/hanoi_play_data.py` (or reproduce the rule exactly)
and compare all 81 against the `prompts` list in
`data/hanoi_cosmos/play_k5/metadata.json` (its `prompts_sha256` is the SHA-256
of the 81 strings joined by newlines). pi0.5 tokenizes them with its own
tokenizer; keeping the strings identical keeps the two policies comparable and
the hardware client the same.

The six tower tasks are the three full-stack sentences; two tasks share a
sentence because the start is in the image:

| Task | Sentence |
|---|---|
| AAAA to CCCC, BBBB to CCCC | Goal: peg A is empty, peg B is empty, peg C holds rings 1, 2, 3 and 4. |
| CCCC to AAAA, BBBB to AAAA | Goal: peg A holds rings 1, 2, 3 and 4, peg B is empty, peg C is empty. |
| AAAA to BBBB, CCCC to BBBB | Goal: peg A is empty, peg B holds rings 1, 2, 3 and 4, peg C is empty. |

Serving must require the sentence on every request, refuse anything else
(including the old task prompts), and echo the resolved goal board.

## 5. Training

Your six-task pi0.5 recipe unchanged: `action_horizon=16`, continuous state,
no augmentation, the pi0.5 base checkpoint, 32,000 updates at batch 32
(about 1.3 passes over the training rows), export and evaluate every 2,000,
selection by decision 11 on validation (lowest mean per-step XYZ error over
valid slots, jaw accuracy at least 0.99, ties to the earlier step). Do not
consult the test split.

## 6. Evaluation

Section 7 metrics for all rows, split by motion, and additionally:

- by `goal_moves_ahead` (0 to 4) and by `goal_graph_distance` (0 to 5);
- decision rows alone: `motion_stages` 2 (approach_source, where the pick is
  chosen) and 6 (transit, where the place is chosen), 32% of rows; the other
  rows are the same whatever the goal;
- serving parity on the selected export (single observation, its sentence,
  0.5 mm).

**Goal-following probe** (required): Cosmos's implementation is
`cosmos_policy/experiments/robot/hanoi/run_hanoi_play_eval.py` (`--probe-rows`).

1. *Matched-goal decision test.* Among decision rows, pairs with the same
   current board (`board_indices`), measured XYZ within 5 mm, different goal
   boards, and labels differing by more than 5 mm over the shared valid slots.
   Predict the first row under both sentences; correct = closer to the label
   of the row whose sentence was used. Report accuracy over both directions;
   chance is 50%.
2. *Shuffled-goal sensitivity.* Decision rows predicted under their own
   sentence and under a random other board's sentence; report the mean chunk
   displacement against the displacement between two draws under the own
   sentence.

The probe is the deployment gate, then first-pose error. The composition test
itself is on the arm: a fixed shared list of (start, goal) pairs at graph
distance 1, 3, 7 and 15, the same resets and sentences for every policy and
for the world-model planner, a step budget of twice the optimal move count,
success and moves before the first error per pair. The execution control is
the same policy given the planner's next-board sentence.

## 7. Deliverables

A results note like your six-task one: the cross-check result, the per-stage
validation table, the selected export with its hashes, validation and test
tables (all, by motion, by goal horizon, decision rows), the probe numbers,
serving parity, and a transfer list. The Cosmos run's results will be
appended to `docs/hanoi_play_composition_related_work.md` for the side by side.
