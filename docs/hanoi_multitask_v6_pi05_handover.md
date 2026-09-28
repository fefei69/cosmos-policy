# Handover: train pi0.5 on the six-task Hanoi set with the Cosmos multitask recipe

Written September 26, 2026, by the Cosmos-side agent for the OpenPI agent.
Goal: a pi0.5 model trained on exactly the data, split, labels and prompts
that the Cosmos six-task run (`hanoi_cosmos_multitask_20260926_video_init`)
uses, so the two policies are comparable offline and on the arm. Everything
Cosmos-specific is left out; where pi0.5 has an established setting from the
dense v5 runs, keep it. Do not write under the cosmos-policy tree; read from
it freely. The Cosmos reference implementation is
`cosmos_policy/datasets/hanoi_multitask_data.py` and the note
`docs/hanoi_cosmos_multitask_v6.md`.

## 1. Data

Six schema-v4 recordings in `/scratch/cw5167/datasets/`, one per directed
tower move, ten successful episodes each (audited, all pass; details in the
Cosmos note and the recordings' `hanoi_wm_roundtrip_60ep_README.md`):

| Task index | Direction | File | SHA-256 (prefix) |
|---|---|---|---|
| 0 | AAAA to CCCC | `hanoi_wm_roundtrip_20260925_171442_AAAA_to_CCCC.h5` | 74fa5a6f |
| 1 | CCCC to AAAA | `hanoi_wm_roundtrip_20260925_171442_CCCC_to_AAAA.h5` | 07a264f4 |
| 2 | AAAA to BBBB | `hanoi_wm_roundtrip_20260926_011840_AAAA_to_BBBB.h5` | 6a15bd6e |
| 3 | BBBB to AAAA | `hanoi_wm_roundtrip_20260926_011840_BBBB_to_AAAA.h5` | d9817aac |
| 4 | BBBB to CCCC | `hanoi_wm_roundtrip_20260926_011840_BBBB_to_CCCC.h5` | de540041 |
| 5 | CCCC to BBBB | `hanoi_wm_roundtrip_20260926_011840_CCCC_to_BBBB.h5` | b670a44e |

The September 10 and 15 recordings are not part of this set.

**Split, per file:** episodes 0 to 7 train, episode 8 validation, episode 9
test. Episode ids are zero-based within each file. Cosmos uses a global id of
task index times 100 plus the episode (0 to 7, 108, 209, ...); use any scheme
that keeps episodes of different files apart.

**Observation rule** (dense v5, unchanged): every row whose image is neither
stale nor repeated (`image_stale`, `image_repeated`) is an observation. No
rows are curated, dropped, re-weighted or relabelled.

**Label rule** (dense v5, unchanged): slot j = 1..H of the chunk is raw row
t + 3 j, XYZ from `reference_pose[row, 0:3]` and jaw intent from
`action_abs[row, 3]`, absolute base-frame metres, 10 Hz (frameskip 3 over the
30 Hz rows). Rows past the episode end repeat the last row and are padded.

**Horizon:** 16 slots (1.6 s), the Cosmos horizon; your h16 comparison run
showed it equal to h30 on the first pose and better on the gripper. Keep your
execution prefix of 3 steps (0.3 s) and asynchronous re-planning.

**State:** six measured joint angles plus the measured jaw stroke
(`joint_positions`, `proprio[:, 6]`), continuous, no velocity, no XYZ, as in
dense v5. Image: the stored 224 x 224 RGB, no augmentation.

**Expected counts** (the Cosmos build, `data/hanoi_cosmos/multitask_v6/metadata.json`):

| Split | Rows | Per task |
|---|---|---|
| train | 363,995 | 60,232 / 60,302 / 60,952 / 60,719 / 60,776 / 61,014 |
| validation | 45,808 | 7,577 / 7,552 / 7,695 / 7,679 / 7,644 / 7,661 |
| test | 45,552 | 7,564 / 7,521 / 7,649 / 7,664 / 7,582 / 7,572 |

Cross-check your archive against the Cosmos one before training: the Cosmos
`{train,val,test}.npz` files carry `source_observation_indices` (row within
the file), `task_indices`, `states`, `actions` (N, 16, 4), `actions_is_pad`,
`source_action_indices` and `episode_indices`; per task, your rows, states and
first 16 slots must be identical, as they were for dense v5. Record the result;
do not copy the Cosmos archive.

## 2. Task conditioning: the prompts

The instruction is the only task signal. Use these six strings verbatim, one
per task (pi0.5 tokenizes them with its own tokenizer; keeping the strings
identical keeps the two policies comparable and the hardware client the same):

```
Move all four rings from peg A to peg C following Tower of Hanoi rules. The goal is peg C, the right peg.
Move all four rings from peg C to peg A following Tower of Hanoi rules. The goal is peg A, the left peg.
Move all four rings from peg A to peg B following Tower of Hanoi rules. The goal is peg B, the middle peg.
Move all four rings from peg B to peg A following Tower of Hanoi rules. The goal is peg A, the left peg.
Move all four rings from peg B to peg C following Tower of Hanoi rules. The goal is peg C, the right peg.
Move all four rings from peg C to peg B following Tower of Hanoi rules. The goal is peg B, the middle peg.
```

Why this wording: the goal peg is not visible in the image and the same board
needs different moves for different goals, so the prompt must carry the goal
unmistakably; the peg letters come early and the goal is stated twice with its
position so it occupies several token positions. No task index, one-hot or
goal image. Serving must require the prompt on every request and refuse
anything else, including the old single-task prompt; there is no default
task. Echo the resolved task in the reply.

## 3. Training

Your dense v5 pi0.5 recipe, with these settings matched to Cosmos where they
differ:

- `action_horizon=16`, continuous state input, no augmentation, the pi0.5 base
  checkpoint, as in your h16 run.
- Budget 32,000 updates at batch 32 (about 2.8 passes over the training rows);
  export and evaluate every 2,000 (or 4,000 if your stage evaluation is slow).
- Normalisation statistics over the combined training split.
- Selection: lowest mean per-step XYZ error over valid slots on the validation
  split across all tasks, jaw accuracy at least 0.99, ties to the earlier step
  (decision 11). Do not consult the test split.

## 4. Evaluation

Section 7 of the guide, reported for all rows, split by motion (stationary =
finite-difference measured speed under 2 mm/s) and **split by task**: slot-1
XYZ error (mean, median, p95, fraction within 2 mm), mean over valid slots,
endpoint, jaw accuracy and flip timing; serving parity on the selected export
(single observation through the served path, its own prompt, 0.5 mm). Each
task's validation and test numbers rest on one episode; say so.

**Language-following probe** (required; an instruction-ignoring model scores
well on everything above). Cosmos's implementation is
`cosmos_policy/experiments/robot/hanoi/run_hanoi_multitask_eval.py`
(`--probe-rows`); reproduce the same definitions:

1. *Same-start decision test.* For each pair of tasks with the same start
   stack, (0, 2), (1, 5), (3, 4): take observations of one task during its
   first move while ring 1 is held (motion stages lift, transit, insertion;
   `move_idx == 0`, `held_disk == 1`, `motion_stage` in {5, 6, 7}), where
   image and state are identical across the pair. Predict under both prompts.
   Score each prediction against the label of the task whose prompt was used,
   at that task's nearest recorded state in its own first move (matched within
   5 mm on measured XYZ); count only rows where the two tasks' labels differ
   by more than 5 mm over the chunk. Correct = closer to the label of its own
   prompt than to the other task's. Report per direction of each pair and the
   average; chance is 50%. Validation offers 40 to 102 such rows per direction.
2. *Reverse-prompt sensitivity.* Moving validation rows predicted under their
   own prompt and under the reverse task's prompt; report the mean chunk
   displacement between the two against the displacement between two noise
   draws under the same prompt. Sensitivity only, not correctness.

The probe is the deployment gate: decision accuracy near 100% and swap
displacement well above the redraw noise, then first-pose error.

Useful geometry: commanded release y per peg is A -0.057 m, B 0.015 m,
C 0.085 m; first moves go A→C: ring 1 to B, C→A: to B, A→B: to C, B→A: to C,
B→C: to A, C→B: to A.

## 5. Deliverables

A results note like your `hanoi_dense_pi05_h16_results_20260920.md`: the
cross-check result, per-stage validation table, the selected export with its
hashes, validation and test tables (all, by motion, by task), the probe
numbers, serving parity, and a transfer list for the deployment host. The
Cosmos run's results will be appended to `docs/hanoi_cosmos_multitask_v6.md`
in the cosmos-policy tree for the side-by-side.
