# Handover: pi0.5 on the one-move expert set (six tasks, H1)

Written October 6, 2026, by the Cosmos-side agent for the OpenPI agent. A
short delta to `docs/hanoi_play_k5_pi05_handover.md`: the interface (goal
sentence, dense chunk cut at the goal, probe) is the play one; only the data
and the budget change. Do not write under the cosmos-policy tree; read from it
freely.

## 1. What this run is

The first move of each of the six tower tasks, ring 1 off a full stack, from
the expert recordings. Six cases, each demonstrated eight times in training:

| Recording (task) | Case | Goal sentence |
|---|---|---|
| AAAA to CCCC | AAAA to BAAA | Goal: peg A holds rings 2, 3 and 4, peg B holds ring 1, peg C is empty. |
| AAAA to BBBB | AAAA to CAAA | Goal: peg A holds rings 2, 3 and 4, peg B is empty, peg C holds ring 1. |
| BBBB to CCCC | BBBB to ABBB | Goal: peg A holds ring 1, peg B holds rings 2, 3 and 4, peg C is empty. |
| BBBB to AAAA | BBBB to CBBB | Goal: peg A is empty, peg B holds rings 2, 3 and 4, peg C holds ring 1. |
| CCCC to BBBB | CCCC to ACCC | Goal: peg A holds ring 1, peg B is empty, peg C holds rings 2, 3 and 4. |
| CCCC to AAAA | CCCC to BCCC | Goal: peg A is empty, peg B holds ring 1, peg C holds rings 2, 3 and 4. |

It is not a composition test and must not be reported as one: every test case
is in training. It measures whether the goal-sentence interface works on
expert data, the execution control for the play run. Two cases share each
start board, so the same image must lead to peg B or peg C depending on the
sentence alone, the same decision the six-task probe scored at 97%.

## 2. Data

The six expert recordings you already use for the six-task run
(`hanoi_wm_roundtrip_20260925_171442_{AAAA_to_CCCC,CCCC_to_AAAA}.h5`,
`hanoi_wm_roundtrip_20260926_011840_{AAAA_to_BBBB,BBBB_to_AAAA,BBBB_to_CCCC,CCCC_to_BBBB}.h5`
in `/scratch/cw5167/datasets/`), with that run's split per file: episodes 0 to
7 train, 8 validation, 9 test.

- **Episode**: move 0 of an expert episode, motion stages 1 to 9 (open to
  retreat), one contiguous block of rows.
- **Rows**: the six-task observation rule on that block (not `image_stale`,
  not `image_repeated`).
- **Goal**: the board after the move, read at the move's last usable row (a
  retreat row). One move, nothing sampled.
- **Labels**: slot j = 1..16 is raw row t + 3 j, XYZ from
  `reference_pose[row, 0:3]`, jaw from `action_abs[row, 3]`; slots past the
  move's last row repeat that row and are padded. State and image as in the
  play handover.

| Split | One-move episodes | Rows |
|---|---|---|
| train | 48 (8 per case) | 24,636 |
| validation | 6 (1 per case) | 3,117 |
| test | 6 (1 per case) | 3,074 |

Cross-check target: `data/hanoi_cosmos/expert6_h1/{train,val,test}.npz`
(contract `hanoi_expert6_h1_cosmos_v1`, same columns as the play archive plus
`task_indices`; builder `cosmos_policy/datasets/hanoi_first_move_data.py`).
The V-JEPA side's archive
`/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/data/expert6_first_move/` is
row-identical to it on rows, states, labels, pads, target rows, goal rows and
goal boards (verified on all three splits and recorded in the Cosmos
metadata), and it already uses your play column names, so you may load that
one directly. Episode ids in both: task index times 10,000 plus the expert
episode times 100.

## 3. Sentences

The 81-sentence template of the play handover, unchanged (`prompts_sha256`
`ac402b1d...`); only the six sentences in section 1 occur. Serving requires
the sentence verbatim and refuses the six-task prompts.

## 4. Training

Your six-task recipe with a smaller budget. The Cosmos run uses 8,000 updates
at batch 32 with an export and validation every 1,000, selection by decision
11 on the six validation episodes. The reason for not scaling the budget
down with the data: from their base checkpoints both Cosmos runs needed about
8,000 updates to get under 3 mm whatever the data size, so use the number of
updates your six-task pi0.5 run needed to converge, capped where validation
stops improving, and say what you chose.

## 5. Evaluation

Section 7 metrics on validation and test, by motion, by stage, and for the
decision rows (motion stages 2 and 6). The goal probe of the play handover,
with pairs taken from the two cases that share a start board: same board and
stage, measured XYZ within 5 mm, different goals, labels differing by more
than 5 mm. Validation yields 299 such pairs, 281 of them in transit, where
the sentence decides the target peg; report the transit pairs separately.
Serving parity at 0.5 mm.

On the arm: the six (start, goal) pairs, three trials each, start from the
full tower at the grasp hover with the jaw open, success when the tracked
board equals the goal within two moves.
