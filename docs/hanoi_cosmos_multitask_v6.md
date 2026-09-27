# Cosmos Hanoi six-task retraining (contract `hanoi_multitask_v6`), Cosmos side

Prepared September 26, 2026, at the user's request, from the six-direction
recordings of September 25 and 26. This note covers only the Cosmos pipeline;
it extends `docs/hanoi_cosmos_dense_v5.md`, whose recipe it reuses, and
changes two decisions of `docs/hanoi_dense_training_guide.md` (the guide file
itself is untouched): decision 6 (one direction) becomes six directions
conditioned by prompt, and decision 8 (16,000 updates) becomes 32,000, since
cycle 2 showed the second 16,000 updates were worth more than any design
change. Everything else (10 Hz labels, chunk 16, absolute XYZ plus jaw,
execution prefix 8, batch 32, schedule shape, video-base initial weights,
selection rule) is the dense v5 recipe.

## The recordings (`/scratch/cw5167/datasets/hanoi_wm_roundtrip_2026092{5,6}_*`)

Six schema-v4 files, one per directed tower move, ten successful episodes
each, 30 Hz rows, the same 224 x 224 crop and gzip frame chunking as the
September 15 recording, and the same camera (first frames differ from
September 15 by 3.1/255 per pixel, the same figure the README gives between
its own two sessions). New motion profile `velocity-noise-v1`: horizontal
legs timed by distance at 0.034 m/s, 75% of eligible legs with a 0.5 to 2 mm
lateral excursion and +-10% speed variation, so episodes are longer and
slower than before. The same section-3 audit as for dense v5, per file:

| Task | Rows per episode | `action_abs` = reference | Jaw flips per episode, on command rows | Stationary rows | Stale or repeated images | First move (ring 1 to) |
|---|---|---|---|---|---|---|
| AAAA to CCCC | 7,803 | exact | 30, all | 18.3% | 3.41% | B |
| CCCC to AAAA | 7,803 | exact | 30, all | 18.1% | 3.40% | B |
| AAAA to BBBB | 7,866 | exact | 30, all | 18.4% | 3.01% | C |
| BBBB to AAAA | 7,845 | exact | 30, all | 18.2% | 3.04% | C |
| BBBB to CCCC | 7,845 | exact | 30, all | 18.8% | 3.12% | A |
| CCCC to BBBB | 7,866 | exact | 30, all | 17.6% | 3.07% | A |

Workspace bounds are identical across the six files and equal the dense v5
bounds except x max (+2.5 mm, the lateral excursions). Commanded release
positions: peg A y = -0.057 m, peg B y = 0.015 m, peg C y = 0.085 m. SHA-256
prefixes: 74fa5a6f, 07a264f4, 6a15bd6e, d9817aac, de540041, b670a44e (the
full hashes are in the dataset metadata). The September 10 and 15 recordings
are not part of this set (fixed leg timing; the README treats them separately).

## Prompts (the only task signal)

Cosmos Policy conditions on a frozen T5-11B embedding of the exact
instruction string through cross-attention, with text dropout 0 and no
guidance; there is no task index, goal image or dataset tag. The wording puts
the peg letters early and states the goal twice, once with its position, so a
task occupies several token positions:

```
Move all four rings from peg A to peg C following Tower of Hanoi rules. The goal is peg C, the right peg.
Move all four rings from peg C to peg A following Tower of Hanoi rules. The goal is peg A, the left peg.
Move all four rings from peg A to peg B following Tower of Hanoi rules. The goal is peg B, the middle peg.
Move all four rings from peg B to peg A following Tower of Hanoi rules. The goal is peg A, the left peg.
Move all four rings from peg B to peg C following Tower of Hanoi rules. The goal is peg C, the right peg.
Move all four rings from peg C to peg B following Tower of Hanoi rules. The goal is peg B, the middle peg.
```

Encoded with `examples/hanoi/prepare_t5.py encode --prompt-set multitask`
(T5-11B at the pinned revision, float32, CPU job 18606212) into
`data/hanoi_cosmos/t5_embeddings_multitask.pkl`; the two-prompt cache of the
deployed dense model is untouched. Every prompt is 32 tokens. Mean-pooled
cosine between prompts is 0.97 to 0.99 and says nothing (T5 states are
anisotropic); per token, the three same-start pairs (A to C versus A to B, C
to A versus C to B, B to A versus B to C) differ at 7 of 32 positions with a
minimum token cosine of 0.04 to 0.67, and the reverse pairs at 6 to 9
positions. The model reads per-token keys, so that is the relevant measure;
the behavioural check is the probe below.

Why this should work, from the literature search of September 26: the same
architecture and training setup reaches 98% on LIBERO-Goal, ten tasks in one
scene whose instructions differ by a few words, and vision-only policies
collapse there because the scene does not determine the task. Hanoi is that
case: every board on a demonstration belongs to two tasks (same start, or
forward and reverse), so the model cannot fit the data while ignoring the
prompt. The known failure mode of language-conditioned policies is ignoring
language when vision alone determines the task; the open risk here is ten
demonstrations per task against fifty. Serving requires the prompt on every
request and refuses any other string, including the old single-task prompt.

## Dataset (`data/hanoi_cosmos/multitask_v6`)

Built by `cosmos_policy/datasets/hanoi_multitask_data.py` from the six raw
files with the dense v5 label rule, plus a task index per row and global
episode ids (task index times 100 plus the episode within its file). Split
per file: episodes 0 to 7 train, 8 validation, 9 test.

| Split | Rows | Episodes | Per task | Padded slots | Stationary |
|---|---|---|---|---|---|
| train | 363,995 | 48 | 60,232 to 61,014 | 0.33% | 18.3% |
| validation | 45,808 | 6 | 7,552 to 7,695 | 0.33% | 18.2% |
| test | 45,552 | 6 | 7,521 to 7,664 | 0.33% | 17.9% |

Normalisation is min/max over the combined training split (actions over
valid slots; bounds equal the dense v5 ones except x max). The validation and
test numbers of a task rest on one episode each.

## Pipeline

| Piece | File |
|---|---|
| Task table, prompts, builder | `cosmos_policy/datasets/hanoi_multitask_data.py` |
| Dataset (one handle per recording, per-sample prompt embedding) | `cosmos_policy/datasets/hanoi_multitask_dataset.py` |
| Config (32,000 updates, export every 2,000, monitoring every 500) | `cosmos_policy/config/hanoi_multitask_config.py` |
| Evaluator (dense metrics for all rows, by motion and by task; parity; language probe) | `cosmos_policy/experiments/robot/hanoi/run_hanoi_multitask_eval.py` |
| Serving adapter and server (prompt required, task echoed) | `multitask_policy.py`, `serve_hanoi_multitask.py` |
| Launcher and batch script | `examples/hanoi/run_multitask.py`, `examples/hanoi/train_multitask.sbatch` |
| Tests | `tests/test_hanoi_multitask.py` (5), on top of the dense tests |

Stage evaluation: every export (2,000 updates) on every ninth validation row
across the six tasks (about 5,100 rows) at 5 denoising steps. Selection:
decision 11 over all tasks. Final passes on the selected export: every third
row at 5 and 10 steps, future-frame decode, 200-observation serving parity,
and the language probe with 64 rows per test; then the locked test split.

### Language-following probe

Reported by the evaluator with `--probe-rows`, on the selected export only:

- **Same-start decision test.** For each pair of tasks with the same start
  stack, observations of one task during its first move (lift, transit and
  insertion of ring 1, where image and state are identical across the pair)
  are predicted under both prompts. Each prediction is scored against the
  label of the task whose prompt was used, at that task's nearest recorded
  state (matched within 5 mm); only rows where the two tasks' labels differ by
  more than 5 mm count. A prediction is correct if it is closer to the label
  of its own prompt than to the other task's. Chance is 50%. Validation offers
  40 to 102 such rows per direction of each pair (label gaps 17 to 38 mm on
  average).
- **Reverse-prompt sensitivity.** Moving rows predicted under their own
  prompt and under the reverse task's prompt; the displacement between the
  two is reported against the displacement between two noise draws under the
  same prompt. Sensitivity only, since a reversed prompt has no unique correct
  chunk from an arbitrary mid-episode pose.

## Runs (submitted September 26, 18:30 EDT)

One run from the video base; a LIBERO-init hedge was started and then
cancelled by the user, since the video-base init had already been shown to
work on this task. Each needs up to three 12-hour allocations; the
continuations are chained with `--dependency=afterany`. A first submission
(18606824, 18606827) failed at start on a metadata key in the launcher's
identity (fixed in 1f43ed6), and a second (18606947, 18606952) at the first
training step because the environment script's two-prompt embedding cache was
picked up (fixed in 81a7329); neither wrote a checkpoint.

| Run | Initial weights | Jobs |
|---|---|---|
| `hanoi_cosmos_multitask_20260926_video_init` | Cosmos-Predict2 2B video base (`fbc4f05d...`) | 18607185, 18607186, 18607188 |
| `hanoi_cosmos_multitask_20260926_libero_init` | LIBERO policy checkpoint | 18607189 (cancelled by the user at 18:55 after qualification: the video-base init had already been shown to work; run directory removed) |

## Results (video-base init, completed September 27, 19:30 EDT)

Training: 32,000 updates at 2.15 s each across three allocations (18607185
stopped at its budget in stage 12, 18607186 in stage 16, 18607188 finished
the last updates and the final passes; both handovers resumed cleanly).
About 25 hours of wall time including the 16 stage evaluations.

Per-export validation (every ninth row, about 5,100 rows over the six
held-out episodes, 5 denoising steps; the last column is the spread of the
six per-task slot-1 means):

| Updates | Slot-1 mm (all / stationary / moving) | Mean over slots | Endpoint | Jaw acc. | Per-task slot-1 |
|---|---|---|---|---|---|
| 2,000 | 14.14 / 14.53 / 14.06 | 22.72 | 35.96 | 0.9722 | 14.02 to 14.52 |
| 4,000 | 4.77 / 3.83 / 4.99 | 10.73 | 16.54 | 0.9927 | 4.50 to 5.03 |
| 6,000 | 2.87 / 2.70 / 2.91 | 5.16 | 7.25 | 0.9961 | 2.81 to 3.01 |
| 8,000 | 2.25 / 1.92 / 2.32 | 4.47 | 6.46 | 0.9961 | 2.19 to 2.35 |
| 10,000 | 2.07 / 1.64 / 2.18 | 4.60 | 6.22 | 0.9968 | 2.01 to 2.12 |
| 12,000 | 1.42 / 1.15 / 1.48 | 3.58 | 5.16 | 0.9978 | 1.35 to 1.47 |
| 14,000 | 1.76 / 1.36 / 1.85 | 3.74 | 4.99 | 0.9974 | 1.71 to 1.80 |
| 16,000 | 1.97 / 2.05 / 1.95 | 3.98 | 5.64 | 0.9981 | 1.90 to 2.08 |
| 18,000 | 1.65 / 1.58 / 1.67 | 3.45 | 4.66 | 0.9985 | 1.62 to 1.67 |
| 20,000 | 1.21 / 1.10 / 1.23 | 3.17 | 4.45 | 0.9986 | 1.16 to 1.24 |
| 22,000 | 1.10 / 1.02 / 1.12 | 2.83 | 4.11 | 0.9984 | 1.08 to 1.11 |
| 24,000 | 1.01 / 1.02 / 1.01 | 2.73 | 3.92 | 0.9990 | 0.97 to 1.05 |
| 26,000 | 1.00 / 0.93 / 1.01 | 2.61 | 3.86 | 0.9991 | 0.98 to 1.01 |
| 28,000 | 0.82 / 0.80 / 0.82 | 2.48 | 3.74 | 0.9991 | 0.79 to 0.83 |
| 30,000 | 0.87 / 0.80 / 0.88 | 2.55 | 3.87 | 0.9991 | 0.81 to 0.91 |
| 32,000 | 0.81 / 0.77 / 0.82 | 2.27 | 3.46 | 0.9993 | 0.78 to 0.83 |

The six tasks never separate by more than 0.5 mm at any stage. Against the
single-task run at equal update counts, the six-task model is behind by about
0.6 mm at 8,000 and ahead from 22,000 on (1.10 against 1.33 mm at 22,000).

Selection (decision 11 over all tasks): step 32,000, `exports/iter_000032000.pt`,
SHA-256 `6521a05937ee906384ec7afe677212ab429516cc10f4944c0d8c72921624cca0`.

Full validation (every third row, 15,264 scored, 5 all-padded rows skipped):

| Subset | Rows | Slot-1 mean / median / p95 | Within 2 mm | Mean over slots | Endpoint | Jaw acc. | Flips predicted |
|---|---|---|---|---|---|---|---|
| All | 15,264 | 0.79 / 0.62 / 1.87 mm | 95.9% | 2.31 mm | 3.59 mm | 0.9993 | 2,786 of 2,792 |
| Stationary | 2,775 | 0.72 / 0.53 / 1.86 mm | 95.9% | 4.79 mm | 8.91 mm | 0.9998 | 41 of 41 |
| Moving | 12,489 | 0.80 / 0.65 / 1.88 mm | 95.9% | 1.76 mm | 2.40 mm | 0.9992 | 2,745 of 2,751 |

Per task slot-1: 0.79, 0.79, 0.80, 0.79, 0.79, 0.77 mm (task order as in the
table above). Ten denoising steps change nothing (0.79 / 2.32). Future frame
PSNR 32.1 dB, value error 0.003. Serving parity on 200 observations with
their own prompts: max first-slot difference 0.064 mm, mean 0.029 mm, passed.

Test split (locked after selection; every third row, 15,179 scored, 6
all-padded rows skipped):

| Subset | Rows | Slot-1 mean / median / p95 | Within 2 mm | Mean over slots | Endpoint | Jaw acc. | Balanced | Flips predicted | Flip timing |
|---|---|---|---|---|---|---|---|---|---|
| All | 15,179 | 0.80 / 0.62 / 1.93 mm | 95.6% | 2.30 mm | 3.56 mm | 0.9993 | 0.9993 | 2,792 of 2,795 | median 0, p95 3 rows |
| Stationary | 2,678 | 0.73 / 0.53 / 1.91 mm | 95.6% | 4.63 mm | 8.77 mm | 0.9998 | 0.9998 | 34 of 34 | median 0, p95 3 rows |
| Moving | 12,501 | 0.81 / 0.64 / 1.93 mm | 95.6% | 1.80 mm | 2.44 mm | 0.9992 | 0.9992 | 2,758 of 2,761 | median 0, p95 3 rows |

Per task on test (slot-1 / mean over slots / jaw): A to C 0.80 / 2.34 /
0.9994; C to A 0.79 / 2.28 / 0.9994; A to B 0.79 / 2.27 / 0.9990; B to A
0.80 / 2.24 / 0.9993; B to C 0.80 / 2.39 / 0.9993; C to B 0.80 / 2.28 /
0.9993. No validation-to-test gap. Each task's figures rest on one episode.

### Language-following probe

| Split | Decision accuracy (chance 50%) | Rows | Per direction of each pair | Swap displacement | Reverse-prompt shift vs same-prompt redraw |
|---|---|---|---|---|---|
| validation | 97.8% | 339 | 0.93 to 1.00 | 16 to 38 mm | 38.3 mm vs 1.6 mm |
| test | 96.7% | 333 | 0.88 to 1.00 | 17 to 35 mm | 29.0 mm vs 0.8 mm |

The lowest entries are the B-start pair scored under the swapped prompt
(0.88 and 0.91 on test, 32 and 45 rows), where the matched window is the
short lift before the transit diverges. Swapping the prompt moves the
predicted chunk by 16 to 38 mm on the same observation, an order of magnitude
above the noise between two draws; the goal is taken from the prompt.

### Against the single-task model (test split)

| Model | Tasks | Updates | Slot-1 | Within 2 mm | Mean over slots | Jaw acc. |
|---|---|---|---|---|---|---|
| Run B, video init | 1 | 16,000 | 1.15 mm | 88.8% | 3.25 mm | 0.9994 |
| Cycle 2 of run B (deployed before) | 1 | 32,000 | 0.81 mm | 95.6% | 2.84 mm | 0.9996 |
| Six-task, video init | 6 | 32,000 | 0.80 mm | 95.6% | 2.30 mm | 0.9993 |

Different recordings (the new set has slower motion with 0.5 to 2 mm lateral
variation), so the comparison is indicative, not controlled. Ready-to-deploy
criteria (guide section 7) all met offline. Nothing here is a hardware
result. The hardware client must send the task prompt with every request.

### Cycle 2 (in progress)

At the user's request (September 27, 01:45), a second cycle
(`hanoi_cosmos_multitask_20260926_video_init_cycle2`, jobs 18626782,
18626783, 18626784) continues from the step-32,000 export with a fresh
optimizer and the same schedule for 32,000 more updates, exporting every
2,000; the best export across both cycles is chosen afterwards on the same
validation rows. Its `run_notes.json` records the starting point. The
first-cycle export above is deployable as it stands.
