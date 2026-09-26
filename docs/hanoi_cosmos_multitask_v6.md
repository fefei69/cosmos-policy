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

Two runs of the same recipe, differing only in initial weights; the probe
decides which to deploy. Each needs up to three 12-hour allocations; the
continuations are chained with `--dependency=afterany`. A first submission
(18606824, 18606827) failed at start on a metadata key in the launcher's
identity (fixed in 1f43ed6) before writing anything; these are the reruns.

| Run | Initial weights | Jobs |
|---|---|---|
| `hanoi_cosmos_multitask_20260926_video_init` | Cosmos-Predict2 2B video base (`fbc4f05d...`) | 18606947, 18606948, 18606949 |
| `hanoi_cosmos_multitask_20260926_libero_init` | LIBERO policy checkpoint (has learned action selection from language on LIBERO-Goal) | 18606952, 18606953, 18606954 |

Results are appended here as they arrive. Nothing here is a hardware result.
The hardware client must send the task prompt with every request.
