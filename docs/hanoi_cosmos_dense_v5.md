# Cosmos Hanoi dense retraining (contract `hanoi_dense_v5`), Cosmos side

Prepared September 19, 2026, from `docs/hanoi_dense_training_guide.md`. This
note covers only the Cosmos pipeline; the pi0.5 pipeline is the OpenPI
agent's. Run B (video init) was submitted as job 18019908 with continuation
18019909 on September 19, 11:22 EDT; run A (LIBERO init) is not queued while
the per-user GPU cap is filled by the OpenPI dense run and run B.

## Section 3 audit of the raw recording (50 episodes, 360,050 rows)

| Check | Result |
|---|---|
| (a) `action_abs[t, :3]` vs `reference_pose[t+1, :3]` | `action_abs[t]` equals `reference_pose[t]` exactly (max 0.0 mm; attribute `action_abs_alignment = post_action_reference`). Against row t+1: mean 1.09 mm, median 0.73, p95 3.64, max 4.33, 42.5% over 1 mm, all in the six moving motion stages, none at gripper rows. That residual is one 30 Hz tick of commanded motion, not a misalignment. |
| (b) jaw intent flips | 1,500 flips, 1,500 gripper commands, every flip on its command row, 30 per episode |
| (c) stationary rows | 17.1% overall and 17.3% in episode 40 by finite-difference speed of the measured XYZ at 30 Hz under 2 mm/s. The SDK velocity field gives 3.0%, so the finite-difference definition is the one used for the motion split. |
| (d) stale / repeated images | 7,852 stale, 14,770 repeated, every stale row also repeated; 14,770 excluded. Per episode 67 to 291 stale, 178 to 450 repeated, 6,751 to 7,023 observations. |

Totals after exclusion: 275,847 train, 34,619 validation, 34,814 test rows.
Chunk 16 at frameskip 3 pads 0.35% of slots. Budget 16,000 updates at batch 32
is 1.9 passes over the training rows. The full audit, including per-episode
counts, is in `data/hanoi_cosmos/dense_v5/metadata.json`.

## Dataset (`data/hanoi_cosmos/dense_v5`)

Built by `cosmos_policy/datasets/hanoi_dense_data.py` directly from the raw
recording using the guide's rules (section 4), so it is deterministic and the
same dataset whichever pipeline builds it. Nothing is written under the openpi
tree. When the OpenPI archive appears, `--cross-check-only` compares rows,
states, the first 16 chunk slots, pads and source indices and writes
`openpi_cross_check.json` beside the metadata (metadata itself is part of the
run identity and is never rewritten).

| Item | Value |
|---|---|
| Contract | `hanoi_dense_v5_cosmos_v1`; deployment contract dict as in guide section 4 |
| Observation | every non-stale, non-repeated row; state = six joints + jaw stroke |
| Label | slot j = row t + 3 j: XYZ from `reference_pose`, jaw from `action_abs`; absolute metres; padded past the episode end |
| Normalisation | Cosmos min/max over the training split (actions over valid slots), mean/std recorded, accumulated in float64 |
| Evaluation-only fields | measured XYZ, finite-difference speed and the stationary flag per observation |
| Archive sizes | train 132 MB, val 17 MB, test 17 MB (images indexed in the raw file) |

## Cosmos pipeline

| Piece | File |
|---|---|
| Platform `hanoi_dense` (chunk 16, action 4, state 7) | `cosmos_policy/constants.py` |
| Dataset | `cosmos_policy/datasets/hanoi_dense_dataset.py` (future row t + 48, value as v4) |
| Config | `cosmos_policy/config/hanoi_dense_config.py`: micro-batch 16 x 2, no block recompute, 16,000 updates, save/export every 1,000, monitoring every 500, v4 schedule shape (warm-up 800, decay to 0.3, hold 0.06) |
| Initial weights | `cosmos_policy/models/hanoi_dense_model.py`: `HANOI_INIT_FORMAT=video_base` (run B) accepts nested `model`, keys with or without `net.`, drops EMA, loads strictly; `policy` (run A) is the v4 strict loader. The video-base path is untested until the checkpoint exists. |
| Launcher | `examples/hanoi/run_dense.py --init video|libero`; runs `hanoi_cosmos_dense_20260919_video_init` and `_libero_init`; identity records the initial weights' SHA-256 |
| Batch script | `examples/hanoi/train_dense.sbatch`: account, one GPU, `--constraint=h200`, 12 h; continuation with `--dependency=afterany:<jobid>` |
| Evaluator | `cosmos_policy/experiments/robot/hanoi/run_hanoi_dense_eval.py`: batched (16 rows per sampler call), section 7 metrics split by stationary/moving, value error, optional decoded future-frame L1/PSNR, serving parity |
| Serving | `dense_policy.py` (`HanoiDensePolicy.infer` returns `actions (16, 4)`, `reference_rate_hz 10`, `execution_prefix 8`, identity under `cosmos_hanoi`) and `serve_hanoi_dense.py` after the official ALOHA `deploy.py` |

Evaluation cadence: every export is evaluated on every ninth validation row
with a random phase per episode (about 3,850 rows, 5 denoising steps). The
selected export is evaluated on every third row (about 11,500) at 5 and 10
steps with future-frame decode and a 200-observation serving-parity check,
then once on the test split. This is a deviation from "one observation per
row" for time: 16 exports at one row per observation would cost more GPU time
than training. Selection is decision 11.

## Gates before launch

1. **Scratch storage.** 95.7% used at the audit (about 217 GB free); by the
   time the pipeline was written usage had dropped to 81.7% (about 900 GB
   free), enough for both Cosmos runs. One Cosmos dense run needs about
   115 GB (16 exports at 3.9 GB plus two resumable checkpoints at 26 GB). The
   launcher re-checks the quota before every stage.
2. **Video base checkpoint.** Fetched September 19, 11:17 EDT with
   `examples/hanoi/fetch_video_base.sh`: 3,913,017,214 bytes, SHA-256
   `fbc4f05d948078539cb5d7a8e59b6f40f940e4836b5b8a31dcab03e3e807a6f0`,
   715 `net.*` tensors with the same keys and shapes as the LIBERO policy
   network. A first attempt through the Python downloader was killed by the
   4 GB interactive memory cap and left a hollow sparse prefix; resuming onto
   it produced a corrupt file, which was discarded.
3. **Per-user GPU cap.** Two concurrent GPU jobs per user were observed on
   September 17. With the OpenPI dense run and one Cosmos run, the second
   Cosmos run waits.

## Deviations from the guide

- Stage evaluations use every ninth row, the final ones every third (above).
- The guide's "waypoint server module" is not in this checkout; the server
  follows the official ALOHA `deploy.py` instead.
- The dataset was built here from raw rather than imported from the OpenPI
  build, because that build did not exist yet; the cross-check records
  agreement once it does.

## Results, run B (video init), jobs 18019908 + 18019909 + 18053668

Training: 16,000 updates in 12.7 h of GPU time at 2.15 s per update on one
H200, across two allocations (the first stopped at update 14,791 on its time
budget; the second resumed from that checkpoint). Peak allocated memory 28 GB.

Per-export validation (every ninth row, about 3,850 rows, 5 denoising steps):

| Step | Slot-1 mm (all / stationary / moving) | Mean over slots | Endpoint | Jaw acc. |
|---|---|---|---|---|
| 1,000 | 20.44 / 18.30 / 20.90 | 15.79 | 27.49 | 0.9794 |
| 2,000 | 7.20 / 7.10 / 7.22 | 7.40 | 11.21 | 0.9961 |
| 3,000 | 3.42 / 2.92 / 3.52 | 6.23 | 9.21 | 0.9967 |
| 4,000 | 3.59 / 3.43 / 3.62 | 5.44 | 7.31 | 0.9971 |
| 5,000 | 2.80 / 2.60 / 2.84 | 5.01 | 7.09 | 0.9977 |
| 6,000 | 2.45 / 2.42 / 2.45 | 4.52 | 6.22 | 0.9984 |
| 7,000 | 2.28 / 1.94 / 2.36 | 4.91 | 6.31 | 0.9972 |
| 8,000 | 1.65 / 1.43 / 1.70 | 3.88 | 5.72 | 0.9987 |
| 9,000 | 1.88 / 1.38 / 1.99 | 3.81 | 5.32 | 0.9987 |
| 10,000 | 1.53 / 1.25 / 1.59 | 4.16 | 5.80 | 0.9987 |
| 11,000 | 1.84 / 1.25 / 1.97 | 4.07 | 5.48 | 0.9990 |
| 12,000 | 1.53 / 1.26 / 1.59 | 3.89 | 5.54 | 0.9992 |
| 13,000 | 1.42 / 1.20 / 1.47 | 3.57 | 5.37 | 0.9988 |
| 14,000 | 1.16 / 1.01 / 1.19 | 3.40 | 5.20 | 0.9993 |
| 15,000 | 1.43 / 1.30 / 1.46 | 3.59 | 5.29 | 0.9995 |
| 16,000 | 1.17 / 1.12 / 1.18 | 3.35 | 4.82 | 0.9995 |

Decision 7 check: slot-1 was 2.45 mm at 6,000, under the 5 mm bar, so run A
(LIBERO init) was not launched. Selection (decision 11): step 16,000, the
lowest mean over slots with jaw accuracy above 0.99.

Selected export `exports/iter_000016000.pt`, SHA-256
`fbea3079d2f3e32446b03eb2787e28fa8c94cccc1687cd54114e83bab706abe5`.
Full validation (every third row, 11,537 scored, 5 all-padded rows skipped):

| Subset | Rows | Slot-1 mean / median / p95 | Within 2 mm | Mean over slots | Endpoint | Jaw acc. | Flips predicted | Flip timing |
|---|---|---|---|---|---|---|---|---|
| All | 11,537 | 1.16 / 0.94 / 2.77 mm | 88.3% | 3.25 mm | 4.68 mm | 0.9995 | 2,313 of 2,315 | median 0, p95 0 rows |
| Stationary | 1,954 | 1.12 / 1.00 / 2.36 mm | 92.8% | 6.61 mm | 11.13 mm | 0.9999 | 22 of 22 | median 0, p95 3 rows |
| Moving | 9,583 | 1.17 / 0.93 / 2.81 mm | 87.4% | 2.56 mm | 3.36 mm | 0.9994 | 2,291 of 2,293 | median 0, p95 0 rows |

Ten denoising steps change nothing (slot-1 1.16, mean 3.26, endpoint 4.70).
Future frame at 1.6 s: L1 0.015, PSNR 31.8 dB. Value error 0.002. Serving
parity on 200 observations: max first-slot difference 0.108 mm, mean 0.033 mm,
passed. Sampling noise between two noise draws: 0.48 mm mean, 4.0 mm max on
slot 1, so averaging a few samples at deployment is worth considering.

Ready-to-deploy criteria (guide section 7): slot-1 under 2 mm on both motion
subsets, jaw accuracy at least 0.99, serving parity passed, hashes recorded.
All met on validation. The balanced-accuracy field in
`selected_validation.json` is invalid (a summary bug, since fixed and
recorded in `code_updates.json`); with 99.95% accuracy over a roughly 50/50
open/closed split, balanced accuracy is at least 0.999.

Test split (locked after selection; every third row, 11,601 scored, 5 all-padded rows skipped), same export:

| Subset | Rows | Slot-1 mean / median / p95 | Within 2 mm | Mean over slots | Endpoint | Jaw acc. | Flips predicted | Flip timing |
|---|---|---|---|---|---|---|---|---|
| All | 11,601 | 1.15 / 0.94 / 2.77 mm | 88.8% | 3.25 mm | 4.73 mm | 0.9994 | 2,315 of 2,321 | median 0, p95 0 rows |
| Stationary | 1,957 | 1.15 / 1.01 / 2.61 mm | 91.7% | 6.69 mm | 11.24 mm | 0.9998 | 25 of 25 | median 0, p95 3 rows |
| Moving | 9,644 | 1.15 / 0.93 / 2.78 mm | 88.2% | 2.55 mm | 3.41 mm | 0.9994 | 2,290 of 2,296 | median 0, p95 0 rows |

Future frame PSNR 31.7 dB, value error 0.003. No validation-to-test gap. The
pipeline recorded completion on September 20, 02:17 EDT (job 18053668).
Nothing here is a hardware result; live success is measured on the arm.

A second training cycle (`hanoi_cosmos_dense_20260919_video_init_cycle2`,
jobs 18059379 + 18059380) was started on September 20 at the user's request,
from this export with a fresh optimizer and the same schedule shape, 16,000
more updates. Its exports are candidates only if they beat 3.25 mm mean over
slots with jaw accuracy above 0.99 on the same validation rows.

### Where the later-slot error comes from

The mean over slots grows from 1.2 mm at slot 1 to 4.7 mm at slot 16, and to
11 mm from a standstill. `examples/hanoi/dense_path_deviation_probe.py`
splits each slot's error into distance from the trajectory the arm actually
took (a placement error) and displacement along it (a timing error), on 577
validation rows:

| Slot | Total | Off-path mean | Off-path p95 | Timing offset |
|---|---|---|---|---|
| 1 (0.1 s) | 1.13 mm | 0.56 mm | 1.13 mm | 0.22 s |
| 4 (0.4 s) | 1.67 mm | 0.49 mm | 0.90 mm | 0.21 s |
| 8 (0.8 s) | 3.04 mm | 0.51 mm | 1.00 mm | 0.20 s |
| 16 (1.6 s) | 4.64 mm | 0.56 mm | 1.25 mm | 0.28 s |

Off-path error is about 0.5 mm at every horizon on both motion subsets. The
growth is timing: from a standstill the policy predicts departure about
0.45 s earlier than the recording. Asynchronous re-planning re-anchors timing
at every observation; the chunk must not be executed open loop to slot 16.

Also checked: slot-1 error is flat (1.10 to 1.31 mm) whether the arm is under
1 mm or 4 to 8 mm off the commanded reference, and on rows more than 3 mm off
the predicted next pose is 1.2 mm from the reference and 13.7 mm from the
arm's actual position. That is the corrective behaviour waypoint_v4 lacked.

### Evaluator fixes after training

Two bugs in the new evaluator surfaced during the final passes and are recorded
in the run's `code_updates.json` with old and new hashes: all-padded chunks
(the last three rows of an episode) crashed the metric code, and balanced jaw
accuracy was computed against the wrong quantity. Neither affects training,
exports, or any other metric.
