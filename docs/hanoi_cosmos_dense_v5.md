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

## Results

Not yet run.
