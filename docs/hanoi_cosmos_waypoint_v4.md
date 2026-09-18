# Cosmos Hanoi training on waypoint_v4 labels

Run `hanoi_cosmos_waypoint_v4_20260917`, Slurm job 17929493, submitted
September 17, 2026, 18:25 EDT. The joint_v3 run (`hanoi_cosmos_joint_sparse_20260917`,
job 17904589) was cancelled at step 17,710 on September 17, 19:21 EDT, to free
its GPU slot; its exports through step 16,000 are kept and none of its modules
were edited.

## Why a second run

The joint_v3 labels are Ramer-Douglas-Peucker path points. Verified today by
recomputing from this repository's own archive (the OpenPI write-up is
`openpi/docs/hanoi_label_issue_for_cosmos_20260917.md`): among 75,339 pairs of
training observations in the same task phase with XYZ within 1 mm and joints
within 0.01 rad, 14.6% have first-target labels more than 10 mm apart (maximum
138 mm), and the fraction rises to 63% by slot 8. The Cosmos split files are
byte-identical to the OpenPI joint_v3 archives, so job 17904589 trains on those
labels. With the same labels OpenPI's free-motion first-target error stayed near
10 mm from step 4,000 to 30,000.

waypoint_v4 labels are the recording's own motion-leg endpoints plus gripper
events: 0 of 69,773 similar pairs disagree, every episode has 89 to 90
observations, and the destinations form a set of 18 points. Observation
contract, image crop, horizon 8, prefix-1 execution and the episode splits
(0–39 / 40–44 / 45–49) are unchanged.

## Data

| Item | Value |
|---|---|
| Importer | `cosmos_policy/datasets/hanoi_waypoint_data.py` (contract `hanoi_waypoint_v4_cosmos_v1`) |
| Source archives | `openpi/data/hanoi/waypoint_v4/indices/aaaa_to_cccc_{train,val,test}.npz` |
| Samples | 3,589 / 445 / 450 |
| Gates | audit hash equals `verification.json`; archive hashes equal the audit; raw h5 size and hash equal the v3 handover; label consistency 0.0% over 10 mm; admission passed; contract differs from v3 only in the four label-extraction fields |
| Regenerated | `dataset_statistics.json` (Cosmos min/max from the v4 training split plus its auxiliary future states); auxiliary future rows (last selected target row + 1) |
| Output | `data/hanoi_cosmos/waypoint_v4/` with label provenance in `metadata.json` |

The same numeric audit as joint_v3 runs on every archive (bounds, padding,
freshness, binary jaw intent, labels equal `action_abs` at the target rows),
with the split sizes taken from the audit instead of constants.

## Training configuration

| Setting | waypoint_v4 value | joint_v3 value |
|---|---|---|
| GPU | one H200 or H100 (`h200_tandon`, `h200_public`) | one H100 |
| Micro-batch × accumulation | 16 × 2 = 32 (`HANOI_WAYPOINT_MICROBATCH`) | 2 × 16 = 32 |
| Activation checkpointing | none (`HANOI_WAYPOINT_ACTIVATION_CHECKPOINT`) | block-wise, all 28 blocks |
| Updates | 8,000 (about 71 epochs) | 30,000 |
| Save / export | every 1,000 | every 2,000 |
| Learning rate | 400-update warm-up to 1e-5, linear decay to 3e-6 at 8,000, then 6e-7 | ALOHA shape over 20,000 |
| Monitoring validation | 64 held-out examples every 500 updates, fixed noise | every 1,000 |
| Scalar metrics | kept on the GPU, read once per 10 updates | six host syncs per micro-batch |
| Everything else | unchanged: Cosmos 2B network, FusedAdam with FP32 master weights, BF16, weight decay 0.1, clipping 1.0, no EMA, seed 195, resumable data order and RNG |

The joint_v3 run spent most of each 4.3-second update on 16 tiny passes with
every block recomputed. The larger micro-batch and no recompute are expected,
not measured, to be 2 to 3 times faster per update. Peak memory and seconds per
update are recorded by the qualification stage. A memory fallback is a resubmit
with `HANOI_WAYPOINT_MICROBATCH=8`.

Because metrics are read back once per logging interval, a non-finite loss is
detected up to 10 updates late; the native gradient-clipping callback still
runs on every update.

## Stages, evaluation and selection

`examples/hanoi/run_waypoint.py` (batch script `examples/hanoi/train_waypoint.sbatch`):

1. Preflight: 160 GB scratch headroom; allocation seconds are parsed from the
   real Slurm time limit so the stop-before-deadline logic is correct for any
   `--time`.
2. Qualification: two updates, save, reload the optimizer/RNG/data order, one
   more update, export, two-sample validation with serving parity.
3. Stages of 1,000 updates to 8,000. After each export the physical decode
   evaluation runs on all 445 validation examples (`--mode actions`).
4. Selection: maximum first-target hit rate (within 5 mm and correct jaw
   intent); ties by lower mean first-target XYZ error, then earlier step.
   `selection.json` is written before any test-split access.
5. The selected export is evaluated on validation and test in `both` mode.

Denoising loss is not used for selection. OpenPI's minimum-loss choice picked a
checkpoint that was physically worse than later ones, and the joint_v3 run's
monitoring loss rose after step 5,000 for reasons unrelated to action accuracy.

`cosmos_policy/experiments/robot/hanoi/run_hanoi_physical_eval.py` evaluates
joint_v3 or waypoint_v4 exports on one H100 or H200 (it reads the contract from
the metadata) and adds, for the committed first target: p95 and median XYZ
error, the fraction within tolerance, jaw-intent accuracy and the hit rate.
Because the v4 destinations are discrete and the shortest move is 40 mm, mean
error is nearly all-or-nothing per sample; read the hit rate first.

Job 17929390 runs this evaluator over every joint_v3 export on the full
validation split, writing to `data/hanoi_cosmos/evals/hanoi_cosmos_joint_sparse_20260917/physical_val/`.

## Scheduling note

`h100_tandon` is capped by a 60-GPU group limit (`QOSGrpGRES`), so one-GPU jobs
were estimated to wait until the next evening even with free H100s on the nodes.
`h200_tandon` had two idle nodes and estimated an immediate start. Without a
partition the account defaults to `a100_tandon`. Requests use the account
`torch_pr_595_tandon_advanced` and no QoS.

## Caveats

- Offline imitation metrics only; nothing about hardware success is measured.
- Training observations were recorded while the arm was moving; deployment
  observes after stopping. That approximation is unchanged from joint_v3.
- With 18 destinations the first-target task is close to classification plus a
  jaw intent; hit rate and p95 are the informative numbers.

## Commands

```bash
# Prepare (refuses an existing output)
PYTHONPATH=. .venv/bin/python -m cosmos_policy.datasets.hanoi_waypoint_data
# Train (from the repository root)
sbatch examples/hanoi/train_waypoint.sbatch
# Memory fallback
HANOI_WAYPOINT_MICROBATCH=8 sbatch examples/hanoi/train_waypoint.sbatch
# Physical evaluation of joint_v3 exports on the H200 pool
sbatch examples/hanoi/eval_joint_exports.sbatch
```
