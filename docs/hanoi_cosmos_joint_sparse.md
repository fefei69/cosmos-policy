# Cosmos: September 15 joint-state / sparse-waypoint experiment

> **September 17, 2026 update.** The joint_v3 labels used here are ambiguous for about 15% of
> similar situations (verified from this repository's own archive). Run 17904589 was cancelled at step 17,710 on
> September 17 (exports through 16,000 are kept) to free its GPU slot for the corrected-label run; the corrected-label run and the physical selection rule are in
> [hanoi_cosmos_waypoint_v4.md](hanoi_cosmos_waypoint_v4.md).

This is a separate experiment from the September 10, XYZ-state, 63-tick Hanoi
policy. Its starting point is the released **Cosmos Policy LIBERO Predict2 2B**
checkpoint. The old step-28,000 checkpoint is not a resume source for this task.
pi0.5 weights and its optimizer are OpenPI-specific and are not loaded by Cosmos.

## Audited inputs

Original handover:
`/scratch/cw5167/workspace/openpi/docs/hanoi_cosmos_training_handoff.md` and `.json`.
OpenPI recordings, prepared archives, code, manager and checkpoints are read-only
inputs to this implementation.

The five uploaded September 15 files are present in `/scratch/cw5167/datasets`.
Both recordings contain 50 successful episodes and 360,050 rows. Independent
checks read all numeric datasets, decoded 300 sampled RGB frames per direction
(including every episode start/middle/end), hashed both complete files, and
confirmed that neither changed during the audit. The forward SHA256 exactly
matches the handover:
`d235dbc8c11628bb4617de95a437ca2d91c100d63bd4718cbb9257d809177663`.
This is not an exhaustive visual inspection of every frame.

Cosmos preparation copies the existing audited NPZ files without changing their
contents, verifies the handover hashes, and checks **all 6,254** admitted states,
XYZ contexts and eight-target chunks against the raw HDF5. Copies and native
normalization are in `data/hanoi_cosmos/joint_sparse_v3/`.

| Split | Episodes | Examples | Padded target slots |
|---|---|---:|---:|
| Train | 0–39 | 5,031 | 1,113 |
| Validation | 40–44 | 611 | 140 |
| Test | 45–49 | 612 | 140 |

The target extraction, 0.5 mm RDP tolerance, preserved gripper events, admitted
moving observations and split assignments are inherited unchanged from the
handover. Images and sparse destinations use their explicit **raw source rows**.
Neither compact row offsets nor eight consecutive raw ticks reconstruct these
labels. Reverse examples are not used in this forward experiment.

## Observation, coordinates and deployment

- One already-cropped RGB224 external camera; no wrist view or history.
- Learned state: six measured joint angles in SDK order 0–5 (radians), then
  measured gripper stroke (metres). No velocity, Cartesian XYZ, commanded jaw,
  motion phase or board state enters the learned state.
- Measured XYZ from the same feedback snapshot is separate conversion context.
- Targets: eight absolute Cartesian XYZ destinations and jaw intent (1=open,
  0=close). Internally, every target's XYZ subtracts the **same observation XYZ**.
  Decoding adds that same XYZ; joint angles never serve as a Cartesian anchor.
- Terminal target repeats participate in loss and normalization. Physical
  accuracy metrics exclude `actions_is_pad` positions.
- Prompt: “Move all four rings from peg A to peg C following Tower of Hanoi rules.”

`HanoiJointPolicy.infer()` accepts the handover's `observation/image`,
`observation/state`, `observation/cartesian_position` and optional matching
`prompt`. It returns eight absolute actions with `commit_count=1` and
`reference_rate_hz=None`. It makes no robot connection.

The external executor commits only the first destination. Keep the old jaw intent
during the Cartesian move; confirm controller readiness and measured arrival;
then apply a changed jaw intent and finish its dwell before obtaining another
fresh observation. Opening: 0.034 m over 1 s. Closing: -20 N over 1.2 s plus 0.2 s
settling. Fixed orientation is RPY `[0, pi/4, 0]`. Do not skip destinations based
on inference latency. Hardware arrival tolerance and task completion remain the
local robot adapter's responsibility; neither extraction nor observation-matching
tolerances establish robot arrival.

## Cosmos-specific choices

The native network uses seven latent slots:

1. Blank start.
2. Current measured joint/gripper state.
3. Current external RGB.
4. Eight-target action chunk.
5. Auxiliary future measured joint/gripper state.
6. Auxiliary future external RGB.
7. Auxiliary discounted terminal success value.

Only the first three slots condition policy inference. The future slots are
training targets and generated outputs, never future observation inputs. The
stock joint action/future-state/value denoising objective is retained.

**Additional temporal alignment:** auxiliary future image/state use raw row
`min(last selected target source row + 1, episode end - 1)`. This is a later raw
feedback/image snapshot, **not evidence of completed arrival** at the last target.
Host image receipt can lag the command, and receipt is not exposure time. At
terminal padding the final recorded snapshot is reused. Auxiliary value uses
`2 * 0.9995 ** remaining_raw_ticks - 1`. No uniformly timed video is reconstructed
from the compact sparse observations. The four repeated raw frames per synthetic
slot and `fps=16` are the pretrained VAE packing convention, not an execution rate.

Normalization uses native Cosmos min/max scaling without clipping. It is fitted
only to the training action chunks, training observation states and training
auxiliary future states. Constant state features receive a small finite span.
The same bounds and inverse are verified at inference. This differs from OpenPI's
q01/q99 normalization, so scalar losses are not comparable between the models.

RGB is the existing frozen crop/resize, with native VAE scaling. No additional
random augmentation or JPEG conversion is enabled for this Cosmos baseline.
Current and future labels are fetched with per-worker HDF5 handles. Source size,
mtime, NPZ hashes, statistics and data-order identities are checked on load/resume.

## Training and checkpoint management

| Setting | Cosmos value |
|---|---|
| Initialization | Local released Cosmos Policy LIBERO Predict2 2B |
| Updates | 30,000 completed optimizer updates |
| Save/export cadence | Every 2,000 updates; final 30,000 |
| GPU | One H100 80 GB; evaluation also one H100 |
| Batch | 2 per microbatch × 16 accumulation = 32 |
| Precision | BF16 model; FP32 Adam master weights/moments |
| Optimizer | Native Cosmos FusedAdam; beta 0.9/0.99, epsilon 1e-8, weight decay 0.1 |
| Learning rate | 2,000-update warmup to 1e-5; linear decay to 3e-6 at 20k; drop to 6e-7 and hold |
| EMA | Disabled, consistent with this Cosmos adaptation |
| Gradient clipping | Native Cosmos global norm 1.0 |
| Seed | 195; resumable sample-order and noise RNG state |
| Monitoring validation | 64 representative held-out examples every 1,000 updates |
| Scalar logging | Every 10 updates; JSONL and W&B |
| Inference sampling | Five native Cosmos denoising steps |
| Resources | 8 CPUs, 128 GB RAM, maximum 48 hours per allocation |

This is full fine-tuning with native Cosmos loss and optimizer behavior. It does
not copy pi0.5 flow loss, its LR/EMA, discrete state tokenization or native padded
width 32. The effective batch and 30k update budget match the handover. The older
run's 0.01 early-stop condition does not stop this experiment early.

Before the full budget, qualification performs two updates, saves, reloads the
complete optimizer/RNG/data order, performs one more update, exports, and checks
validation inference/serving parity. This is an implementation check, not a
5,000-update training trial. A qualification run uses a distinct run identity.

At each 2k stage, a validated inference export is retained. Once a newer complete
resumable checkpoint and export exist, older resumable checkpoints **from this
new run only** are removed. Keep all 15 scheduled inference exports and one full
resumable checkpoint, with space for another during saving. Exporting does not
modify original DCP files. A process lock prevents concurrent writers to a run.

The launcher checks quota before stages, stops before its allocation deadline,
and records a recoverable checkpoint. It does not automatically resubmit jobs.
Code/data/batch identities must match to resume. An interrupted partial stage is
continued from its latest complete checkpoint.

## Evaluation

After all 30k updates, evaluate every scheduled export on all 611 validation
examples using fixed-noise joint denoising loss; choose the minimum, breaking ties
by earlier step. Write `selection.json` before accessing the test split. Evaluate
the selected policy on all validation and all 612 test examples. Test evaluation
requires that locked checkpoint identity.

Physical results use absolute coordinates and report mean first-target error,
per-observation mean valid-horizon error, last-valid-target error, first-target
open/close event errors, nonpadded jaw confusion/support and balanced accuracy,
and per-episode results. The actual serving adapter and offline path are checked
using the same seeded inference noise. No raw images or weights are uploaded to
W&B. These are offline imitation metrics, not robot task success.

## Commands

From this repository, preparation (refuses existing output):

```bash
PYTHONPATH=. .venv/bin/python -m cosmos_policy.datasets.hanoi_joint_data
```

Bounded GPU qualification:

```bash
sbatch --time=00:25:00 \
  --export=ALL,HANOI_JOINT_ALLOCATION_SECONDS=1500,HANOI_JOINT_WANDB_MODE=disabled \
  examples/hanoi/train_joint.sbatch --qualify-only \
  --run-name hanoi_cosmos_joint_qualification_20260917
```

Full run (or resume the same run identity after interruption):

```bash
sbatch examples/hanoi/train_joint.sbatch \
  --run-name hanoi_cosmos_joint_sparse_20260917
```

The full run may be submitted with `--dependency=afterok:<qualification_job>` and
`--kill-on-invalid-dep=yes` to require successful qualification before scheduling.
Run state lives under `data/hanoi_cosmos/runs/cosmos_policy/hanoi/<run>/`.

## Verification status

Independent source audit and numeric handover equality passed. Seven new CPU
contract tests and 54 inference/sampler/optimizer checks passed (61 distinct
tests, including the new seven-state/eight-target native packing case).
CPU configuration resolution also passed; CUDA configuration validation ran in
the GPU job. Qualification **17904534 completed successfully in 5m49s**: two
updates, full state resume for update three, export, finite validation/inference,
and exact offline/serving action parity with the same seed. These three warmup
updates are an implementation check, not evidence of trained policy accuracy.

The separate 30k job **17904589** was submitted with `afterok:17904534` and
`--kill-on-invalid-dep=yes`; the qualification dependency has now been satisfied.
Its run name is `hanoi_cosmos_joint_sparse_20260917`. Current scheduler status
must be refreshed before describing it as running or complete. Submission and
updated status are recorded under `data/hanoi_cosmos/operations/handoff_20260917/`.

The explicit moving-demonstration versus stopped-deployment approximation remains.
The forward-only fixed-scene split does not establish new-camera/new-board
generalization. Cartesian command simplification does not certify physical path
clearance or contact success.
