# Hanoi Cosmos training without velocity

## Current run

The user requested removal of XYZ velocity on September 15. Cosmos job
**17850256** was stopped after 6 minutes 37 seconds; Slurm confirmed cancellation.
The earlier step-3,341 checkpoint and all old artifacts are preserved. OpenPI
and CPU jobs were not changed.

The replacement is a fresh run named
`hanoi_cosmos_aaaa_to_cccc_pos_only_20260915`, initialized from the existing public
`Cosmos-Policy-LIBERO-Predict2-2B.pt` weights. It does not resume the
velocity-trained weights or optimizer. Its submission and actual job ID are in
`data/hanoi_cosmos/operations/no_velocity_20260915_submission.json`.

## Inputs and targets

- One fixed third-person RGB image, 224 × 224, and the cached forward-task prompt.
- Four measured state values: **X, Y, Z, jaw opening**, selected from raw
  `proprio` columns **[0,1,2,6]**. Both current and auxiliary future state exclude
  velocity (columns 3–5) and commanded jaw (column 7).
- Metadata format **2**, schema `xyz_measured_jaw_v1`, in
  `data/hanoi_cosmos/aaaa_to_cccc_pos_only`. The loader rejects the old metadata
  and a resume into the old state schema. Serving rejects seven/eight-value
  state vectors and uses four-value normalization statistics.
- Action horizon remains **63 × 4**: Cartesian XYZ relative to one measured
  observation anchor and absolute jaw intent. All 30 Hz reference labels remain.
- The raw recordings are read-only. Episode splits remain 0–39 train,
  40–44 validation, 45–49 test. Normalization is recomputed from training data.

Regression tests change the recorded velocity and commanded-jaw fields and
verify that observations, future-state targets, actions and normalization do
not change. Real-data checks also compare training and evaluation state packing.

## Training budget and checks

- One H100, eight CPUs, 128 GiB RAM, at most **36 hours** and **50,000 steps**.
- Effective batch **16**: batch 2 × gradient accumulation 8 on one GPU.
- Peak learning rate **1e-5**, 2,000-step warmup, gradual decay to **3e-6** at
  step 20,000, then a fivefold drop to **6e-7** for the remaining steps. This
  uses ALOHA's schedule shape with the Hanoi adaptation's lower peak LR.
- First qualify two real optimizer steps, a complete checkpoint save/reload,
  and two action predictions using the four-value state. Continue only after
  this succeeds, restoring all new-run state from that checkpoint.
- Save full checkpoints every 2,000 steps and at stage ends. Export and evaluate
  100 fixed validation anchors at 10k, 20k, 35k and 50k, or a time-budget stop.
- At those evaluations, stop early when the latest 500 logged optimizer steps
  average training action L1 ≤ **0.01**, all three mean XYZ distances are no
  worse than the old pilot report, and jaw accuracy is within two percentage
  points of that report or better. The old report supplies comparison numbers
  only; its velocity observations and weights are not used by the new policy.
- Preserve 20 minutes for finalization and 60 seconds before the allocation limit.
  No automatic job resubmissions or additional GPU allocations occur.
- Check quota before qualification and before every stage. Allow **800 GB** for
  at most 27 complete checkpoints, four exports and small reports, and reserve
  at least 150 GB extra headroom before launching a training stage. Preserve all
  completed checkpoints. Shared usage may still change during a stage.

The launcher is `examples/hanoi/train_long.sbatch`, calling
`examples/hanoi/run_long.py`. Set `HANOI_RUN_NAME` to the new run and
`HANOI_BASELINE_REPORT` to the preserved pilot's 100-sample validation JSON.
The launcher sets `HANOI_TRAINING_SCHEDULE=aloha` and explicitly removes the
legacy continuation-anchor override. The resolved config and `pipeline.json`
record the chosen schedule and state schema.

`metrics.jsonl` records training every 10 steps and denoising validation every
250. Complete checkpoint saves are validated, non-finite losses fail, and the
trainer has a stall timeout. Each subprocess has a deadline inside the single
Slurm allocation. Node-local temporary files avoid the earlier NFS cleanup issue.

A completed training allocation is not a deployment qualification. Inspect
`stop_reason` and `assessment.training_target_met`; physical task rollouts,
controller integration and deployment-GPU timing remain necessary.
