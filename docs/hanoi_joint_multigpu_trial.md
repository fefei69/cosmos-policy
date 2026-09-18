# Joint/sparse Cosmos: multiple-GPU trial

Submitted September 16, 2026, 21:49 EDT as **17906554**. This is a separate,
bounded qualification, authorized by the request to try multiple GPUs. The
one-H100 full training job **17904589** continues independently; none of its
source files or checkpoint identities were changed. OpenPI job 17903508 is also
untouched.

## Resources and learning settings

- One node, four H100s, 16 CPUs, 256 GiB host RAM, 30-minute maximum.
- Native distributed data parallel (DDP): full model and optimizer on each GPU;
  gradients are averaged at the end of each accumulation window.
- Batch 2 per GPU, accumulation 4, global batch **32**.
- Seven measured state inputs, one external RGB view, eight sparse Cartesian
  targets, no velocity. The existing audited dataset, splits, normalization,
  initialization, optimizer, learning-rate schedule, and objective are reused.
- Public Cosmos initialization in a new run. No ongoing training checkpoint is
  migrated to a different GPU count. Equal global batch does not make the
  distributed random draws or epoch-tail sampling identical to the single-GPU run.
- W&B disabled for this short infrastructure trial; JSONL metrics and five-second
  GPU utilization/memory/power samples are retained locally.

## Qualification sequence

1. Complete five optimizer updates and save the full training state.
2. Launch a fresh four-process trainer, reload step five, and reach step 25.
3. Verify exact agreement of averaged gradients between ranks on the first
   update of each process launch.
4. At saves, verify exact agreement of all model parameters and FP32 Adam
   moments/master weights across ranks. Save an independent Python/NumPy/Torch
   random-state record for each rank.
5. On reload, verify exact model bytes and optimizer bytes, including optimizer
   step and LR; restore each rank's own random states and sampler position.
6. Measure update time after initial warmup/audits, excluding checkpoint export
   and save time. Export the final model and run two validation examples through
   the existing single-GPU inference/serving parity check.
7. Publish `qualification.json` only if all phases succeed; exit the allocation.

The trial does not establish convergence, policy accuracy, deployment readiness,
or bitwise equivalence with a one-GPU trajectory. The inference examples only
check the training/export/serving path. Full training on four GPUs remains a
separate decision after the trial's actual speed and recovery results exist.

## Checks completed before submission

- 44 CPU tests passed: new RNG/allocation, tensor/optimizer hash, and distributed
  sample/resume checks together with existing resumable-sampler regressions.
- Configuration preflight resolved DDP, four accumulation batches, unchanged
  model dimensions, and global batch 32. Shell syntax and whitespace checks passed.
- Slurm accepted the requested four H100s, 256 GiB and 30-minute cap.
- Running single-GPU source hashes still exactly match its saved run identity.
- At submission, one-H100 training reached step 60 with finite objective 4.81766;
  its early median update interval was about 4.27 seconds. This is a provisional
  baseline, not a measured multi-GPU speedup.

Latest scheduler check at about 21:50 EDT: pending with `QOSGrpGRES` (the shared
QoS GPU allocation limit), estimated start **September 16 at 23:00:10 EDT**.
This replaced the earlier dry-run estimate of September 17 at 05:40 EDT;
scheduler estimates are not reservations.

## Files and commands

```bash
sbatch examples/hanoi/qualify_joint_multigpu.sbatch
squeue -j 17906554,17904589
squeue --start -j 17906554
```

Trial run directory:
`data/hanoi_cosmos/runs/cosmos_policy/hanoi/hanoi_cosmos_joint_4h100_trial_17906554/`

Scheduler logs:
`data/hanoi_cosmos/logs/cosmos-joint-multigpu-17906554.{out,err}`

Entry point: `examples/hanoi/qualify_joint_multigpu.py`.
Config: `cosmos_policy/config/hanoi_joint_multigpu_config.py`.
Distributed monitoring/recovery: `cosmos_policy/utils/hanoi_multigpu_training.py`.
