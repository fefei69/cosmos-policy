# Hanoi Cosmos Policy on Torch

This experiment is separate from the active OpenPI `hanoi_20260914` run. Its
source is this repository; caches, prepared data, logs, and checkpoints stay here.
Raw `/scratch/cw5167/datasets/*.h5` files are opened read-only.

## Source contracts

- Dataset handoff: `/scratch/cw5167/workspace/openpi/docs/hanoi_dataset_handoff.md`.
- HPC guide: `/scratch/cw5167/workspace/openpi/docs/torch_gpu_quickstart.md`.
- Scheduler: account `torch_pr_595_tandon_advanced`; omit partition and QoS.
- Start with a bounded, single-H100 pilot. The two-hour allocation is a pilot
  budget, not an estimate of time to converge or a claim of task success.

The upstream [README](../README.md#system-requirements) recommends eight 80GB
GPUs for training and reports 48 hours on eight H100s for its small ALOHA run.
It permits fewer GPUs with gradient accumulation. Our single-H100 configuration
still needs an actual forward/backward memory check: its BF16 weights and
gradients plus FP32 master weights and Adam moments use about 31.3 GB before
activations, the VAE, and workspaces. Gradient accumulation does not remove
that fixed storage cost or provide multi-GPU throughput. The README's 6–9GB
figures are inference requirements.

The selected source is **AAAA → CCCC only**, with 50 episodes. Episodes
0–39 are train, 40–44 validation, and 45–49 test. Observations use the stored
224x224 RGB and `proprio[[0,1,2,6]]`: measured XYZ and jaw opening. Velocity,
commanded jaw and collector annotations are excluded at the user's request.
The four-value state uses metadata format 2 under `aaaa_to_cccc_pos_only`;
older velocity-trained checkpoints are retained as historical runs. A valid anchor has a monotonic receipt age between zero and 50 ms.
All reference rows remain available as action labels.

Targets are 63 dense references starting at `action_abs[t]`, padded only with the
same episode's last reference. All XYZ targets subtract the *same* measured XYZ
at the observation anchor; jaw intent stays absolute. Normalization bounds are
computed only from eligible training anchors and their supervised targets.

## Model and validation

The configuration is `cosmos_policy/config/hanoi_config.py`. It preserves the
Cosmos Predict2 2B network and joint action, future-state, and value objective.
The seven latent slots are blank/current proprio/current RGB/actions/future
proprio/future RGB/value. The future state is at `min(t+63, episode_last)`.
The success return uses gamma=0.9995 per 30 Hz row (roughly a 46-second reward
half-life), so the four-minute episodes do not have almost entirely zero returns.

The initial public policy checkpoint is
`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`, with the public original
`Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` tokenizer. Revisions and SHA256 hashes are
recorded in `checkpoints/public/manifest.json`. Validate architecture/state keys
and actual GPU execution before using this combination for a long run. The
original Nvidia base video checkpoint requires Hugging Face gated-model access.

Full-parameter BF16 fine-tuning starts with batch size 2, gradient accumulation
8 (effective batch 16 on one GPU), learning rate 1e-5, a 100-step warmup, and
block activation checkpointing. These are conservative pilot settings; tune
batch size and the longer training schedule from measured memory/throughput and
held-out losses. No augmentation or JPEG recompression is applied to the stored
images. The timestep sampling and positional configuration match the released
policy. No full T5 encoder is loaded during training or inference.

Training records local `metrics.jsonl`. Validation uses fixed-noise denoising
loss on a limited held-out subset. `run_hanoi_eval.py` separately evaluates
actual five-step action predictions and reports first-action, nine-action, and
full-chunk XYZ errors in millimetres, jaw accuracy, latency, and GPU memory.
These are offline metrics, not physical task success. Test episodes remain
reserved unless explicitly selected in the evaluation command.

## CPU preparation

Use the isolated root `.venv` with the root `cu128` dependencies and Python 3.10.
The supplied native extension wheels target Python 3.10. Source the environment:

```bash
source examples/hanoi/env.sh
uv sync --extra cu128 --python /scratch/cw5167/uv-python/cpython-3.10-linux-x86_64-gnu/bin/python3.10
uv pip install --python .venv/bin/python --no-deps -r examples/hanoi/requirements.txt
.venv/bin/python examples/hanoi/download_public_assets.py
.venv/bin/python examples/hanoi/prepare_t5.py download
.venv/bin/python -m cosmos_policy.datasets.hanoi_data --directions aaaa_to_cccc
```

The T5 download is about 45 GB. Generate its two embeddings on a CPU allocation
with 64 GiB host RAM, using the FP32 encoder to match the existing policy's
text preprocessing. The source model has a legacy full-state checkpoint, so
memory during loading is substantially larger than the embedding output.
The reviewed Transformers 4.57.1 loader constructs the model on meta, loads the
42.12-GiB checkpoint, discards decoder tensors, and assigns the encoder weights
without a second full encoder copy. The retained FP32 encoder weights are
18.12 GiB. The 64-GiB request is an estimate with headroom, not a measured peak;
`prepare_t5.py` now records process peak RSS through loading and inference.

CPU preparation job **17820405** requests **64 GiB**, eight CPUs, and no GPU.
At the user's request, it replaced pending job **17818083**, which requested
128 GiB and was cancelled before allocation. Torch had rejected an in-place
memory reduction. The dependent pending GPU submission was also replaced so
it waits for the new CPU job. See
`data/hanoi_cosmos/operations/t5_memory_assessment.json` and
`data/hanoi_cosmos/operations/resource_resubmission_20260914.json` for the memory
estimate, exact submission arguments, and verified cancellations. Use Slurm
for live state and scheduling estimates.
The selected indices, normalization statistics, and audit reports live under
`data/hanoi_cosmos/aaaa_to_cccc/`. The earlier bidirectional indices in the parent
directory are not used. The loader and evaluator reject metadata containing an
unrequested direction. The shared text cache remains in the parent directory;
only the forward task embedding is consumed.
The selected file yields 282,303 training, 35,399 validation, and 35,284 test
anchors after freshness filtering; its 360,050 raw reference rows remain intact.
See [the suitability assessment](hanoi_cosmos_suitability.md) for the distinction
between dense reference labels and the collector's sparse controller commands.

Apply the Hanoi requirements after upstream `uv sync`: h5py 3.16 supplies HDF5
2.0, which can read the raw collection's Boolean attributes. The upstream lock's
h5py 3.15.1/HDF5 1.14.6 fails when reading `dry_run`; this is a reader-version
issue. Preserve the original HDF5 files.

```bash
sbatch --test-only --chdir="$PWD" examples/hanoi/prepare.sbatch \
  examples/hanoi/prepare_t5.py encode --threads 8 --precision float32
sbatch --parsable --chdir="$PWD" examples/hanoi/prepare.sbatch \
  examples/hanoi/prepare_t5.py encode --threads 8 --precision float32
```

## GPU pilot and resume

**September 15 update:** pilot continuation **17830640** completed at step
**3,341**, including checkpoint export and 100-sample validation. Its final
training action L1 was 0.0276, above ALOHA's suggested 0.01. The following
September 14 submission notes are historical. See
[the longer-training plan](hanoi_cosmos_long_training.md) for the requested
continuation toward that target and its live status artifacts.

The current forward-only pilot retry is GPU job **17824058**, submitted
September 14 at 22:46 EDT. It requests one H100, eight CPUs, and 128 GiB host RAM
for **1:58:00**, using the completed CPU preparation from **17820405**. Its run
name is `hanoi_cosmos_aaaa_to_cccc_20260914_pilot_retry1`; the exact arguments
and budget are in
`data/hanoi_cosmos/operations/pilot_aaaa_to_cccc_retry1_submission.json`.
The training stop is 6,420 seconds after launch, leaving ten minutes for final
checkpoint/export/evaluation inside the allocation.

The preceding GPU job **17820432** ran on `gh006` from 22:03:03 to 22:04:54 EDT
and failed before any optimizer step. Strict initialization rejected 28 empty
Transformer Engine attention metadata entries. Inspection confirmed that all
28 contain the same four-byte serialization of `None`. The loader now omits
only empty metadata for existing attention operators with no corresponding
state, retaining strict checks on learned weights and nonempty metadata.
Nine loader regression tests pass, including activation-checkpoint wrappers.
The failed allocation used 111 seconds; the retry's 7,080-second limit keeps
combined GPU allocation time within the authorized 7,200 seconds.

GPU training memory fit, throughput, and policy latency remain unmeasured.
Use Slurm and the run's `pipeline.json` for live execution status. The original
submission record is
`data/hanoi_cosmos/operations/pilot_aaaa_to_cccc_64g_prep_submission.json`.
That job replaced **17819691**, cancelled while pending to replace its CPU
dependency; the suitability audit, 86 CPU tests, and actual launcher dry-run
passed before its submission.

The September 14 pilot was submitted as GPU job **17818517**, then cancelled
while still pending when the user clarified a 50-episode scope. The user then
selected AAAA → CCCC and the configuration was restricted to that file.
No GPU execution occurred in the cancelled job. The original submission
arguments are recorded in `data/hanoi_cosmos/operations/pilot_submission.json`.
At submission, 84 CPU tests and the actual offline launcher dry-run passed;
GPU memory, kernel execution, and training throughput remained unmeasured.
Use Slurm for the current job state rather than treating this note as live status.

Choose a unique run name and keep it stable only when resuming that same run.
The output path is
`data/hanoi_cosmos/runs/cosmos_policy/hanoi/$HANOI_RUN_NAME/`.

```bash
source examples/hanoi/env.sh
export HANOI_RUN_NAME=hanoi_cosmos_aaaa_to_cccc_20260914_pilot
sbatch --test-only --chdir="$PWD" examples/hanoi/train.sbatch
sbatch --parsable --chdir="$PWD" examples/hanoi/train.sbatch
```

Record the actual returned job ID and submission arguments. Do not treat the
ID or estimate printed by `--test-only` as a submitted job. Monitor with
`squeue -u cw5167`, `scontrol show job JOBID`, and `sacct -j JOBID`.

`run_pilot.py` runs the following phases inside the same two-hour allocation:

1. Train two optimizer steps at the pilot's batch size and accumulation, then
   save a full checkpoint.
2. Load that checkpoint for a two-anchor inference qualification.
3. Resume training, restoring optimizer, scheduler, iteration, random states,
   and the next position in the training sample order.
4. Save the final checkpoint, export a model-only `.pt`, and evaluate 100
   validation anchors from the five held-out forward episodes.

The script reserves ten minutes for shutdown, export, and evaluation by asking
the trainer to stop at an optimizer boundary before 110 minutes. All phases
share the allocation deadline. `pipeline.json` records each phase and failure;
`metrics.jsonl` records training, validation, checkpoint, and memory events.
`data_order.json` prevents a resume with incompatible batch or dataset settings.
The Hanoi optimizer also preserves its FP32 master weights and Adam moments;
complete checkpoints occupy approximately 27 GB, including the policy weights.
The qualification and final inference reports contain measured action errors
and GPU resource usage. A completed pilot does not establish convergence.
Preemption is disabled for this pilot.

Inspect failures before retrying, preserve completed checkpoints, and avoid
duplicate submissions. Only replace a verified still-pending job owned by this
Cosmos experiment, with a materially better compatible resource estimate.
Never cancel or write the OpenPI run. For long queue waits, follow the retry
thresholds in the source Torch guide rather than repeatedly polling/submitting.

## Offline inference

Within a GPU allocation, use the prepared embeddings and a local checkpoint:

```bash
source examples/hanoi/env.sh
.venv/bin/python -m cosmos_policy.experiments.robot.hanoi.run_hanoi_eval \
  --checkpoint /absolute/path/to/checkpoints/iter_000000100/model \
  --metadata-dir data/hanoi_cosmos/aaaa_to_cccc --samples 100 \
  --output /absolute/path/to/run/validation_actions.json
```

This entry point sends no robot commands. Deployment on the RTX 5080 still
requires a test on that card; H100 timing does not establish 5080 timing.
