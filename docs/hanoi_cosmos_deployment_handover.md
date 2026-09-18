# Hanoi Cosmos policy: deployment handover

Written September 17, 2026, for the agent working on the `hardware` branch.
Updated September 18 after the selected export was loaded and checked on a
deployment machine (RTX 5080); see sections 2, 3, 6 and 8.
Everything here describes what the trained policy expects and produces. Nothing
about robot task success has been measured; the acceptance work in section 6 is
what establishes that.

## 1. Read these first

| File | Why |
|---|---|
| `docs/hanoi_cosmos_waypoint_v4.md` | The run that produced the deployable checkpoint, its selection rule, caveats |
| `docs/hanoi_cosmos_joint_sparse.md` | The observation/target contract shared by v3 and v4 |
| `cosmos_policy/experiments/robot/hanoi/waypoint_policy.py` | Loader and serving adapter to call |
| `cosmos_policy/experiments/robot/hanoi/run_hanoi_physical_eval.py` | Offline evaluation and serving-parity check (cluster only; section 6 has the deployment-machine variant) |
| `examples/hanoi/dream_local.py` | Parity and future-frame check on any CUDA GPU, from a single-episode extract |
| `data/hanoi_cosmos/waypoint_v4/metadata.json` | The full contract under `deployment` (crop, frame, jaw parameters, timing) |
| OpenPI `docs/hanoi_deployment_handoff.md` | The robot-side executor this policy is meant to plug into |

Do not modify `waypoint_policy.py`, `joint_policy.py`, `cosmos_utils.py` or the
dataset modules on the hardware branch. The serving-parity check compares the
adapter with offline inference; changing them silently invalidates that check.
Put robot I/O in a new module that calls `HanoiWaypointPolicy.infer`.

## 2. Artifacts

Run `hanoi_cosmos_waypoint_v4_20260917` completed September 18, 01:08 EDT.
`selection.json` names `exports/iter_000008000.pt`.

| Artifact | Path on the cluster |
|---|---|
| Run directory | `data/hanoi_cosmos/runs/cosmos_policy/hanoi/hanoi_cosmos_waypoint_v4_20260917/` |
| Selected export | `exports/iter_000008000.pt` (3.7 GB), named in `selection.json` |
| Contract identity | `joint_contract.json` in the run directory; the loader reads it from two levels above the export, so **copy the run directory layout**, not the `.pt` alone |
| Normalization | `data/hanoi_cosmos/waypoint_v4/dataset_statistics.json` (hash is checked against the contract) |
| Prompt embedding | `data/hanoi_cosmos/t5_embeddings.pkl` (no T5 encoder is needed at runtime) |
| VAE | `checkpoints/public/Wan2.1_VAE.pth` |
| Base checkpoint | not needed at runtime |

Minimum copy for a deployment machine:

```
<run>/joint_contract.json
<run>/exports/iter_000008000.pt
data/hanoi_cosmos/waypoint_v4/{metadata.json,dataset_statistics.json}
data/hanoi_cosmos/t5_embeddings.pkl
checkpoints/public/Wan2.1_VAE.pth
```

For the acceptance checks in section 6 add `<run>/selected_validation.json` and
the episode-40 extract `data/hanoi_cosmos/exports_local/hanoi_episode_040.h5`
(made by `examples/hanoi/extract_episode.py`, 569 MB).

The simplest layout is the repository root of a `hardware` checkout, so every
default path resolves without overrides. From that root one rsync recreates it;
`-R` with the `/./` marker keeps the paths relative to the repository:

```bash
SRC=cw5167@dtn.torch.hpc.nyu.edu:/scratch/cw5167/workspace/cosmos-policy
RUN=data/hanoi_cosmos/runs/cosmos_policy/hanoi/hanoi_cosmos_waypoint_v4_20260917
rsync -avhPR --ignore-missing-args \
  "$SRC/./$RUN/joint_contract.json" "$SRC/./$RUN/selection.json" \
  "$SRC/./$RUN/selected_validation.json" "$SRC/./$RUN/exports/iter_000008000.pt" \
  "$SRC/./data/hanoi_cosmos/waypoint_v4" "$SRC/./data/hanoi_cosmos/t5_embeddings.pkl" \
  "$SRC/./data/hanoi_cosmos/exports_local/hanoi_episode_040.h5" \
  "$SRC/./checkpoints/public/Wan2.1_VAE.pth" .
```

The export's SHA-256 is
`64208a3da1806524e073052a37f971336fa40d8df466b14ce3a1c089bad20b1f`. If the
files must live outside the repository, set `HANOI_VAE_PATH` to the VAE and
pass the other three paths explicitly; the loader still needs
`joint_contract.json` two directories above the `.pt`.

Runtime: one CUDA GPU with about 10 GB free. Only the evaluation scripts insist
on H100/H200; the loader does not. Install with
`uv sync --extra cu128 --python 3.10` and then `examples/hanoi/requirements.txt`
(h5py 3.16, needed for the recording's Boolean attributes). Three things go
wrong easily:

- `examples/hanoi/env.sh` exports `COSMOS_POLICY_PLATFORM=hanoi`. Source it
  first, then `export COSMOS_POLICY_PLATFORM=hanoi_joint`, before any
  `cosmos_policy` import. The other order fails with a platform-mismatch error.
- Run Python from the repository root. The inference config's `config_file` is
  the relative path `cosmos_policy/config/hanoi_waypoint_config.py`.
- The image must be the contract crop (section 4) in RGB order. Nothing in the
  loader can detect a BGR frame or a shifted crop; only the replay in section 6
  does.

## 3. Loading and calling the policy

```python
import os
os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_joint'
from cosmos_policy.experiments.robot.hanoi.waypoint_policy import (
    HanoiWaypointInferenceConfig, HanoiWaypointPolicy)

cfg = HanoiWaypointInferenceConfig(
    '<run>/exports/iter_XXXXXXXXX.pt',
    'data/hanoi_cosmos/waypoint_v4/dataset_statistics.json',
    'data/hanoi_cosmos/t5_embeddings.pkl')
policy = HanoiWaypointPolicy(cfg)          # loads to CUDA, validates the contract

result = policy.infer({
    'observation/image': rgb224,             # (224, 224, 3) uint8, the contract crop
    'observation/state': state7,             # six joint angles (rad) + jaw stroke (m)
    'observation/cartesian_position': xyz,   # measured XYZ (m), same SDK snapshot
}, seed=1)
result['actions']       # (8, 4): absolute destination XYZ in metres + jaw intent 0/1
result['commit_count']  # 1: execute row 0 only, then re-observe
```

Inference is five denoising steps and takes well under a second on an H100.
Measured on an RTX 5080 (sm_120, 16 GB): 0.66 s per query after the first
warm-up call, 7.5 GiB peak allocated, Transformer Engine's fused RoPE kernel
running without fallback. With a fixed seed the output is deterministic for a
given observation on a given GPU; the offline evaluator uses seed 1 and the
parity check asserts equality. Across GPUs the outputs are close but not
bit-identical (different attention and RoPE kernels): the same six episode-40
observations scored within 0.06 mm of the H200 evaluation on the 5080, with
identical jaw intents.

## 4. Observation contract

| Field | Requirement |
|---|---|
| Image | The `/camera/camera/color/image_raw` frame, cropped with `rgb_crop_xywh` [151, 90, 360, 360], resized to 224 x 224, RGB uint8. Same crop as the recording; no second crop, no augmentation |
| Image age | At most 50 ms between image receipt and the command that consumes it |
| State | `joint_0_rad` to `joint_5_rad` in Trossen driver order 0 to 5, then `jaw_stroke_m`. Measured values, no velocities, no commanded jaw |
| Cartesian position | Measured tool XYZ in the commissioned base-tool frame, from the same snapshot as the state. Used only to convert the relative prediction back to absolute; never a model input |
| Prompt | Fixed: "Move all four rings from peg A to peg C following Tower of Hanoi rules." Only this direction was trained |

## 5. Action and timing contract

| Item | Value |
|---|---|
| Output | eight destinations, prefix 1 executed |
| Destination | absolute XYZ (m) in the base-tool frame; orientation fixed at RPY [0, pi/4, 0] |
| Jaw intent | 1 = open after arrival, 0 = closed after arrival. Apply only when it differs from the current state |
| Jaw open | stroke 0.034 m over 1.0 s |
| Jaw close | effort -20 N over 1.2 s, then 0.2 s settle |
| Loop | observe, infer, move to row 0, apply jaw change, wait for completion, take a fresh observation, repeat |
| Termination | external goal confirmation; the policy has no stop signal |
| Reference rate | none; the loop is completion-driven |

Training observations were recorded while the arm was moving toward a
reference, and deployment observes after stopping. This approximation is
documented in both label sets and unmeasured on hardware.

## 6. Acceptance work before the arm moves

1. **Parity on the deployment GPU.** `run_hanoi_physical_eval.py` cannot run
   there: it refuses any GPU but H100/H200, and the dataset class checks the raw
   recording's path, size and mtime. Use the deployment-machine variant, which
   takes the same `load_policy` and `get_action` path and needs only the
   single-episode extract:
   ```
   source examples/hanoi/env.sh && export COSMOS_POLICY_PLATFORM=hanoi_joint
   .venv/bin/python examples/hanoi/dream_local.py --samples 6
   ```
   Compare its per-sample `first_target_xyz_mm` with the same rows in
   `<run>/selected_validation.json` (episode 40 starts at raw row 288040).
   Done on the RTX 5080 for `iter_000008000.pt`: six of six within 0.06 mm,
   jaw intents identical, all six first targets within 5 mm.
2. **Capture-path replay.** Feed recorded validation observations through the
   robot-side capture, crop and state code, then through `infer`, and compare
   with offline predictions for the same rows. The evaluators store per-sample
   errors, not predictions; the reference predictions are the
   `predicted_targets_abs` entries in `dream_local.py`'s `report.json` (use
   `--samples 90` for every validation row of the extract). Catches BGR/RGB,
   crop offset, unit and joint-order mistakes before any motion.
3. **Static prediction check.** From the robot's actual start pose and a real
   camera frame, print the predicted first destination and jaw intent and check
   by eye that it is the expected first ring pickup.
4. **Supervised motion** at reduced speed with the stop within reach: one ring
   transfer, then one full episode. Record placement error at every stop.
5. Only then compare against tolerance and decide whether the stopped-observation
   shift needs a short on-robot fine-tune (the v4 importer accepts a new audit
   file without code changes).

## 7. What is known and what is not

Known (offline, waypoint_v4, selected export `iter_000008000.pt`, run completed
September 18, 2026):

| Split | n | Hit rate (5 mm, correct jaw) | Mean first-target error | p95 | Worst |
|---|---|---|---|---|---|
| validation | 445 | 100% | 0.76 mm | 1.76 mm | 2.79 mm |
| test | 450 | 100% | 0.76 mm | 1.85 mm | 3.05 mm |

Jaw intent is 100% correct on both splits, first target and all eight slots.
Serving parity passed on both. Per-export numbers for all eight exports are in
`<run>/evaluation/`; see `docs/hanoi_cosmos_waypoint_v4.md`.

Not known: anything on hardware; behaviour when the arm is stopped at
observation time; behaviour if the board state is not one seen in the 50
recorded episodes; recovery after a failed grasp.

## 8. World-model head check

`examples/hanoi/predict_future_images.py` decodes the model's predicted future
frame for held-out examples and saves strips [current | predicted | recorded |
difference] with L1/PSNR against the recorded frame and the copy-current
baseline. Batch script: `examples/hanoi/predict_future_images.sbatch`. It is a
qualitative sanity check of the auxiliary head, not a task metric.

`examples/hanoi/dream_local.py` is the same check for a deployment machine: any
CUDA GPU, the single-episode extract instead of the raw recording, and it also
decodes the full 25-frame latent video the model emits (`*_dream.gif`). Only
frames 5 to 8 (current image) and 17 to 20 (predicted future) carry pictures;
the other slots hold the injected joint state, waypoints and value and decode
to black by design. On the 5080, `iter_000008000.pt` predicts the frame about
21 s ahead at 2.5 mean absolute error versus 19 for copying the current frame.
