# Hanoi Cosmos policy: deployment handover

Written September 17, 2026, for the agent working on the `hardware` branch.
Everything here describes what the trained policy expects and produces. Nothing
about robot task success has been measured; the acceptance work in section 6 is
what establishes that.

## 1. Read these first

| File | Why |
|---|---|
| `docs/hanoi_cosmos_waypoint_v4.md` | The run that produced the deployable checkpoint, its selection rule, caveats |
| `docs/hanoi_cosmos_joint_sparse.md` | The observation/target contract shared by v3 and v4 |
| `cosmos_policy/experiments/robot/hanoi/waypoint_policy.py` | Loader and serving adapter to call |
| `cosmos_policy/experiments/robot/hanoi/run_hanoi_physical_eval.py` | Offline evaluation and the serving-parity check to reproduce on the robot machine |
| `data/hanoi_cosmos/waypoint_v4/metadata.json` | The full contract under `deployment` (crop, frame, jaw parameters, timing) |
| OpenPI `docs/hanoi_deployment_handoff.md` | The robot-side executor this policy is meant to plug into |

Do not modify `waypoint_policy.py`, `joint_policy.py`, `cosmos_utils.py` or the
dataset modules on the hardware branch. The serving-parity check compares the
adapter with offline inference; changing them silently invalidates that check.
Put robot I/O in a new module that calls `HanoiWaypointPolicy.infer`.

## 2. Artifacts

Available when run `hanoi_cosmos_waypoint_v4_20260917` finishes (about 01:15 on
September 18), or earlier from any export listed in
`<run>/evaluation/iter_*_validation_actions.json`.

| Artifact | Path on the cluster |
|---|---|
| Run directory | `data/hanoi_cosmos/runs/cosmos_policy/hanoi/hanoi_cosmos_waypoint_v4_20260917/` |
| Selected export | `selection.json` names it; exports are `exports/iter_*.pt` (3.9 GB each) |
| Contract identity | `joint_contract.json` in the run directory; the loader reads it from two levels above the export, so **copy the run directory layout**, not the `.pt` alone |
| Normalization | `data/hanoi_cosmos/waypoint_v4/dataset_statistics.json` (hash is checked against the contract) |
| Prompt embedding | `data/hanoi_cosmos/t5_embeddings.pkl` (no T5 encoder is needed at runtime) |
| VAE | `checkpoints/public/Wan2.1_VAE.pth` |
| Base checkpoint | not needed at runtime |

Minimum copy for a deployment machine:

```
<run>/joint_contract.json
<run>/exports/iter_XXXXXXXXX.pt
data/hanoi_cosmos/waypoint_v4/{metadata.json,dataset_statistics.json}
data/hanoi_cosmos/t5_embeddings.pkl
checkpoints/public/Wan2.1_VAE.pth
```

Runtime: one CUDA GPU with about 10 GB free. Only the evaluation scripts insist
on H100/H200; the loader does not. Set `COSMOS_POLICY_PLATFORM=hanoi_joint`
before importing anything from `cosmos_policy`, and source `examples/hanoi/env.sh`
for the cache and NVRTC settings.

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
With a fixed seed the output is deterministic for a given observation; the
offline evaluator uses seed 1 and the parity check asserts equality.

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

1. **Parity on the deployment GPU.** Run the evaluator there in `actions`
   mode on a handful of validation examples; it asserts served == offline.
   ```
   python -m cosmos_policy.experiments.robot.hanoi.run_hanoi_physical_eval \
     --checkpoint <run>/exports/iter_XXXXXXXXX.pt --metadata data/hanoi_cosmos/waypoint_v4 \
     --mode actions --samples 8 --output /tmp/parity.json
   ```
2. **Capture-path replay.** Feed recorded validation observations through the
   robot-side capture, crop and state code, then through `infer`, and compare
   with the offline predictions for the same rows. Catches BGR/RGB, crop offset,
   unit and joint-order mistakes before any motion.
3. **Static prediction check.** From the robot's actual start pose and a real
   camera frame, print the predicted first destination and jaw intent and check
   by eye that it is the expected first ring pickup.
4. **Supervised motion** at reduced speed with the stop within reach: one ring
   transfer, then one full episode. Record placement error at every stop.
5. Only then compare against tolerance and decide whether the stopped-observation
   shift needs a short on-robot fine-tune (the v4 importer accepts a new audit
   file without code changes).

## 7. What is known and what is not

Known (offline, 445 held-out validation examples, waypoint_v4):

| Export | Hit rate (5 mm, correct jaw) | Mean first-target error | p95 |
|---|---|---|---|
| step 1,000 | 68.5% | 4.6 mm | 9.9 mm |
| step 2,000 | 98.7% | 1.5 mm | 3.4 mm |

Later exports are appended to `<run>/evaluation/` as the run progresses.

Not known: anything on hardware; behaviour when the arm is stopped at
observation time; behaviour if the board state is not one seen in the 50
recorded episodes; recovery after a failed grasp.

## 8. World-model head check

`examples/hanoi/predict_future_images.py` decodes the model's predicted future
frame for held-out examples and saves strips [current | predicted | recorded |
difference] with L1/PSNR against the recorded frame and the copy-current
baseline. Batch script: `examples/hanoi/predict_future_images.sbatch`. It is a
qualitative sanity check of the auxiliary head, not a task metric.
