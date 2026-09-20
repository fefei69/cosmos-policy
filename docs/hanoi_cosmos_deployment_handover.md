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
| `cosmos_policy/experiments/robot/hanoi/serve_waypoint.py` | Policy server: wraps `HanoiWaypointPolicy.infer` behind the OpenPI WebSocket protocol |
| OpenPI `examples/hanoi/deployment/cosmos_client.py` | The robot-side client (camera, arm, watchdogs, recovery) that calls the server |
| OpenPI `docs/hanoi_deployment_handoff.md` | The robot-side executor contract this policy plugs into |

Do not modify `waypoint_policy.py`, `joint_policy.py`, `cosmos_utils.py` or the
dataset modules on the hardware branch. The serving-parity check compares the
adapter with offline inference; changing them silently invalidates that check.
Robot I/O lives in the OpenPI checkout as `examples/hanoi/deployment/cosmos_client.py`
(launcher `run_cosmos_client.sh`), which cannot import this package: the ROS and
Trossen environment is Python 3.12 and this one is 3.10. It talks to
`serve_waypoint.py`, which calls `HanoiWaypointPolicy.infer` and nothing else.

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
identical jaw intents. Two processes on the same 5080 differ by up to 0.2 mm
(bf16 variation); within one process the output is bit-identical.

### Serving

```bash
source examples/hanoi/env.sh && export COSMOS_POLICY_PLATFORM=hanoi_joint
.venv/bin/python -m cosmos_policy.experiments.robot.hanoi.serve_waypoint --port 8001
```

The server loads the export, hashes it, warms the model once, then serves. On
connect it sends `{"cosmos_hanoi": {contract, export_sha256, statistics_sha256,
seed, num_denoising_steps, commit_count, gpu}}`; the client refuses any other
contract or export. Each request carries the three observation keys above; each
reply is `{"actions": (8, 4), "commit_count": 1, "server_timing"}`. Requests
are validated before inference; a failure returns the traceback as text and
closes the connection. Port 8001 leaves the pi0.5 server on 8000 untouched.

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
   jaw intents identical, all six first targets within 5 mm. Wire parity is
   checked by replaying the same rows through the server from the robot
   environment (`./run_cosmos_client.sh --mode replay --replay-rows 0 1443 2880
   4258 5700 7138`, in the OpenPI checkout): identical across two client runs
   against one server, within 0.2 mm of the offline process.
2. **Capture-path replay.** Feed recorded validation observations through the
   robot-side capture, crop and state code, then through `infer`, and compare
   with offline predictions for the same rows. The evaluators store per-sample
   errors, not predictions; the reference predictions are the
   `predicted_targets_abs` entries in `dream_local.py`'s `report.json` (use
   `--samples 90` for every validation row of the extract). Catches BGR/RGB,
   crop offset, unit and joint-order mistakes before any motion.
3. **Static prediction check.** `./run_cosmos_client.sh --mode shadow` reads the
   real camera and arm, logs every prediction to `events.jsonl`, and never moves.
   From the rod-A start pose, check by eye that row 0 is the expected first ring
   pickup with jaw intent 1.
4. **Supervised motion** at reduced speed with the stop within reach:
   `./run_cosmos_client.sh --duration-s 90` for one ring transfer, then a longer
   run for one full episode. Moves are rest-to-rest at half speed; a jaw change
   waits for arrival; stroke at or below 8 mm during a closed grip stops the run
   and returns home; Ctrl-C holds and opens; when the duration ends the client
   finishes any placement in progress, opens, and returns to joint home. In-flight
   tracking tolerance is 8 mm (a loaded 105 mm lift lagged 3 mm within 0.6 s on
   the first live run), arrival 3 mm after a 0.5 s settle. Record placement
   error at every stop.
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

Hardware, September 18, 2026, two live runs of `iter_000008000.pt` through the
OpenPI `cosmos_client.py` on the real arm from the rod-A pose:

- Run 1 descended to the recorded grasp point, closed to a 14 mm grip on the
  ring (the pi0.5 runs never exceeded 9 mm), and was stopped by a 3 mm tracking
  watchdog 0.6 s into the loaded lift. The limit was inherited from pi0.5 and is
  now 8 mm in flight, 3 mm on arrival.
- Run 2 completed the first ring transfer, peg A to peg B, in one attempt:
  grip, lift (worst lag 2.6 mm), carry, descend, release, every arrival within
  0.5 mm. It was then stopped by the pi0.5-derived workspace ceiling, which the
  lift-away exceeded by 0.07 mm. The client now uses bounds derived from this
  policy's labels.
- Run 3 completed move 1 the same way, then missed the second grasp: hovers were
  predicted 1.5 to 3.6 mm high and the grasp 3.5 mm above the ring level (81.0
  versus 77.5 mm), a 16.9 mm edge grip that slipped on the lift. Run 4 started
  from a non-reset board and missed outright. Joint states at these decisions
  are within 0.02 rad of training, so this is not the stopped-observation
  shift. It is small per-step bias accumulating through the relative XYZ
  encoding: each destination is predicted relative to the measured position,
  so a hover that is 3 mm high yields a grasp that is 3 mm high. Grasp x is
  3.6 mm off in every run for the same reason: the shared rod-A start pose
  comes from the September 10 recording, and "descend straight down"
  inherits that offset.
- Re-analysed September 18 with two more raw runs (5 and 6), which reproduce
  runs 2 and 3 step for step. The x error is a start-pose mismatch, not
  drift: the rod-A hover (492.6, -56.2, 191.1) is a carry pose in
  waypoint_v4 (238 training windows begin there, none precedes a grasp),
  and the grasp hover over A is at x = 496.1 mm. All 40 v4 episodes start
  at (414.0, 15.8, 191.2) behind peg B, and from there replay predicts the
  hover over A within 0.5 mm of the column. The height error does compound,
  partially: regressing each step's error on the offset the arm started from
  gives a slope of about 0.6 in y and z, plus under 1 mm of upward bias per
  step, reaching +3.8 mm at the second grasp. The OpenPI client now starts
  at the recorded v4 start (`--start-xyz-m`) and checks the start joints
  against the recorded ones (`start_joints_verified`); the compounding part
  is still open and is what the fine-tune below addresses.
- Runs 7 to 9 (v4 start pose, dreams saved from run 8 on): four moves
  completed, then the same fifth grasp on peg C missed, within 2 mm of
  each other at every step. Offline the export is precise on the validation
  episode (under 1 mm at most steps, 2.5 mm at worst, jaw intent 90/90); its
  weakest step is the hover-B-to-grasp-hover-C transition of move 5, one of
  the rarest in the data (40 examples against 280 for A-side hovers). Live,
  the arm reaches that hover 3 to 4 mm off after releasing the big ring, and
  from an off-column hover the model anti-corrects: from 4.7, 5.5 and 6.6 mm
  off it descended 3.0, 3.1 and 3.7 mm further off and 3 to 6 mm too high.
  Training poses never leave their grid point by more than 1 mm, so this is
  coverage, not imbalance.
- Runs 10 to 12 tried executing three chunk waypoints per observation
  (`--commit-count 3`; the chunks are consistent, positions 1 to 7 as accurate
  as position 0). Worse: the open-loop waypoints carry the offset present at
  prediction time and each re-plan adds to it, so y drifted at 0.15 mm and z
  at 0.1 mm per waypoint (no trend with commit 1) and the third ring's grasp
  at move 4 failed 2.4 to 3.0 mm high; run 10 also missed the very first
  grasp, planned 1.6 mm high from the start pose. Re-observing after every
  waypoint corrects part of the offset at most steps; the anti-correction is
  specific to the peg C grasp approach. The client default is back to 1.
  Twelve runs of saved observations now exist for the fine-tune.
- Comparison baseline (September 18): a pi0.5 checkpoint trained on the same
  waypoint_v4 dataset (`pi05_hanoi_waypoint_aaaa_to_cccc`, export 29999,
  validation first-waypoint error 0.47 mm mean) is served by
  `examples/hanoi/deployment/serve_waypoint.py` in the OpenPI checkout on
  port 8000 and driven by the same client with the same start pose, joint
  check and commit-one execution (`--server ws://127.0.0.1:8000`). Its
  contract is identical to this policy's deployment contract; inference is
  about 0.1 s against 0.48 s here. Replay from the v4 start predicts the
  hover over A within 1.3 mm. Its first live run made 11 of 15 moves with
  three null moves (ring put back where it was taken); those come from a
  state shortcut on the recording's grasp/release column offset, shown by
  swapping joint states between observations. Details in the OpenPI README.
- This is the policy's real-world precision as it stands: about 1 mm offline,
  3 to 4 mm live, against a grasp tolerance of about 3 mm. The client executes
  the raw output by default. The server publishes the 18 recorded destinations
  and the client can snap onto them (`--snap-to-recorded-destinations`), but
  only as an ablation that separates "wrong point" from "imprecise point"; it
  replaces the model's output with task knowledge and must be off when the
  policy is being measured.
- The model-side fix is a short fine-tune on the deployment observations
  themselves: every live request's image and state are saved, the correct
  destination for each is the recorded grid point, and the v4 importer accepts
  a new audit file. That teaches the model to predict absolute destinations
  from a slightly displaced stopped arm instead of replaying the recorded
  displacement.

Not known: a full episode on hardware; how much height drift remains once
the chain starts from the recorded start pose; behaviour if the board state is not one
seen in the 50 recorded episodes; recovery after a failed grasp.

Dreams from deployment runs: the server returns only the destinations unless
started with `--dream`, which adds the predicted future frame and value to
every reply (one VAE decode; actions unchanged) so the client saves them under
`inference_dreams/`. For a run recorded without it,
`examples/hanoi/dream_run.py --run-dir <openpi run dir>` re-runs the policy on
the saved `inference_inputs/` and writes strips, a contact sheet and a report
under `<run dir>/dreams/`; offline and live decisions agree to about 0.2 mm.

Dense contract five: the OpenPI client `examples/hanoi/deployment/dense_client.py`
executes 10 Hz reference chunks in velocity-continuous 0.3 s segments
(see the OpenPI README, "Dense contract-five client"). A Cosmos dense
checkpoint served with the same reply shape (`actions` (16, 4), `reference_rate_hz`
10, `execution_prefix` 8, identity under `hanoi_dense`) can use it after the
client's horizon and prefix are read from the contract instead of fixed.

Result to beat (September 20): the pi0.5 dense contract-five policy solved the
full puzzle on the arm, 15 legal moves in 270 s from the hover over peg A, raw
output, no interventions (OpenPI `dense_live_1789936407839752778`). A Cosmos
dense checkpoint on the same recording is the direct comparison.

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
