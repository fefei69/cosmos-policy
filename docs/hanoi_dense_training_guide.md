# Hanoi dense retraining guide (contract `hanoi_dense_v5`)

Written September 19, 2026, for the cluster-side agent training in `/scratch/cw5167/workspace/openpi`
(pi0.5) and `/scratch/cw5167/workspace/cosmos-policy` (Cosmos Policy). Both pipelines consume the same
recording and must produce checkpoints the existing deployment client can drive. Read the whole guide
before changing code; the decisions table is the only part the user expects to revise.

## 1. Why this run exists

The waypoint_v4 checkpoints (pi0.5 export 29999, Cosmos iter 8000) were trained on 3,589 curated decision
windows from 40 scripted episodes. Live on the arm (September 18, twelve Cosmos runs and one pi0.5 run,
all with raw execution and live observations only) they showed two failure modes that offline validation
did not predict:

- Cosmos: from a hover a few millimetres off a recorded column it moves further off instead of back
  (4.7, 5.5 and 6.6 mm off the peg C column became 7.7, 8.6 and 10.3 mm at the grasp, 3 to 6 mm too
  high). Every training pose sits within 1 mm of one of 18 points, so correction was never demonstrated.
- pi0.5: eleven of fifteen moves, then null moves (a ring put back where it was taken). Swapping the
  joint state between two observations swaps the decision with the images unchanged; the recording's
  grasp and release hover columns differ by a few millimetres per peg, so the discretised state leaks the
  phase.

Both are consequences of training on a tiny set of near-identical states. This run trains on the raw
30 Hz recording instead, every row an observation, with a chunk of future reference poses as the label.
No curated windows, no DAgger, no perturbation collection, no synthetic data.

## 2. Decisions (defaults the user has accepted unless this table is edited)

| # | Decision | Default | Alternatives considered |
|---|---|---|---|
| 1 | Action rate | 10 Hz (frameskip 3 over 30 Hz rows) | 15 Hz (skip 2) or 6 Hz (skip 5) |
| 2 | Chunk length | pi0.5: 30 steps (3.0 s). Cosmos: 16 steps (1.6 s) | Cosmos 32 steps |
| 3 | Action space | absolute base-frame XYZ of the commanded reference pose, plus jaw intent 0/1 | relative to measured XYZ (the v3/v4 encoding) |
| 4 | State | six joint angles plus measured jaw stroke, no velocity | add Cartesian XYZ |
| 5 | pi0.5 state input | continuous (`discrete_state_input=False`) | discretised into tokens, as in v4 |
| 6 | Direction | AAAA to CCCC only | both directions as a multitask run |
| 7 | Cosmos initial weights | the Cosmos-Predict2 2B video base (run B); the LIBERO policy checkpoint used by v4 only as a comparison run A if a second GPU is free | LIBERO init only |
| 8 | Budget | pi0.5 30,000 updates at batch 32 (about 2.8 epochs). Cosmos 16,000 updates at effective batch 32 (about 1.5 epochs) | shorter Cosmos budget as in v4 (8,000) |
| 9 | Episode split | the waypoint_v4 episode split, unchanged, so results are comparable | 45 train / 5 validation, no test |
| 10 | Image augmentation | none | mild colour jitter |
| 11 | Selection rule | lowest mean per-step XYZ error on validation with jaw accuracy at least 0.99; ties by earlier step | validation flow loss |
| 12 | Execution prefix at deployment | pi0.5 3 steps (0.3 s), Cosmos 8 steps (0.8 s) | longer prefixes |

Decision 7: the user wants the video base. The action, proprio and value slots then start untrained and
must be learned from 50 episodes (about 3.3 hours of video, 340k observations), which the LIBERO
checkpoint had already learned on other data. Watch for that: if the video-init run's validation slot-1
error is not below 5 mm by 6,000 updates, launch run A (LIBERO init) as well and report both. Run A on a
second GPU from the start if one is available.

Decision 5 explained: pi0.5 can feed the proprioceptive state either as a continuous vector into the
action expert, or discretised into bins and given to the language model as tokens. v4 used the tokens.
Tokens make the policy's output a step function of the state, which is how the phase shortcut arose: the
decision flipped within 0.2 mm of hover position. Continuous is the default here.

## 3. Source data

Raw forward recording (50 successful episodes, 360,050 rows at 30 Hz):
`/scratch/cw5167/datasets/hanoi_wm_roundtrip_20260915_223424_AAAA_to_CCCC.h5` with its matching JSON.
The single-episode extract used locally (`exports_local/hanoi_episode_040.h5`) has the same layout.

Fields to use, per row:

| Field | Shape | Use |
|---|---|---|
| `pixels` | (224, 224, 3) uint8 | observation image, already the deployment crop (`[151, 90, 360, 360]` of 640x480, INTER_AREA to 224); never crop again |
| `joint_positions` | (6,) | state[0:6], driver arm indices 0 to 5, radians |
| `proprio` | (8,) | `[x, y, z, vx, vy, vz, jaw_stroke_m, ...]`; state[6] = column 6; columns 0:3 are the measured XYZ for audits only, not a model input |
| `reference_pose` | (6,) | commanded XYZ (0:3) at that row; the XYZ label source |
| `action_abs` | (4,) | `[x, y, z, jaw_intent]` for the next tick; jaw intent 0/1 is the label source for the gripper column |
| `image_stale`, `image_repeated` | bool | rows to exclude as observations (about 2.4% and 3.9% of rows) |
| `episode_success`, `ep_offset`, `ep_len` | | episode bounds; use successful episodes only |
| `labels/joint_v3`, `labels/waypoint_v4` | groups | previous contracts; do not use |

Before building anything, audit and report: (a) `action_abs[t, :3]` against `reference_pose[t+1, :3]`
(expected within about 1 mm; explain any larger residual); (b) the jaw intent column flips exactly at
gripper command rows (`gripper_command_issued`); (c) the fraction of stationary rows (measured speed
under 2 mm/s; about 17% in episode 40); (d) stale and repeated image counts per episode.

## 4. Dataset construction (shared by both pipelines)

For every row `t` of every training episode that is not stale or repeated:

- observation: `pixels[t]`, state `[joint_positions[t], proprio[t, 6]]`, prompt
  "Move all four rings from peg A to peg C following Tower of Hanoi rules."
- actions: for `j = 1..H`, row `r_j = t + 3 j`; XYZ = `reference_pose[r_j, 0:3]`, jaw = `action_abs[r_j, 3]`.
  Rows past the episode end repeat the last row and are marked in `actions_is_pad`.
- `H` = 30 for pi0.5, 16 for Cosmos. Produce both archives from one builder.

Do not subsample stationary rows and do not oversample anything; the row distribution is the data.
Use random frameskip phase only in the sense that every row is an observation, so all three phases of
the skip occur naturally. Splits are by whole episode (decision 9). Write for each split an archive with
`images` (or row indices into the raw file), `states`, `actions`, `actions_is_pad`, `episode_indices`,
`source_observation_indices`, `source_action_indices`, plus `metadata.json` carrying the contract below,
the raw file SHA-256, the split, counts, and `dataset_statistics.json` (normalisation over the training
split: state min/max/mean/std, action min/max/mean/std over all valid chunk slots).

Contract dict published by both servers (the deployment client compares these fields):

```
version: 5
robot: trossen_wxai_single
reference_rate_hz: 10
action_horizon: 30 (pi0.5) or 16 (Cosmos)
execution_prefix: 3 (pi0.5) or 8 (Cosmos)
state: [joint_0_rad .. joint_5_rad, jaw_stroke_m]
actions: [reference_x_m, reference_y_m, reference_z_m, jaw_open_intent]
internal_xyz_encoding: absolute
frame: commissioned_base_tool_frame
orientation_rpy_rad: [0, pi/4, 0]
rgb_topic, rgb_crop_xywh [151, 90, 360, 360], max_image_age_s 0.05
jaw_open_stroke_m 0.034, jaw_open_duration_s 1.0, jaw_close_effort_n -20, jaw_close_duration_s 1.2, jaw_close_settle_s 0.2
training_observation_alignment: every non-stale row; chunk starts three rows after the observation
training_deployment_timing: moving observations in training; asynchronous chunk execution at deployment
recording: hanoi_wm_roundtrip_20260915_223424
```

Keep the recorded 1.2 s close in the contract; the live driver uses 2.4 s and the timeline client already
skips elapsed references during the longer dwell.

## 5. pi0.5 pipeline (`/scratch/cw5167/workspace/openpi`)

- New policy module `hanoi_dense_policy.py` next to `hanoi_joint_policy.py`: `HORIZON = 30`, the contract
  above, an inputs transform that takes `observation/image`, `observation/state` (7,), `prompt`, and
  `actions` (30, 4) absolute; no Cartesian context and no relative conversion (decision 3). Both wrist
  slots masked as before.
- Data config: a `LeRobotHanoiDenseDataConfig` that repacks those keys and reads the archive. Because
  every row is an observation the dataset is about 340k windows; build it as a LeRobot dataset with the
  chunk indices in the archive rather than materialising 340k image copies (images are shared with the
  raw file; index them).
- Model: `Pi0Config(pi05=True, action_horizon=30, discrete_state_input=False)` (decision 5). Weight loader
  `gs://openpi-assets/checkpoints/pi05_base/params`. Optimiser and schedule as v4: cosine, warmup 1,000,
  peak 2.5e-5, decay to 2.5e-6 over the run. Batch 32, 30,000 updates, EMA on, export every 2,000, keep
  every export (decision 8).
- Config name `pi05_hanoi_dense_aaaa_to_cccc`, assets under `data/hanoi/dense_v5/assets`, run directory
  `checkpoints/pi05_hanoi_dense_aaaa_to_cccc/hanoi_dense_<date>/`, exports under `exports/<step>/` with
  `export.json`, params and `assets/local/hanoi_dense_v5/norm_stats.json`, exactly like v4.
- Serving: `policy_output_context_keys` is no longer needed. The serve wrapper must return
  `{"actions": (30, 4) float32 absolute, "reference_rate_hz": 10, "execution_prefix": 3}` with jaw
  thresholded at 0.5, and metadata `hanoi_dense` carrying the contract, prompt, `export_sha256` of
  `export.json`, `normalization_sha256`, `num_steps` (10), `config_name`, `checkpoint`.

## 6. Cosmos pipeline (`/scratch/cw5167/workspace/cosmos-policy`)

- New platform `hanoi_dense` in `cosmos_policy/constants.py`: `NUM_ACTIONS_CHUNK = 16`, `ACTION_DIM = 4`,
  `PROPRIO_DIM = 7`. Selected with `COSMOS_POLICY_PLATFORM=hanoi_dense` before model imports; keep
  `hanoi_joint` intact.
- Dataset module `hanoi_dense_data.py` / `hanoi_dense_dataset.py` after the waypoint pair: same latent-slot
  packing (blank, proprio, RGB, actions, future proprio, future RGB, value), min/max normalisation from
  `dataset_statistics.json`, no clipping. The auxiliary future frame and future state are the raw row at
  the end of the chunk, `t + 3 * 16 = t + 48` rows (1.6 s), not verified arrival, not a conditioning
  input. The value label keeps the v4 definition (discounted steps to episode end).
- Config `hanoi_dense_config.py` after `hanoi_waypoint_config.py`: effective batch 32 (micro 16), the v4
  schedule shape, `max_iter` 16,000, save every 1,000, one H100 or H200 (decision 8).
- Initial weights: run B, the primary, from the Cosmos-Predict2 2B video base that the LIBERO policy was
  itself fine-tuned from (decision 7); the newly added slots (actions, proprio, future proprio, value)
  are initialised the way the Cosmos Policy release initialises them from that base. Run A, the
  comparison, from `checkpoints/public/Cosmos-Policy-LIBERO-Predict2-2B.pt` as in v4. Record the
  SHA-256 of the initial weights in `joint_contract.json` as v4 does. Run names
  `hanoi_cosmos_dense_<date>_video_init` and `hanoi_cosmos_dense_<date>_libero_init`.
- Export every saved iteration with the existing exporter; `joint_contract.json` gets contract
  `hanoi_dense_v5_cosmos_v1`, the statistics hash, the raw hash, batch, updates and the selection rule.
- Serving: the waypoint server module is the template; the reply becomes
  `{"actions": (16, 4) float32 absolute, "reference_rate_hz": 10, "execution_prefix": 8}` with the
  identity under `cosmos_hanoi` extended by `reference_rate_hz` and `execution_prefix`. Inference at 5
  denoising steps as now; also report validation metrics at 10 steps so the deployment side can choose.

## 7. Validation protocol (both pipelines, every export, EMA weights)

Compute on the validation episodes, one observation per row (or every third row if time-bound, phase
randomised), never on test until selection is locked:

1. Per-step XYZ error along the chunk (mm): mean, p95, and per-slot means. Slot 1 is the deployment-critical one.
2. Chunk endpoint XYZ error (last valid slot).
3. Jaw intent accuracy per slot, balanced accuracy, and event timing error in rows for each flip.
4. All of the above split by observation motion: stationary rows (measured speed under 2 mm/s) versus
   moving rows. Report both; the stationary subset is what the arm sees while it waits for inference.
5. Cosmos only: future-frame L1 and PSNR against the real frame 48 rows later, and value error.

Selection: decision 11. After selection, evaluate the chosen export once on the test split and report it
separately. Also run the native serving path (the server module, not the training evaluator) on 200
validation observations and confirm it reproduces the evaluator's first-slot outputs within 0.5 mm; v4
called this the serving-parity check.

Ready-to-deploy means: slot 1 mean error under 2 mm on both motion subsets, jaw accuracy at least 0.99,
serving parity passed, `export.json` and normalisation hashes recorded.

## 8. Deliverables

- Archives and `metadata.json`, `dataset_statistics.json` under `data/hanoi/dense_v5/` (openpi) and
  `data/hanoi_cosmos/dense_v5/` (cosmos), with the audit report from section 3.
- Selected exports with identity files, plus a transfer list per repository in the same form as
  `waypoint_v4_transfer.txt` (paths relative to the repository root, exports and assets only, no
  intermediate checkpoints). Total size matters: the pi0.5 export is about 12 GB.
- A results note per pipeline: the validation tables of section 7 for every export, the selection, the
  test result, the serving-parity result, and anything that deviated from this guide. Update the
  respective handover document. Do not state or imply live success; that is measured on the arm.
- Wandb URLs as in previous runs.

## 9. Do not

- Curate, simplify, or re-label rows; do not drop stationary rows; do not add synthetic or perturbed data.
- Put velocity in the state, or the measured XYZ, or any board or held-disk label.
- Re-crop the stored images or add augmentation beyond decision 10.
- Consult the test split before selection is locked.
- Change the prompt, the crop, the orientation, or the jaw timings in the contract.
- Reuse the `hanoi_joint` platform constants for the Cosmos dense run.

## 10. Scheduling

Follow `docs/hanoi_training_plan.md` for resource discovery, submission, monitoring and preemption
policy. Two Cosmos runs and one pi0.5 run can proceed in parallel on separate GPUs; the dataset build is
shared and must finish first. Report the dataset audit before launching training.
