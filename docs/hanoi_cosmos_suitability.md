# Suitability of the selected Hanoi demonstrations

Assessment: suitable for a bounded Cosmos Policy imitation-learning pilot on
the fixed AAAA → CCCC task. Neither one camera nor Cartesian actions prevents
training. This does not establish reliable robot control, convergence in two
hours, or transfer to unfamiliar Hanoi configurations.

## What the recordings contain

The robot was commanded through Cartesian motion segments. The saved
`action_abs` labels are dense nominal references along those segments, not
only segment endpoints and not measured end-effector positions. The JSON
records `segmented_quintic_velocity_noise_v1`, seven nine-tick segments per
motion leg, a 30 Hz reference rate, and 0.3-second segments. Preserve these
references and their intended noise.

Direct read-only checks of episodes 0, 24, and 49 found 7,201 rows over about
240 seconds each, median timestamp intervals of 33.32 ms, and XYZ changes
larger than one micrometre on 5,670 of 7,200 adjacent transitions in each
episode. Thus the supervised sequence is dense even though the controller
received sparse motion commands. The machine-readable evidence and raw image
probes are in `data/hanoi_cosmos/aaaa_to_cccc/suitability_data_audit.json`.

Use `action_abs[t:t+63]` with no additional shift; subtract the same measured
XYZ anchor from every target in the chunk. Keep jaw intent absolute. The
63-reference horizon spans 2.1 seconds. Observe only measured XYZ/jaw (`proprio[[0,1,2,6]]`) and the
single stored RGB image; commanded jaw and collector annotations are excluded.

## Compatibility with Cosmos Policy

The upstream `LIBERODataset` has separate `use_wrist_images` and
`use_third_person_images` flags and conditionally constructs the corresponding
latent frames. `replace_latent_with_action_chunk` in
`cosmos_policy/models/policy_text2world_model.py` accepts an arbitrary action
tensor, flattens it, and repeats it within a latent frame. At 224x224, one frame
has 16x28x28 = 12,544 elements; the proposed 63x4 chunk uses 252 elements.
The model therefore does not require joint-space actions, seven action
dimensions, or a wrist camera. The four measured state values likewise fit
the generic proprioception injection.

The Hanoi adapter supplies seven slots: blank, current measured state, current
RGB, actions, future measured state, future RGB, and value. Absent camera
indices are -1. Training and inference must use this same layout. This is a
custom adaptation of the released LIBERO policy, not an unchanged published
single-camera Hanoi configuration. CPU tests cover packing, normalization,
target alignment, episode boundaries, and absent cameras; an actual GPU
forward/backward and prediction remain necessary.

The [Cosmos Policy paper, §4.1 and Appendix A.1](https://arxiv.org/html/2601.16163v1)
describes the generic modality injection and action-chunk extraction that
support this adaptation. Its published benchmark results do not directly
validate this Hanoi setup.

## What the pilot can and cannot show

- All 50 sidecar episodes are successful and use one identical 15-move route,
  with 50 motion seeds. This supports learning the demonstrated task; it does
  not supply broad board-state coverage or failed-grasp recovery experience.
  Split whole episodes 40/5/5 rather than adjacent frames. The 360,050 rows are
  correlated samples from 50 demonstrations, not independent task executions.
- A fixed camera can show the board and arm, but fine ring/peg alignment and
  grasp state may be occluded. Measured XYZ and jaw stroke help;
  they do not replace a missing view of the ring. Offline imitation errors
  cannot establish successful four-minute physical rollouts.
  Raw episode-0 images at rows 0, 156, 201, and 7200 show all three pegs and the
  colored rings; the fingers partially cover the top ring during grasping.
  These four probes do not establish visibility throughout all demonstrations.
- Dense targets describe the collector's segment trajectory. Serving must
  preserve controller interpolation, fixed tool orientation, gripper dwell,
  and motion continuity. Sending each reference as a fresh stop/start waypoint
  could produce different dynamics. Robot execution is outside this pilot.
- The proposed nine-reference execution interval is 0.3 seconds. This is an
  unverified latency target, especially for an RTX 5080. The paper's Appendix
  A.4.2 reports 0.61 seconds per chunk for five-step inference on one H100 in
  its LIBERO/RoboCasa settings. Our smaller camera layout may differ; measure
  synchronized latency and its tail before choosing a serving schedule.
- Successful demonstrations support the initial policy objective and auxiliary
  future-state targets. They do not establish a calibrated failure-aware
  planner. The paper refines planning with policy rollout outcomes (§4.3).

The two-hour, single-H100 run is a feasibility and early-learning measurement:
test actual memory fit, checkpoint/resume, held-out XYZ/jaw predictions, and
inference latency. Its duration is not a claim that this policy will converge.
