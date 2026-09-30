# Related work for the play-data composition comparison

Written September 30, 2026, by the Cosmos-side agent, in answer to: "has any
paper made a similar comparison to highlight the composition ability of world
models?" The comparison in question: the CIDM goal-image world model with
graph-distance planning versus a behaviour-cloned VLA (pi0.5, Cosmos Policy),
both trained on the same Hanoi play recording (`hanoi_wm_20260924_210743.h5`,
125 coverage walks of 20 moves, about 12 h at 30 Hz; manifest
`/scratch/cw5167/datasets/dataset_manifest_v1/`), on the six 15-move tower
tasks that never appear whole in that data.

Provenance: six web sweeps (planning-vs-imitation, stitching, play data,
video/world-model planners, Hanoi, counter-evidence), 34 unique papers read in
full and checked by a second agent against the paper text, then a synthesis
pass. 13 papers kept, 21 dropped (section 8). Values marked ≈ were read from
bar charts or plots; unmarked values come from tables. Workflow run
`wf_d37c9aa4-c57`, output `tasks/wa1tpdn4z.output` in the session scratchpad.

## 1. Answer

No published paper makes this comparison. The missing conjunction is: a real
robot, human-teleoperated play (not scripted random walks, not expert demos),
a pretrained VLA fine-tuned on that play, a learned visual world model with an
explicit planner trained on the identical data, target tasks whose absence
from the data is verified, and a composition depth of 15 primitive moves. Each
existing paper covers at most three or four of these:

| Paper | Real robot | Play data | VLA | World model + planner | Same data | Absence verified | Depth |
|---|---|---|---|---|---|---|---|
| PLDM (Sobal et al., NeurIPS 2025) | no | scripted random walks | no (small GCBC) | yes (latent dynamics + MPPI) | yes | yes (by construction) | ~90 to 200 steps of navigation |
| CompPlan (Farebrother et al., ICML 2026) | no | scripted cube play | no (GC-BC) | yes (jumpy successor models + sub-goal shooting) | yes | no | up to 4 cubes |
| World Action Planner (Zhang and Du, preprint 2026) | no | no (LIBERO-90 expert demos) | yes (pi0.5, cosmos-policy) | yes (video world model + VLM proposals) | yes | yes (LIBERO-90 only) | 2 sub-tasks |
| TVF (Wu et al., IROS 2022) | yes | no (expert demos) | no (Transporter BC) | yes (visual foresight + tree search) | yes | yes (new structures) | ~3 to 8 pick-and-places |
| TACO-RL (Rosete-Beas et al., CoRL 2022) | yes | yes (CALVIN teleop play) | no (Play-LMP) | no (model-free offline RL) | yes | no | 2 skills |
| OGBench (Park et al., ICLR 2025) | no | scripted play | no (MLP GCBC) | no (value-based RL) | yes | yes | up to 8 to 24 behaviours |
| Lorang et al. (CoRL 2025) | no | no (single-move demos) | no (diffusion policy) | symbolic model + classical planner | no | yes | 7 moves (3-disk Hanoi) |

## 2. Closest precedents, ranked

### 2.1 PLDM: Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models

Sobal et al., NeurIPS 2025, arXiv 2502.14819 (v4, October 29, 2025).

- Data: scripted, reward-free random walks that never demonstrate the
  evaluation tasks. Two-Rooms: von Mises random walks, 91-step episodes, 3M
  transitions, with variants of 16/32/64-step episodes and a variant with no
  door-crossing trajectory. Ant U-Maze: noisy directional policy, trajectories
  of 25 to 500 steps. Observations 64 x 64 top-down images.
- Model side: JEPA latent encoder plus latent dynamics model, MPPI planning
  in latent space toward the goal-image embedding, replanned every step.
- Imitation side: goal-conditioned behaviour cloning (OGBench implementation,
  small Impala ConvNet), same data. Other baselines are value or
  representation methods: GCIQL, HIQL, CRL, HILP.
- Stitching tests (Section 4.4): with 16-step episodes the ~90-step goal "is
  never observed. To succeed, methods must stitch together multiple offline
  trajectories."

| Setting | GCBC | PLDM | GCIQL | HILP | Source |
|---|---|---|---|---|---|
| Two-Rooms, full-coverage data | 86.0 | 97.8 | 98.0 | 100.0 | Table 2, exact, 3 seeds |
| Two-Rooms, no door-crossing trajectory | 8.4 | 34.4 | 99.6 | 100.0 | Table 2, exact (CRL 14.7, HIQL 26.3) |
| Two-Rooms, 16-step episodes, ~90-step goals | ≈39 | ≈87 | ≈97 | ≈100 | Fig. 4 centre; Table 8 Welch p = 1.5e-12 for PLDM > GCBC |
| Ant U-Maze, 25-step trajectories, ~200-step goals | 0 | ≈97 | ≈37 | ≈97 | Fig. 6 |

- Mechanism stated (Section 4.5): with short or random trajectories the
  sampled state-goal pairs are close together, so far-away goals become out
  of distribution for the goal-conditioned policy.
- Caveats: 2D navigation and a MuJoCo ant, no manipulation, no language, no
  VLA. PLDM is not the best stitcher in the paper: GCIQL and HILP stitch as
  well or better, and the abstract calls its stitching "comparable to leading
  model-free methods". The two verification passes disagree on the held-out
  layout figure (baselines 46 to 86% vs "roughly 0%"), so quote nothing from
  Figures 2 and 8. The arXiv HTML renders hidden phantom digits in Table 2
  ("118.4" is 8.4); check the PDF before quoting.
- Cite as: "Sobal et al. (2025) show that a latent-dynamics planner trained
  on random-walk data reaches goals that never occur inside any training
  episode (about 87% vs 39% for goal-conditioned BC when episodes are 16
  steps and goals about 90 steps away; about 97% vs 0% in Ant U-Maze), and
  attribute the BC failure to far-away goals being out of distribution for
  the goal-conditioned policy." Say "planning composes where behaviour
  cloning trained on the same data does not", never "planning beats
  model-free RL at stitching".

### 2.2 CompPlan: Compositional Planning with Jumpy World Models

Farebrother, Pirotta, Tirinzoni, Bellemare, Lazaric, Touati; ICML 2026;
arXiv 2602.19634.

- Data: OGBench cube-1..4 play (a scripted policy repeatedly picks a random
  block and places it somewhere random) and antmaze navigate. Low-dimensional
  state, simulation.
- Model side: policy- and horizon-conditioned generative successor models
  ("jumpy" world models, TD-flow plus a horizon-consistency loss) trained on
  the same offline data; random-shooting planner over sub-goal waypoints that
  sequences pre-trained goal-conditioned policies; replans every step.
- Imitation side: the same goal-conditioned BC (flow matching, hindsight
  relabelling) run zero-shot on the final goal.

| Domain | GC-BC zero-shot | GC-BC sequenced by CompPlan | Source |
|---|---|---|---|
| cube-1 | 0.90 | 0.99 | Table 1 |
| cube-2 | 0.15 | 0.97 | Table 1 |
| cube-3 | 0.09 | 0.92 | Table 1 |
| cube-4 | 0.00 | 0.76 | Table 1 |
| antmaze-medium / large / giant | 0.49 / 0.18 / 0.00 | 0.85 / 0.73 / 0.03 | Table 1 |
| cube-4 with HFBC base: HIQL / SHARSA / HFBC / CompPlan | 0.00 / 0.09 / 0.34 / 0.67 | | Table 2 |

- Mechanism stated (Appendix F.1): value-free GC-BC "struggles to generalize
  to distant goals"; the planner does not improve the policy, it conditions
  the same policy on nearby sub-goals proposed and scored by the world model.
- Caveats: simulation, state input, no pixels, language or VLA; BC is the
  primitive inside the planner, so this is not a standalone head-to-head; 5
  tasks x 10 rollouts x 3 seeds; HFBC zero-shot numbers differ between
  Tables 1 and 2 (name the table); composition needs a locally competent base
  policy (GC-BC on antmaze-giant stays at 0.03).
- Cite as: "Farebrother et al. (2026) show on OGBench cube play data that the
  same goal-conditioned BC policy that scores 0% on four-cube tasks reaches
  76% when a world model trained on the same data proposes and verifies
  sub-goals for it." Say "on the cube play domains". This is also the
  template for the control the Hanoi paper needs: the VLA fed the planner's
  next-board sub-goal.

### 2.3 World Action Planner: Generalizable Decision-Making with Action-Conditioned World Models

Zhang and Du, arXiv 2607.27599, submitted July 30, 2026, Harvard. Preprint,
not peer reviewed.

- Data: LIBERO-90 teleoperated expert demonstrations (the world model also
  sees Gaussian-noise-perturbed copies). Not play.
- Model side: a VLM (Gemini 3.0 Flash) decomposes the instruction and
  proposes bridging actions, refined by local search over imagined rollouts of
  an action-conditioned video world model (Wan-T2V-1.3B backbone, four
  views); a LIBERO-90 diffusion policy executes each atomic sub-task.
- Imitation side: pi0.5 fine-tuned by the authors on LIBERO-90 (95.8%
  in-domain) and cosmos-policy (93% in-domain), the same two VLAs as in the
  Hanoi comparison.
- Test: four LIBERO-Long tasks, each the concatenation of two LIBERO-90
  tasks, all models trained on LIBERO-90 only, 50 trials per task.

| Method | Task 1 / 2 / 3 / 4 success % | Source |
|---|---|---|
| pi0.5 | 4 / 0 / 0 / 0 | Table 3 |
| cosmos-policy | 0 / 0 / 0 / 0 | Table 3 |
| SAILOR (same world model, policy-sampled actions) | 18 / 0 / 8 / 2 | Table 3 |
| GPC-RANK (same world model) | 10 / 0 / 0 / 0 | Table 3 |
| VLM-only planner, no world model | 56 / 28 / 46 / 32 | Table 3 |
| World Action Planner | 72 / 68 / 78 / 70 | Table 3 |

- Control (Table 8): the same pi0.5 scores 90% on the one LIBERO-Long task
  that has a direct LIBERO-90 counterpart and 0 to 18% on the compositional
  ones, so the failure is composition, not an out-of-distribution scene.
  Table 4 (new-layout LIBERO-Object): World Action Planner 66 to 90% vs pi0.5
  0 to 10% and cosmos-policy 0%.
- Quotes: "while they often complete the first sub-task successfully, they
  stagnate immediately afterward with near no-op actions"; "This behavior
  stems from the policy's inability to transition from the terminal state of
  the first sub-task to the starting configuration of the next, which is
  missing in the training demonstrations." Theorems 1 to 3 give a formal
  separation: imitation suboptimality Omega(|C|/K) vs model-based
  O~(1/sqrt K).
- Caveats: expert demos, simulation, two-sub-task compositions, preprint,
  baselines fine-tuned by the authors with more demos per task than the
  planner's policy. The world model alone does not deliver the gain: SAILOR
  and GPC-RANK use the same world model and get 0 to 18%, and the VLM-only
  planner already reaches 28 to 56% ("These results highlight the importance
  of agent intervention in our system").
- Cite as: "Zhang and Du (2026) fine-tune pi0.5 and Cosmos Policy on
  LIBERO-90 and find 0 to 4% success on LIBERO-Long tasks that merely
  concatenate two LIBERO-90 tasks, the VLA emitting near no-op actions after
  the first sub-task, while a VLM-guided planner over an action-conditioned
  video world model trained on the same data reaches 68 to 78%." Frame it as
  "model-based planning composes seen sub-tasks; end-to-end imitation on the
  same data does not", not as pure world-model composition. For the Hanoi
  paper this is the most directly citable precedent that the two VLAs under
  test do not bridge between demonstrated pieces. Their "cosmos-policy" is a
  VLA baseline; the Hanoi world model is the CIDM planner, so keep the two
  names distinct in the text.

### 2.4 TVF: Transporters with Visual Foresight for Solving Unseen Rearrangement Tasks

Wu et al., IROS 2022, arXiv 2202.10765.

- Data: expert demonstrations (scripted oracle in simulation, human
  teleoperation on the real robot). Not play. The foresight model also sees
  two random actions appended to each demo; the BC policy does not.
- Model side: a goal-image-conditioned visual foresight model (predicts the
  next top-down RGB-D image after a pick-and-place) plus breadth-first tree
  search (TVF-Large: 3 candidates, depth 3) scored by L1 distance to the goal
  image, replanned every step. Candidate actions come from the BC policy.
- Imitation side: goal-conditioned Transporter Network (GCTN), same demos,
  same goal image.
- Test: 8 never-demonstrated multi-step block structures assembled from the
  one pick-and-place primitive, trained on 6 structures.

| Setting | GCTN | TVF-Large | Source |
|---|---|---|---|
| Sim, 8 unseen structures, 1 / 10 / 100 / 1000 demos per task | 1.3 / 55.4 / 49.0 / 54.2 | 2.9 / 78.5 / 71.7 / 85.6 | Tables I-II |
| Sim at 1000 demos: Stair 3 / Pallet / Rectangle / Building | 16.7 / 31.7 / 41.7 / 3.3 | 90.0 / 90.0 / 95.0 / 25.0 | Table II |
| Real robot, 3 unseen structures, average | 30.0 | 63.3 | Table III (Stair 2 40 to 80, Rectangle 50 to 50, Twin Tower 0 to 60) |

- Caveats: expert demos; "unseen" means new structures of roughly 3 to 8
  blocks (eyeballed from Fig. 5), not systematically longer tasks; GCTN
  already reaches 87 to 88% on the simplest unseen structures; the planner is
  an add-a-model-and-search ablation on top of the BC policy.
- Cite as: "Wu et al. (2022) add a learned visual foresight model and tree
  search to a goal-image-conditioned BC policy and raise success on
  never-demonstrated multi-step block structures from 55% to 79 to 86% in
  simulation and from 30% to 63% on a real robot, with the gain concentrated
  on the longest structures where BC fails." Add "trained on expert
  demonstrations". Cite it for the goal-image + search design, the same-data
  control and the real-robot precedent.

### 2.5 TACO-RL: Latent Plans for Task-Agnostic Offline Reinforcement Learning

Rosete-Beas et al., CoRL 2022, arXiv 2209.08959.

- Data: play. CALVIN environment D, 66 h of teleoperated play; real Franka,
  9 h of VR-teleoperated play.
- Stitching side: model-free hierarchical offline RL (CQL over the latent-plan
  space of a Play-LMP-style low-level policy). Explicitly "without access to
  a model". Not a world model.
- Imitation side: Play-LMP (goal-image-conditioned imitation from play),
  Relay Imitation Learning, CQL+HER, same play data.

| Setting | LMP | TACO-RL | Source |
|---|---|---|---|
| One task, sub-goal image given | 91.4 | 95.4 | Table 1 |
| Five tasks in a row, sub-goal images given one at a time | 0.2 | 6.9 | Table 1 (RIL 0.1, CQL+HER 0) |
| Final goal image two tasks away, no sub-goals | 2.7 | 27 | Table 2 (RIL 13.3, CQL+HER 2.4) |
| Real robot, 25 tasks | 40 | 61 | (CQL+HER 11) |

- Caveats: not a world model; Table 1's chaining is done by the evaluator
  (it re-conditions the policy on each sub-goal), so only Table 2 tests
  policy-side composition, at a horizon of two sub-tasks; the paper never
  verifies that the evaluated task pairs are absent from the play stream;
  Play-LMP, not a VLA.
- Cite as: "Rosete-Beas et al. (2022) train Play-LMP and an offline-RL
  skill-chaining policy on the same 66 h of play and find that LMP reaches
  91% of one-skill goals but 2.7% of goal images two skills away, versus 27%
  with explicit stitching." Use Table 2, and cite it as "offline-RL stitching
  beats goal-conditioned BC on play data" next to a true world-model paper.

### 2.6 OGBench: Benchmarking Offline Goal-Conditioned RL

Park, Frans, Eysenbach, Levine; ICLR 2025; arXiv 2410.20092.

- Data: scripted play in the Lynch (2019) style: cube (random pick-and-place),
  scene (random interaction), puzzle (random button presses, Lights Out), with
  temporally correlated noise; locomotion stitch, explore and navigate sets.
- Stitching side: value-based offline goal-conditioned RL (GCIQL, HIQL,
  GCIVL, QRL, CRL). No world model or planner.
- Imitation side: the simplest MLP goal-conditioned BC, same data.
- Definition of goal stitching: "an agent can stitch two atomic pick-and-place
  behaviors to sequentially move two objects in a single episode, even when
  the dataset does not contain any double pick-and-place behaviors."

| Dataset (Table 2, 8 seeds, state-based) | GCBC | GCIVL | GCIQL | QRL | CRL | HIQL |
|---|---|---|---|---|---|---|
| puzzle-3x3-play | 2 | 6 | 95 | 1 | 3 | 12 |
| cube-single-play | 6 | 53 | 68 | 5 | 19 | 15 |
| cube-double-play | 1 | 36 | 40 | 1 | 10 | 6 |
| scene-play | 5 | 42 | 51 | 5 | 19 | 38 |
| puzzle-4x4-play | 0 | 13 | 26 | 0 | 0 | 7 |
| antmaze-large-stitch | 3 | 18 | 7 | 18 | 11 | 67 |
| humanoidmaze-medium-stitch | 29 | 12 | 12 | 18 | 36 | 88 |

Everyone fails on the longest compositions (cube-quadruple-play all 0,
puzzle-4x6-play at most 12, antmaze-giant-stitch at most 2). Table 5
(Markovian noisy data instead of play): GCBC 8 / 1 / 1 / 0 vs GCIQL 99 / 26 /
94 / 29 on cube-single / scene / puzzle-3x3 / puzzle-4x4, so GCBC's failure is
not merely the non-Markovian nature of play.

- Caveats: the stitcher is value-based RL; the paper never singles out GCBC
  in a sentence (it is a Table 2 reading); the BC is the simplest MLP, and
  the authors explicitly ask how modern expressive BC would do, which is what
  a VLA tests. puzzle-3x3-play (512 button states, goals up to 9 presses
  away, random-press data) is the closest discrete analogue to 81-state Hanoi
  with 15-move goals.
- Cite as: "OGBench (Park et al., 2025) formalises goal stitching on play data
  and shows goal-conditioned BC collapsing on puzzle-3x3-play (2% vs 95% for
  GCIQL) and on cube and scene play datasets, even when the temporally
  correlated noise of play is replaced by Markovian noise."

### 2.7 CompDiffuser: Generative Trajectory Stitching through Diffusion Composition

Luo, Mishra, Du, Xu; NeurIPS 2025 spotlight; arXiv 2503.05153.

- Data: OGBench stitch sets (goal-reaching segments of at most 4 blocks from
  a noisy expert) and explore sets (random walks confined to 2 to 3 blocks,
  the closest analogue to coverage play); Ghugare et al. region-separated
  mazes. Simulation, state-based navigation.
- Model side: one goal-conditioned diffusion trajectory model trained on
  short chunks; 3 to 12 overlapping chunks denoised jointly into a
  start-to-goal plan, executed by an inverse-dynamics MLP trained on the same
  data. Composition happens inside diffusion sampling, not in a dynamics
  model or graph search; the "world model" framing is ours, not theirs.
- Imitation side: MLP GCBC (OGBench), RvS and DT on the Ghugare mazes.

| Dataset (medium / large / giant unless noted) | GCBC | CompDiffuser | Others |
|---|---|---|---|
| PointMaze-stitch | 23 / 7 / 0 | 100 / 100 / 68 | HIQL 74 / 13 / 0, QRL 80 / 84 / 50, GSC 100 / 100 / 29 |
| AntMaze-stitch | 45 / 3 / 0 | 96 / 86 / 65 | HIQL 94 / 67 / 21, GSC 97 / 66 / 20 |
| AntMaze-explore (medium / large) | 2 / 0 | 81 / 27 | HIQL 37 / 4, GSC 90 / 21 |
| HumanoidMaze-stitch | 29 / 6 / 0 | 91 / 72 / 67 | |
| Ghugare PointMaze (U / medium / large) | RvS 17 / 1 / 3, DT 17 / 20 / 22 | 100 / 100 / 100 | RvS + state aug 97 / 55 / 38, DT + goal aug 54 / 62 / 39, DD 0 / 30 / 0 |

- Caveats: simulation; value-based GCRL partially stitches (QRL 50 on
  PointMaze-giant); another compositional planner (GSC) ties on medium and
  large mazes; a monolithic diffusion planner (DD) fails, so the gain comes
  from compositional inference.
- Cite as: "Luo et al. (2025) train a chunk-wise diffusion planner and an
  inverse-dynamics model on random exploration data and reach goals up to 30
  blocks away while goal-conditioned BC on the same data scores 0 to 2%."

### 2.8 Few-Shot Neuro-Symbolic Imitation Learning for Long-Horizon Planning and Acting

Lorang, Lu, Huemer, Zips, Scheutz; CoRL 2025; arXiv 2508.21501 (author
list verified on the arXiv page on September 30, 2026).

- The only prior Tower of Hanoi planning-vs-imitation result. Robosuite,
  3 disks x 3 pegs (7-move optimum), 6D-pose state observations, scripted
  demos with injected noise.
- Model side: a PDDL action model learned by an ASP solver from single-MOVE
  skill demonstrations (as few as 5), a classical planner (MetricFF), and
  diffusion-policy sub-skills per operator. Not a neural world model.
- Imitation side: end-to-end diffusion policy, hierarchical H-IL and
  H-IL-Dense, trained on full-task demonstrations.
- Results (Fig. 4, ≈, bar chart, 30 episodes x 5 seeds): neuro-symbolic ≈1.0
  at 20 to 500 demos, ≈0.9 at 10, ≈0.7 at 5; IL, H-IL, H-IL-Dense ≈0 at every
  count including 500 full demonstrations. Fig. 5: zero-shot transfer from
  3x3 skill demos to 4x3 and 4x4 boards at 1.0 with 30 demos. Quote:
  "Baselines completely fail to solve the long-horizon Towers of Hanoi task
  even with 500 full demonstrations, unable to ground sub-goals and diverging
  into random outputs."
- Caveats: not the same data (the planner learns from single-move demos, the
  imitation baselines from full-task demos, so IL is never tested on
  single-move data); no neural model; simulation; state input.
- Cite as: "On a simulated 3-disk Tower of Hanoi, planning over an action
  model learned from single-move demonstrations solves the task while
  diffusion-policy imitation fails even with 500 full demonstrations (Lorang
  et al., 2025)." State that their imitation baselines saw full-task
  demonstrations. Cite it as the Hanoi precedent, not as a same-data
  world-model-vs-VLA precedent.

### 2.9 Mechanism and structure papers (no same-data world-model comparison)

- Closing the Gap between TD Learning and Supervised Learning: A
  Generalisation Point of View. Ghugare et al., ICLR 2024, arXiv 2401.11237.
  Stitching formalised as combinatorial generalisation to (state, goal) pairs
  seen separately but never together (Definition 1, Lemma 4.1); supervised
  outcome-conditioned policies have only i.i.d. guarantees, so "we should not
  expect SL-based RL methods to perform stitching, even in the limit of large
  datasets and models". Region-separated mazes (Fig. 5, ≈): RvS point-maze
  umaze / medium / large ≈0.17 / 0.02 / 0.03, with their temporal
  augmentation ≈0.76 / 0.21 / 0.31; DT ≈0.17 / 0.20 / 0.22, with augmentation
  ≈0.53 / 0.62 / 0.39. Fig. 7 (≈): DT stays at ≈0.20 to 0.25 at 1e5 to 1e7
  transitions and 3 to 6 layers. Two reusable precedents: Section 6.1 found
  that D4RL's "stitching" mazes did not test stitching because the unseen
  pairs occurred in the data; Appendix B attributes partial imitation success
  to state-only or goal-only shortcuts. No world model, planner or TD baseline
  in any experiment.
- When does return-conditioned supervised learning work for offline RL?
  Brandfonbrener et al., NeurIPS 2022, arXiv 2206.01079. Return-conditioned
  SL "is using trajectory level information during training, which precludes
  combining information across trajectories." Appendix B point-mass datasets
  (≈, bar chart, return out of 400): stitch-easy RvS ≈265, IQL ≈325, DT ≈0;
  stitch-hard (distractor continuations from the start state) RvS ≈55, %BC
  ≈60, IQL ≈320, DT ≈0. Hanoi play, with six goals from the same start and
  random-walk continuations from every board, is the stitch-hard regime.
- Free from Bellman Completeness: Trajectory Stitching via Model-based
  Return-conditioned Supervised Learning (MBRCSL). Zhou et al., ICLR 2024,
  arXiv 2310.19308. Theorems 2 and 3: Markovian RCSL and Decision
  Transformers cannot stitch even under uniform coverage. Robotics success
  (PickPlace / ClosedDrawer / BlockedDrawer): MBRCSL 0.48 / 0.51 / 0.68, DT
  0 / 0 / 0, diffusion BC 0.07 / 0.38 / 0.61. The clean 0% is the
  return-conditioned DT; unconditioned diffusion BC stitches by default on the
  drawer tasks because each task has one fixed goal and phase-2 data starts
  where phase-1 data ends. The apt analogue of a task-conditioned VLA is DT.
  Scripted segmented offline RL data, not play; offline model-based synthesis,
  no test-time planning.
- Learning Universal Policies via Text-Guided Video Generation (UniPi). Du et
  al., NeurIPS 2023, arXiv 2302.00111. Text-conditioned video diffusion
  planner plus inverse dynamics vs BC, Trajectory Transformer and Diffuser
  with the same T5 embeddings: Table 1 Novel Place 60.1 vs best baseline 13.2,
  Novel Relation 46.1 vs 9.6. Composition over language attributes at a fixed
  two-stage horizon on 200k scripted demos per environment; not temporal
  stitching.
- Search on the Replay Buffer (SoRB). Eysenbach et al., NeurIPS 2019, arXiv
  1906.05253. Dijkstra over stored observations with RL-learned distances,
  waypoint chain to a goal-conditioned controller. Fig. 7a (≈): a supervised
  inverse-model policy trained on pairs at most 8 steps apart reaches ≈13% of
  20-step goals alone and ≈85% with search on top. Online RL exploration
  data, navigation, no BC baseline by name; cite for the graph-distance
  planner structure only.

## 3. Clusters

1. Same-data head-to-head, learned model plus search composes and imitation
   on the same data does not: PLDM, CompPlan, World Action Planner, TVF,
   CompDiffuser, UniPi, MBRCSL. Only PLDM and CompPlan use play or random
   data, only World Action Planner has a VLA, only TVF is a real robot,
   UniPi composes attributes rather than time, and MBRCSL's diffusion BC
   mostly stitches by default. None combines a real robot, play data, a VLA
   and depth 15.
2. Goal stitching from play where the stitcher is value-based offline RL:
   OGBench, TACO-RL, and PLDM's own GCIQL and HILP rows. The strongest
   evidence that goal-conditioned BC from play cannot chain comes from papers
   whose fix is value learning, so the literature supports "imitation from
   play does not compose" far more strongly than "only world models compose".
3. Why outcome-conditioned imitation cannot stitch, theory and negative
   results with no world model: Ghugare et al., Brandfonbrener et al., MBRCSL
   Theorems 2 and 3, World Action Planner Theorems 1 to 3. This is the
   argument that a bigger VLA or more play would not fix the Hanoi failure.
4. Planner-structure precedents, search over stored or imagined states with
   learned distances and a local executor: SoRB, TVF, CompPlan,
   CompDiffuser, PLDM. Search plus a short-horizon controller repeatedly
   extends competence from the horizon present in the data to far goals.
5. Hanoi-specific: Lorang et al. (3 disks, asymmetric data, no neural
   model). Latplan (Asai and Fukunaga, AAAI 2018, arXiv 1705.00154) produced
   the 15-step optimal 4-disk plan by A* from all 240 image transitions with
   no policy baseline; it is the precedent a reviewer will use to argue that
   the search, not the world model, does the composing.

## 4. What the Hanoi comparison adds

Two measurements nobody reports that the 81-state graph (diameter 15) makes
possible:

1. A quantitative stitching-depth audit: per task, the number of training
   walks that connect its endpoints, the longest contiguous prefix of the
   optimal path present in the data, and the distribution of graph distances
   between (state, goal) pairs that co-occur within one 20-move episode versus
   the 15 required. The only prior audit of this kind is Ghugare et al.'s
   finding that D4RL's stitching mazes were not stitching tests.
2. A horizon curve for both systems on the same play data (goal at graph
   distance 1, 3, 7, 15), the manipulation analogue of PLDM Fig. 4, SoRB
   Figs. 6 and 7 and TACO-RL Table 1, plus the VLA-with-planner-sub-goals
   control that separates composition from execution (only World Action
   Planner Table 8 and TACO-RL Table 1 versus Table 2 approximate it).

Two real-robot play-data world-model planners were read and dropped because
their BC baselines were not trained on the same data, and are worth citing as
the nearest suggestions of the result: WorldPlanner (Khorrambakht et al., arXiv
2511.03077: about 4 h of teleoperated play trains a diffusion world model;
MCTS/MPC reaches 69 / 92 / 95 / 97% on real Push-T at 2.5 to 10 cm tolerances
vs Diffusion Policy 49 / 70 / 83 / 88% and ACT 32 / 54 / 57 / 60%, but the BC
baselines were trained on fewer than 100 separate task demos) and V-JEPA 2-AC
(arXiv 2506.09985: unlabeled DROID video, CEM planning 80% / 65% pick-and-place
vs Octo 15% / 10%, whose pretraining data differ).

## 5. Reviewer counter-evidence and defensible phrasing

1. Imitation from play does compose when sub-goals are supplied. MCIL/LangLfP
   (Lynch and Sermanet, RSS 2021) completes 52.1% of 4-instruction chains from
   pixels vs 7.1% for demo-trained BC, but the human supplies each next
   instruction after the previous one succeeds. Relay Policy Learning (Gupta
   et al., CoRL 2019) reaches 21.7% of 4-stage compound goals vs 8.8% for flat
   GCBC, on goals seen in the demonstrations. MimicPlay (CoRL 2023)
   generalises to new sub-goal compositions with the sub-goal sequence given
   by the prompt video, while flat GC-BC, C-BeT, LMP and R3M-BC score at most
   0.17 on tasks with three or more sub-goals. CALVIN's MCIL completes 48.9 /
   12.9 / 2.6 / 0.5 / 0.08% of 1 to 5 instructions in a row; later
   play-trained CALVIN policies chain further (numbers not verified here).
   Common thread: composition succeeds when the intermediate sub-goals come
   from the evaluator or a learned high-level planner and fails when only the
   final goal is given (TACO-RL: 91.4% one skill away, 2.7% two away). Scope
   the claim to flat, outcome-conditioned imitation given only the final
   goal. Do not cite Play-LMP either way: its full text has no chaining
   experiment.
2. Stitching-aware augmentation partly closes the gap for supervised
   policies: Ghugare's temporal augmentation lifts RvS from ≈0.17 to ≈0.76 on
   point-maze umaze; CompDiffuser's table has RvS with state augmentation at
   97 / 55 / 38 vs 17 / 1 / 3. Write "without a mechanism for combining
   segments (planning, value bootstrapping or stitching-aware relabelling)"
   rather than "BC cannot compose".
3. Value-based offline goal-conditioned RL stitches as well as or better than
   the world-model planner: PLDM Table 2 GCIQL 99.6 and HILP 100.0 vs PLDM
   34.4 vs GCBC 8.4; OGBench GCIQL 95 on puzzle-3x3-play. Never claim world
   models are the only or the best route; claim "world model + search
   composes, end-to-end imitation on the same data does not". A reviewer will
   ask for a GCIQL or HIQL baseline on the play data.
4. Expressive BC stitches by default when there is one fixed goal and no
   distractor continuations: MBRCSL diffusion BC 0.61 vs 0.68; TVF's GCTN
   87 to 88% on the simpler unseen structures; Brandfonbrener's stitch-easy
   panel; PLDM's GCBC at 86% with long coverage trajectories. Hanoi with six
   goals from the same start and random-walk continuations from every board
   is the stitch-hard regime; say so explicitly.
5. The data audit is the single biggest risk. Ghugare et al. sank D4RL's
   stitching claim by finding the "unseen" pairs in the data. The manifest's
   "no whole task episodes" rule holds for AAAA to CCCC (and no whole walk
   exists for AAAA to BBBB or BBBB to AAAA), but 7 of the 40 full-stack to
   full-stack 20-move walks are whole in the CIDM training split (episodes
   15, 60, 84, 100, 103, 104, 121: four BBBB to CCCC, two CCCC to BBBB, one
   CCCC to AAAA), and a 20-move walk between boards at graph distance 15 is a
   noisy demonstration, not play. Report the section 4 audit per task, and
   exclude or disclose those seven walks and the four expert AAAA to CCCC
   clips (the opening and closing moves of Sept 25 episodes 0 to 3) for both
   systems.
6. Conditioning parity and label ambiguity. A language-conditioned VLA given
   six prompts the play data never carried fails partly for lack of labels.
   The BC baseline that matches the literature gets the same goal image with
   hindsight relabelling that the world model gets, and a second control is
   fed the planner's next-board sub-goal (World Action Planner Table 8,
   TACO-RL Table 1 vs 2, CompPlan); if that control executes single moves,
   the failure is composition, not execution. 45% of (board, goal)
   occurrences in the walks have more than one next move (64% for goal
   CCCC): pre-empt "it is label noise" with OGBench Table 5, and report moves
   completed before the first error and per-task results, not only binary
   success (Ghugare Appendix B; the first moves of several Hanoi tasks
   coincide).
7. Who composes, the world model or the planner? If the 81-state graph and
   its distances are hand-coded from Hanoi rules rather than built from play
   transitions, a reviewer will point to World Action Planner's controls
   (VLM-only 28 to 56% vs world model alone 0 to 18%), Lorang et al. and
   Latplan and say the search does the work. State how the graph is obtained,
   phrase the contribution as "the world model makes each planned move
   executable from play data; search composes them", and ablate the world
   model (planner + inverse dynamics with retrieved rather than generated goal
   images).
8. Naming. World Action Planner's failing baselines are pi0.5 and
   "cosmos-policy", the two VLAs in the Hanoi comparison, so it is direct
   evidence about these models. The Cosmos Policy paper (ICLR 2026, arXiv
   2601.16163) makes no compositional claim, and V-JEPA 2's Table 3 reports an
   action-conditioned Cosmos video model with CEM planning at 0% pick-and-place
   (per the workflow's read; check before citing). Keep "Cosmos Policy, the VLA
   baseline" and "the CIDM world model with graph-distance planning" clearly
   distinct throughout.

Phrasing that survives all eight, with the blanks to fill from the
experiments: "Trained on the same play recording, in which every one-move
transition appears but none of the six 15-move tasks does [audit table], the
goal-image-conditioned world model with graph-distance planning solves X% of
the never-demonstrated tasks, whereas a behaviour-cloned VLA conditioned on the
same goal image solves Y% (Z of 15 moves on average), even though it executes
N% of single moves when given the planner's next-board sub-goal. This is
consistent with prior evidence that outcome-conditioned imitation does not
combine trajectory segments that never co-occur in the data (Ghugare et al.,
2024; Park et al., 2025; Rosete-Beas et al., 2022) whereas search over a
learned model does (Sobal et al., 2025; Farebrother et al., 2026; Wu et al.,
2022; Zhang and Du, 2026). We do not claim that world models are the only route
to composition: value-based offline goal-conditioned RL also stitches (Park et
al., 2025; Sobal et al., 2025) and stitching-aware augmentation partially
closes the gap for supervised policies (Ghugare et al., 2024)."

## 6. The same-data VLA on the play recording: recipe

Written September 30, 2026, after a verification pass (OGBench, PLDM,
CompPlan, TACO-RL, CALVIN and TVF source code and appendices; the Cosmos
code path for an extra image slot; an adversarial review of the draft).
Nothing here is built.

Section 6.10 holds the final labeling rules; where 6.2 or 6.3 differ from
it, 6.10 wins.

### 6.1 Data and the absence audit

Exactly the manifest's training set (78 whole walks, 18 one-move crops, 4
expert clips), their frame filter, every remaining 30 Hz row as an
observation, their validation and test walks. The manifest's "no whole task
episodes" rule removes walks whose endpoints are task endpoints, but it does
not catch walks that pass through both endpoints mid-episode. Routes of the
78 training walks (from the recording's JSON sidecar; the 81-state graph has
diameter 15 and a unique shortest path between full stacks):

| Task | Training walks containing start before goal | Longest contiguous optimal-path segment in any training walk |
|---|---|---|
| AAAA to CCCC | 1 (walk 26: AAAA at move 4 to CCCC at move 20, 16 moves, 14 on the optimal path) | 13 of 15 (walk 52, BCAA to CCCC); 12 in walks 44 and 70 |
| CCCC to AAAA | 1 (walk 121: whole 20-move walk, 10 optimal) | 12 (walk 81) |
| AAAA to BBBB | 0 | 8 (walk 0) |
| BBBB to AAAA | 0 | 5 |
| BBBB to CCCC | 5 (walk 9: 17 moves, 12 optimal; walks 60, 84, 100, 104 whole, 8 to 13 optimal) | 14 (walks 48 and 56, ABBB to CCCC) |
| CCCC to BBBB | 3 (walk 39: 16 moves, 14 optimal; walks 15 and 103 whole, 13 and 11 optimal) | 14 (walk 39, CCCC to ABBB) |

Co-occurring (current board, future board) pairs within one training walk:
16,380; graph distance 15 for 298 of them (1.8%), at least 10 for 3,294
(20%). Future board is a full stack in 780 pairs for CCCC, 78 for BBBB, 24
for AAAA.

Only AAAA to BBBB and BBBB to AAAA are clean. Two consistent options, the
choice is the user's: rebuild the manifest without walks 9, 15, 26, 39, 60,
84, 100, 103, 104 and 121 and the four expert clips and retrain both systems
on it; or keep the manifest for both systems, make AAAA to BBBB and BBBB to
AAAA the headline tasks, and publish this table per (start, goal) pair.
Excluding walks for the VLA only would handicap it on four of six tasks.

### 6.2 Conditioning and goal sampling

Goal image only, the signal CIDM gets; no language (Cosmos needs one constant
T5 prompt embedding because text dropout is 0 and the conditioner requires an
embedding; the multitask cache serves one fixed prompt as the degenerate
case). Training goal: a future frame of the same walk, drawn uniformly from
the rows after the current row up to the walk's last row. This is the OGBench
GCBC default (gcbc.py: actor_p_trajgoal 1.0, actor_geom_sample False,
actor_p_randomgoal 0.0; datasets.py samples uniformly from idx+1 to the
trajectory's final state), used unchanged by PLDM; CompPlan uses a geometric
version with a per-domain discount; TACO-RL and CALVIN's LMP use the last
frame of a window of 8 to 16 or 16 to 32 frames; TVF uses the episode's final
image. One disclosed deviation: snap the sampled goal row to the nearest
canonical frame (settled board, arm in a canonical pose), because a goal
frame with the arm mid-move leaks the move just made (a goal-only shortcut in
the sense of Ghugare et al., Appendix B). The same canonical-frame rule must
be what CIDM receives at test time, which depends on the open question in
6.9. Optional ablation: geometric sampling with a mean horizon of one to two
moves (500 to 1,000 rows), the CompPlan analogue.

### 6.3 Labels

Dense v5 rule unchanged: 16 slots at 10 Hz, absolute XYZ plus jaw intent,
slot j = row t + 3j, padded past the walk's end. The execution side stays
identical to the validated six-task recipe.

### 6.4 Cosmos implementation

Nine latent slots in the LIBERO geometry: blank, proprio, image, goal image,
action, future proprio, future image, future goal (the goal repeated, a
fixed target), value; state_t 9, min and max conditional frames 4 (the
conditioner marks the first num_conditional_frames latent frames as clean, so
the goal must sit among the leading slots), chunk_duration 33. No model-code
change: the DiT and VAE accept any 1 + 4k length up to 128 latent frames and
the checkpoint has no state_t-shaped weights (the state_t 9 LIBERO checkpoint
already loads into the state_t 7 Hanoi net). What changes: a new dataset
class and contract that records state_t, slot order, goal-source rule and the
sqrt(state_t) noise multiplier; the config; validate_dense_config
(num_third_person_images 2) and the (7, 3) check in load_dense_policy;
make_joint_observation (secondary_image), HanoiDensePolicy.infer and the
server payload (goal-image key); the evaluator constants HANOI_UNDO_INJECTION
and FUTURE_FRAME_INDEX; the parity path. About 29% more tokens per sample, so
the micro-batch may need lowering. Video-base init as in every Hanoi run
(LIBERO's slot 3 was its primary image, so its weights would transfer with
different semantics). pi0.5 mirror: the goal image as a second camera image,
owned by the OpenPI agent.

### 6.5 Training and checkpoint selection

The dense recipe, effective batch 32, export every 2,000, at least 64,000
updates (two cycles, as the six-task run needed): the training split is about
0.8M rows, so 32,000 updates is 1.3 passes, while OGBench's pixel GCBC trains
500,000 steps at batch 256. Do not select by overall validation chunk error:
a move lasts 16.5 s on average (2,625 move segments, mean 495 rows) against a
1.6 s chunk, so most rows are goal-insensitive and a goal-ignoring checkpoint
can score best. Select on decision rows (pre-pick motion stages) with a
shuffled-goal control: the same state decoded under the true goal and a wrong
goal, reporting chunk divergence and source-peg agreement, with the full
curves published.

### 6.6 Runs

- (a) The goal-image VLA given the final goal frame: the head-to-head.
- (c) The same VLA given the planner's next-board goal frame: the execution
  control (World Action Planner Table 8, TACO-RL Table 1 vs 2, CompPlan). If
  the planner consumes anything beyond images (the hand-coded 81-node graph,
  a board classifier), (a) is a system-level comparison and (c) is the
  headline model-level one; add a VLA variant with the same sub-goal
  privilege so both sides have equal information. State what the planner
  consumes.
- (b) A per-move-instruction VLA with a scripted solver: supplementary; it
  measures a different model's execution.

### 6.7 Evaluation

A fixed, shared list of (start board, goal board) pairs per graph distance 1,
3, 7 and 15, with the same physical resets, the same recorded goal frame and a
step budget of twice the optimal move count for both systems; report success
and moves before the first error per pair, plus the in-data prefix per pair.
Offline proxy on decision rows only: classify the source peg from the
approach heading and jaw intent inside the chunk (the release comes about 15
s later, so release position is not in the chunk), or decode the future-image
slot and read the board; calibrate the classifier on same-episode goals first
and report its accuracy; score whether the implied move reduces graph
distance to the goal; report same-episode and cross-episode goals side by
side.

### 6.8 Audit tables to publish

Under the chosen sampler: the fraction of (state, goal, action) triples whose
action increases graph distance to the goal, split by phase (the scramble
half labels distance-increasing actions whenever the goal lies past the
scramble, the Brandfonbrener distractor regime); per-goal-board counts at
each graph distance; the 45% and 64% label-ambiguity figures with OGBench
Table 5 against the label-noise objection; optionally an ablation that
restricts goal sampling to the current phase.

### 6.9 The open question

What is the world model's test-time goal image: a real recorded frame of the
target board (from which episode, with the arm in what pose), a canonical
render, or a same-episode frame; and will the VLA be handed that identical
frame at both training-goal sampling and evaluation? The goal-frame rule in
6.2 cannot be fixed until this is answered.

### 6.10 Labeling rules for the play data (final, language-conditioned)

Written September 30, 2026, after a second verification pass; revised the
same day when the user decided the VLA is conditioned on language only, no
goal image. Rules 3 to 5 changed accordingly; the measured statistics and
rules 1, 2, 6, 7 and 8 stand. Precedent
rules checked at the source: OGBench GCBC samples the goal uniformly over the
future states of the same trajectory, k ~ Unif(min(t+1, T-1), T-1), with no
filtering or reweighting of (state, action, goal) triples; GCSL uses every
(t, h) pair without reweighting; WGCSL down-weights low-advantage labels but
never removes them (floor 0.05); Play-LMP, LangLfP and CALVIN take the last
frame of a 16-to-32-frame window as the goal and put the loss only on the
actions inside the window; TACO-RL uses 8-to-16-frame windows and is the one
precedent that discusses the arm in goal images ("LMP has a strong bias
towards the end-effector position ignoring the changes in the environment");
TVF uses the demonstration's final image. Measured on the 78 training walks
(routes from the recording's sidecar, stages and poses from the h5 file):

| Quantity | Value |
|---|---|
| Training rows after the stale/repeated filter | 785,001 |
| Rows per move, mean / median | 480 / 500 (16 s) |
| Pre-grasp rows per move (open, approach_source, descend_source) | 161 |
| Decision rows (approach_source + transit stages) | 251,010 (32%) |
| Labels under rule 3 that reduce / keep / increase graph distance to the goal | 78.4% / 16.6% / 5.0% |
| (board, goal) occurrences whose pair has more than one next move in training | 59.1% (majority move 82.3%) |
| Labels that are an optimal move for their goal | 78.4% |
| Goal at graph distance 1 / at least 10 / 15 | 11.7% / 20% / 1.8% |
| Future full-stack goal boards, CCCC / BBBB / AAAA | 780 / 76 / 24 pairs |
| (row, goal) pairs whose segment is a shortest path (rule 7) | 47.7%; farthest optimal goal median 4 moves, at least 5 moves for 45% of moves |
| (board, goal) pairs with more than one optimal next move | 282 of 6,480 (4.4%) |
| Commanded y at the first row of a move, across pegs | std 57 mm, range 142 mm |
| Stage at the first / last row of every move | open / retreat; board label updated by the last row |

Rules, one per training row t:

1. Rows and observation. The identical manifest the world model uses (78
   whole walks, 18 one-move crops, 4 expert clips), the same filter, every
   surviving 30 Hz row a training row; observation = the image and the six
   joints plus jaw; no row weighting.
2. Action target. The dense v5 chunk geometry (16 slots at 10 Hz, slot j =
   row t + 3j, absolute XYZ from reference_pose plus jaw intent from
   action_abs), cut at the last row of the commanded segment and padded with
   that row's pose and jaw, the existing walk-end rule: the goal move's last
   row in the goal run, the current move's last row in the instruction run,
   the walk's last row otherwise. Rows after the release of the goal move
   carry an all-hold chunk. Without the cut, rows in the last 1.6 s of the
   goal move would be labelled with the start of the walk's next, unrelated
   move, a target nothing observable explains, and the VLA would never learn
   to stop at the goal.
3. Goal board. Sample the move index j uniformly over the current move and
   every later move of the walk; the goal board is the board after move j.
   Lateral and regress labels are kept; no filtering, no reweighting. The
   snap to move ends is the one deviation from row-level uniform sampling,
   disclosed. This is hindsight relabelling in the LangLfP and CALVIN sense,
   with a templated sentence instead of a human or a task detector.
4. Goal sentence. The goal board is put into words with one fixed template
   over all 81 boards, rings numbered from the smallest, one sentence per
   board, served as cached T5 embeddings exactly as the six-task run serves
   its six prompts (prompt required verbatim, no default, no text encoder at
   inference). The six tower tasks are the six full-stack instances of the
   same template, so the test prompt is a training prompt. No goal image, so
   the arm-pose shortcut of the earlier rule disappears, and the sentence
   carries what a goal image carries after perception, the board, which is
   a stated advantage for the VLA. Template choice: the family whose
   per-token T5 embeddings separate one-ring-different boards best (four
   candidates encoded on September 30; result below). The existing six-task
   phrasing is not reused, because "from peg A to peg C" presumes a
   full-stack start that the play rows do not have.

   Template check (T5-11B, float32, CPU job 18882631, script and JSON under
   `data/hanoi_cosmos/smoke/prompt_diagnostic*`): four templates were
   encoded for all 81 boards and compared by the summed per-token L2
   divergence between prompts, the quantity that separated the six task
   prompts (their closest pair scores 18.1 and their probe reached 97%).
   The hardest cases are boards that differ by one ring.

   | Template | Tokens | Closest pair, all boards | Closest one-ring pair | Mean one-ring pair | Pooled cosine, min |
   |---|---|---|---|---|---|
   | Peg contents: "Goal: peg A holds rings 1, 2, 3 and 4, peg B is empty, peg C is empty." | 25 | 18.1 | 72.4 | 103.4 | 0.833 |
   | Ring on peg: "Goal: ring 1 on peg A, ring 2 on peg A, ring 3 on peg A, ring 4 on peg A." | 35 | 11.0 | 11.0 | 46.8 | 0.947 |
   | Sizes: "Arrange the rings so that the smallest ring is on peg A, ..." | 48 | 13.7 | 13.7 | 59.6 | 0.947 |
   | Task style: "Move the rings following Tower of Hanoi rules until ring 1 is on peg A, ..." | 47 | 15.1 | 15.1 | 58.8 | 0.955 |
   | Reference: the six task prompts | 32 | 18.1 | | 37.7 (all pairs) | 0.972 |

   Decision: the peg-contents template. Moving one ring changes two clauses,
   so its closest one-ring pair is four times the six-task reference and
   six times the ring-on-peg template's; its closest pair overall equals the
   reference. Rings are numbered from the smallest; a peg with one ring reads
   "peg C holds ring 1", an empty peg "peg B is empty"; every board is 25
   tokens. The divergence is a proxy: the deployment gate remains the
   language probe after training, as in the six-task run.
5. Execution control. The same VLA handed the planner's next-board sentence
   at test (the template instance of the next board), with the same
   cut-and-hold semantics, so both sides consume the same sub-goal privilege.
6. Instruction run, supplementary. Every row of move i, open through retreat,
   carries the command of move i (ring, source peg, target peg: 24 templates
   served as cached T5 embeddings, the six-task mechanism); chunk cut at move
   i's last row; at test the controller issues the next command once the
   board is settled, the same event as the training cut; a walk's terminal
   hold rows carry a hold command.
7. Oracle-filtered ablation. Rules 1 to 4 restricted to (row, goal) pairs
   whose recorded segment is a shortest path to the goal board. It uses the
   Hanoi graph, so it is reported only next to a statement of what the
   planner consumes (hand-coded or learned graph), paired with rule 5, and
   never as the head-to-head.
8. Selection and test. Checkpoint selection on approach_source and transit
   rows: sample several chunks, classify the implied source or target peg
   from the approach heading and jaw intent (calibrated on same-episode
   goals), report agreement with the recorded peg and with the optimal-peg
   set, and require a shuffled-goal control to fall to chance; XYZ error
   reported, not selected on. Test resets from the play distribution (arm
   above a peg after a settled move) or a stated home start added to both
   systems' data, because the manifest's task start "arm still at home"
   occurs in none of the play rows. Goal frames from the held-out walks,
   identical for both systems, episode and arm pose recorded; a fixed shared
   (start, goal) list stratified by graph distance 1, 3, 7, 15 and by goal
   board, with the per-goal-board training counts published; CCCC never
   headlined alone.

## 7. Corrections to the interim list sent on September 30

- World Action Planner trains on LIBERO-90 expert demonstrations, not play,
  and its world model alone does not deliver the gain (VLM-only planner 28 to
  56%). It remains the closest VLA precedent because its failing baselines
  are pi0.5 and cosmos-policy.
- PLDM beats GCBC in every stitching test but not GCIQL or HILP; the
  interim wording "planning beats model-free methods" is wrong.
- CompDiffuser is a compositional diffusion planner plus inverse dynamics;
  calling it a world model is our framing.
- Free from Bellman Completeness: the clean 0% is the return-conditioned DT;
  its diffusion BC stitches by default on two of three tasks.
- Lorang et al.: author list now verified.
- WorldPlanner, HWM, V-JEPA 2-AC, DINO-WM, Latplan, MimicPlay, CALVIN,
  Play-LMP, VLP, HiP, SPTM and MBOLD were listed in the interim as analogues
  or lineage; on a full read none is a same-data composition comparison
  (section 8). UniPi and SoRB are kept only as attribute-composition and
  planner-structure precedents.

## 8. Papers read and dropped

| Paper | Why it is not a precedent |
|---|---|
| Slot-MPC (arXiv 2605.14937) | Four single-skill tasks fully demonstrated end to end; only object positions are unseen. |
| MBOLD (arXiv 2012.15373) | Unseen goals are states the same 30-step random policy reaches; no longer-than-training claim; never uses compose or stitch. |
| Goal-Conditioned Hierarchical Predictors (arXiv 2006.13205) | Training trajectories are full start-to-goal paths from the test distribution; the result is improvement over suboptimal data, not composition. |
| WorldPlanner (arXiv 2511.03077) | Composition stated as a hypothesis; push-T problems are cut from play chunks; BC baselines trained on separate task demos, not the play set. |
| V-JEPA 2 (arXiv 2506.09985) | Pick-and-place composed through two human-supplied sub-goal images; the task is common in DROID; Octo's data differ. |
| Play-LMP (arXiv 1903.01973) | 18 single short tasks that appear as subsequences of play; no chaining experiment. |
| Goal-conditioned Offline Planning from Curious Exploration (arXiv 2311.16996) | Gains framed as correcting value-estimation artifacts; no absence check. |
| Video Language Planning (arXiv 2310.10625) | The three evaluated goals are in the training goal set; baselines trained on long-horizon trajectories. |
| Semi-parametric Topological Memory (arXiv 1803.00653) | Navigation in unseen mazes after a walkthrough; no composition claim. |
| HiP (arXiv 2309.08587) | Recombination of attributes at the same 3 to 6 sub-goal horizon, not longer tasks. |
| Cosmos Policy (arXiv 2601.16163) | No compositional claim; planning evaluated on hard initial states of trained tasks. |
| MimicPlay (arXiv 2302.12422) | Sub-goal sequence supplied by the prompt video; executes given transitions. |
| CALVIN (arXiv 2112.03227) | Imitation-only negative result, no planning comparison. |
| Relay Policy Learning (arXiv 1910.11956) | Test goals seen in the demonstrations; extrapolation listed as future work. |
| MCIL / LangLfP (arXiv 2005.07648) | Next instruction supplied by the human after each success; no absence claim. |
| Latplan (arXiv 1705.00154) | 15-step 4-disk plan by A* from all 240 transitions with an oracle model; no policy baseline. |
| Diffuser (arXiv 2205.09991) | Qualitative straight-line to V-shape stitching demo without a baseline; Maze2D compared to offline RL only. |
| HWM (arXiv 2604.03208) | Non-greedy multi-stage tasks from one goal image; no absence claim; VLA comparison framed as data efficiency. |
| Grounded World Model (arXiv 2604.11751) | Semantic generalisation; test tasks solvable with demonstrated motions; MPC retrieves demonstrated trajectories. |
| Latent Diffusion Planning (arXiv 2504.16925) | Same short single-stage tasks as the demonstrations. |
| DINO-WM (arXiv 2411.04983) | Short-range goals (at most 25 steps) and unseen configurations; tasks shorter than the training trajectories. |
