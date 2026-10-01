# Arm protocol for the play-trained policies (hanoi_play_k5)

October 1, 2026, Cosmos-side agent, for the deployment agent. Applies to the
Cosmos play export (`hanoi_cosmos_play_20260930_video_init`, step 32,000),
the pi0.5 play model when it exists, and the world-model planner. The
six-task models only run the distance-15 trials of protocol A with their own
task prompts.

## Protocols

- **A, final goal (the comparison).** The goal board's sentence is fixed for
  the whole trial. No sub-goals.
- **C, next board (diagnostic, reported in its own table).** After every
  move, the client sends the sentence of the next board on the shortest path
  from the tracked board to the trial's goal. This is the execution control:
  it separates "cannot compose" from "does not read the goal".

## Trial

Start: the full tower of the task, arm at the grasp hover over the start
peg, jaw open (where every pick in play begins). Budget: twice the goal's
graph distance in moves (2, 6, 14, 30). A move is one ring released on a
peg. The trial ends at success (the tracked board equals the goal board),
when the budget is spent, or on a failure (dropped ring, placement on a
smaller ring, or 60 s without a completed move). Completion is judged from
the tracked board, never from the policy going quiet (a policy whose board
already matches its sentence emits hold chunks).

Scored per trial: success; moves before the first error, where an error is a
move that does not reduce the graph distance to the trial's goal or a failed
move; the full move sequence. Three trials per pair, the same resets and
sentences for every policy, pairs in a fixed order.

## Pairs

Protocol A, distance 15: all six tasks, 3 trials each (18 trials).
Protocol A, distances 1, 3, 7: two tasks, AAAA to BBBB (no training walk
connects them; BBBB is a rare goal) and AAAA to CCCC (the common goal),
3 trials each (18 trials). Protocol C, distance 15: all six tasks, 3 trials
each (18 trials). 54 trials per policy.

Boards are peg per ring with ring 1 the smallest. The goal boards at
distance d lie on the task's unique shortest path:

| Task | d = 1 | d = 3 | d = 7 | d = 15 |
|---|---|---|---|---|
| AAAA to CCCC | BAAA | CCAA | BBBA | CCCC |
| CCCC to AAAA | BCCC | AACC | BBBC | AAAA |
| AAAA to BBBB | CAAA | BBAA | CCCA | BBBB |
| BBBB to AAAA | CBBB | AABB | CCCB | AAAA |
| BBBB to CCCC | ABBB | CCBB | AAAB | CCCC |
| CCCC to BBBB | ACCC | BBCC | AAAC | BBBB |

Full shortest paths (for protocol C and for scoring errors):

```
AAAA->CCCC: AAAA BAAA BCAA CCAA CCBA ACBA ABBA BBBA BBBC CBBC CABC AABC AACC BACC BCCC CCCC
CCCC->AAAA: CCCC BCCC BACC AACC AABC CABC CBBC BBBC BBBA ABBA ACBA CCBA CCAA BCAA BAAA AAAA
AAAA->BBBB: AAAA CAAA CBAA BBAA BBCA ABCA ACCA CCCA CCCB BCCB BACB AACB AABB CABB CBBB BBBB
BBBB->AAAA: BBBB CBBB CABB AABB AACB BACB BCCB CCCB CCCA ACCA ABCA BBCA BBAA CBAA CAAA AAAA
BBBB->CCCC: BBBB ABBB ACBB CCBB CCAB BCAB BAAB AAAB AAAC CAAC CBAC BBAC BBCC ABCC ACCC CCCC
CCCC->BBBB: CCCC ACCC ABCC BBCC BBAC CBAC CAAC AAAC AAAB BAAB BCAB CCAB CCBB ACBB ABBB BBBB
```

## Sentences

Generate every sentence with `prompt_for_board` in
`cosmos_policy/datasets/hanoi_play_data.py` or `GET /prompt/<board>` on the
play server; the strings must match the 81 trained sentences byte for byte.
The ones used by the pairs above:

```
AAAA: Goal: peg A holds rings 1, 2, 3 and 4, peg B is empty, peg C is empty.
BBBB: Goal: peg A is empty, peg B holds rings 1, 2, 3 and 4, peg C is empty.
CCCC: Goal: peg A is empty, peg B is empty, peg C holds rings 1, 2, 3 and 4.
BAAA: Goal: peg A holds rings 2, 3 and 4, peg B holds ring 1, peg C is empty.
CCAA: Goal: peg A holds rings 3 and 4, peg B is empty, peg C holds rings 1 and 2.
BBBA: Goal: peg A holds ring 4, peg B holds rings 1, 2 and 3, peg C is empty.
CAAA: Goal: peg A holds rings 2, 3 and 4, peg B is empty, peg C holds ring 1.
BBAA: Goal: peg A holds rings 3 and 4, peg B holds rings 1 and 2, peg C is empty.
CCCA: Goal: peg A holds ring 4, peg B is empty, peg C holds rings 1, 2 and 3.
```

## Reporting

Per policy and protocol: success and mean moves before the first error per
pair, then by distance and by goal board. For the Cosmos play policy, read the
offline goal probe beside it (`probe_v2_val.json`, `probe_v2_test.json` in the
run directory): the sentence decides about half of the matched decisions
offline, so a low distance-15 success under protocol A with a high one under
protocol C is the expected signature.
