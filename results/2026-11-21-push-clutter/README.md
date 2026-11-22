# Two new tasks and their hand-written seed skills: push and pick-from-clutter (2026-11-21)

Sim only (MuJoCo Franka), hand-written skills, no agent runs. Gate-style seeds 0-99, 12 workers,
`eval.parallel.evaluate_parallel`. "Stress x2" is T4's `stress_spec(randomization, 2.0)`.

## Tasks
- `tasks/franka_push_box_to_target.json`: a 10 x 10 x 5 cm box (0.3 kg), wider than the 8 cm
  gripper, pushed ~33 cm diagonally into a 6 x 6 cm square. Jitter: xy 4 cm, yaw 0.6 rad,
  friction 0.6-1.4x, mass 0.7-1.8x, gain 0.8-1.2x, obs noise 5 mm, latency 0-2 steps.
- `tasks/franka_pick_cube_from_clutter.json`: the 4 cm cube between two 10 cm cans (8.5 cm
  either side along y) and a 3 cm block (8.5 cm along +x); put it in the bin. Jitter: xy 1 cm per
  object, yaw 0.785 rad, the rest as `franka_pick_cube_to_bin`. Designed so that one grasp axis
  is blocked by the tall cans and the other is clear over the low block.

## Results
The table below was measured before the arm-armature fix (#22, armature 0.1). On the fixed sim
(armature 0.5), seeds 0-99: clutter `PickFromClutter` 100/100 (the 14 timeouts were the servo
bang-bang artifact), clutter `PickPlace` 71/100 (collision 26: real hand-into-can geometry),
push `PushToRegion` 0.2.0 under stress x2 100/100. The other cells were not re-run.

| Task | Skill | Nominal | Stress x2 |
| --- | --- | --- | --- |
| push | `PushToRegion` 0.2.0 (new) | 100/100 | 98/100 (collision 1, IK 1) |
| push | `PushToRegion` 0.1.0 (first version) | 99/100 (timeout 1) | 91/100 (timeout 8, collision 1) |
| push | `PickPlace` 0.2.0 | 0/20 (timeouts: no feasible grasp) | not run |
| clutter | `PickFromClutter` 0.1.0 (new) | 86/100 (timeout 14) | 83/100 (collision 8, timeout 5, slip 3, IK 1) |
| clutter | `PickPlace` 0.2.0 | 63/100 (collision 37) | 73/100 (collision 23, ...) |

Stress x2 is easier than nominal for `PickPlace` on clutter: doubling the jitter spreads the
objects apart, so fewer hands hit cans.

## How the skills work
- `PushToRegion`: fingers half open with the closing axis across the push direction (two contact
  points ~5 cm apart), pusher placed just behind the object on the line to the region centre and
  re-aimed every step; if the object drifts > 2.5 cm off the line the hand lifts and re-approaches.
  v0.2, from tracing v0.1's 8 stress timeouts: (a) with up to 1 cm of observation noise it called
  success early and never looked again, so the object's xy is now smoothed and it keeps checking
  after retreating (re-pushing if outside); (b) high friction or mass tipped the box, so it pushes
  below mid-height. 91 -> 98 under stress (v0.1 failed seeds 0, 13, 22, 31, 40, 52, 61, 62, 96;
  v0.2 fails 18, 97). The three changes were not ablated separately.
- `PickFromClutter` = `PickPlace` plus three changes, each found by tracing failures:
  1. Grasp choice by hand clearance. The Panda hand is 0.2 m long along the closing axis and its
     underside is only 3.7 cm above the TCP (measured from the menagerie collision geometry; my
     first guess, 5.5 cm, was wrong and the first task version was unsolvable top-down). Grasps
     blocked only by short neighbours may be raised up the object. Collisions 37 -> 0.
  2. Carry height over the tallest neighbour. With the default 10 cm hover the held cube's
     underside was level with the can lids and got dragged across one (slip 34 -> 0).
     Lower clearances measured worse (1 cm: 66, 2 cm: 64; 4 cm: 86).
  3. Gripper pre-shape: open to the grasp width + 2 cm instead of 8 cm (86 vs 83 without;
     within noise, kept). Added to `PickPlace` as `preshape_margin` (default None = unchanged).

## Open
- The 14 clutter timeouts (cube yaw 23-37 deg or about 0): traced to the arm, not the skill. In
  seed 5 the arm holds the cube at carry height with joint 2 at 0.11 rad, where joints 1 and 3 are
  nearly coaxial (shoulder singularity). From then on joints 1 and 3 sit at +87 / -87 Nm (their
  force limits) and joint 7 at its 12 Nm limit, joint 1 reports a steady 1.16 rad/s while its
  angle moves 0.004 rad per 0.5 s, and the hand creeps several cm from its target for 300 steps:
  most likely a high-frequency chatter between the two shoulder servos. No contact, IK failure or
  joint limit is involved and no event is raised. The IK's null-space pull goes toward the home
  posture, whose joint 2 is 0, i.e. toward the singular configuration. Reported to T1 (HANDOVER).
- Neither skill has been through the agent loop.
