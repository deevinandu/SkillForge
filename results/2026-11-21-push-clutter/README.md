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
| Task | Skill | Nominal | Stress x2 |
| --- | --- | --- | --- |
| push | `PushToRegion` 0.1.0 (new) | 99/100 (timeout 1) | 91/100 (timeout 8, collision 1) |
| push | `PickPlace` 0.2.0 | 0/20 (timeouts: no feasible grasp) | not run |
| clutter | `PickFromClutter` 0.1.0 (new) | 86/100 (timeout 14) | 83/100 (collision 8, timeout 5, slip 3, IK 1) |
| clutter | `PickPlace` 0.2.0 | 63/100 (collision 37) | 73/100 (collision 23, ...) |

Stress x2 is easier than nominal for `PickPlace` on clutter: doubling the jitter spreads the
objects apart, so fewer hands hit cans.

## How the skills work
- `PushToRegion`: fingers half open with the closing axis across the push direction (two contact
  points ~5 cm apart), pusher placed just behind the object on the line to the region centre and
  re-aimed every step; if the object drifts > 2.5 cm off the line the hand lifts and re-approaches.
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
- The 14 clutter timeouts: cube yaw 23-37 deg or about 0. The arm holds the cube at carry height
  and creeps away from its commanded target (4+ cm, for 300 steps) with no contact, IK failure or
  joint limit flagged. Looks like a controller issue at that wrist configuration (T1); not traced.
- Push under stress: the 8 timeouts are not traced.
- Neither skill has been through the agent loop.
