# Brief · Scoped skill library across arms

**Goal:** measure whether scoped repairs (general vs robot-specific, with general fixes checked on
the other robots first) beat sharing everything, starting from scratch, and copying unchanged.
Background and design: [`docs/PROJECT.md`](../PROJECT.md).

**Gate for this brief:** for one task, every condition in the PROJECT.md table has repeated
runs on at least two robot types in sim, with repair counts, costs, paired results on held-out
seeds, and a negative-transfer number. Sim only until the real items at the end.

Work in this order: each item needs the ones before it. Owners follow `CLAUDE.md`; contract
changes go through a contracts-only PR.

## 1. Robot profiles (T1)
Same as `docs/tracks/dominoes/T1-sim.md` item 2 and T7's open request. Replace the hardcoded
`PandaProfile` with a registry keyed by `RobotSpec.source`. A profile names:
- arm joints and actuators, gripper actuator, finger and pad bodies, hand and base bodies;
- home posture, TCP offset, gripper command range;
- joint limits.

The controller (IK, speed limits, gravity compensation) and event detection read only the
profile. Test: every existing Franka baseline is unchanged.

## 2. A second arm in sim (T1)
- Settle the Gen3's degrees of freedom first (6 or 7; see PROJECT.md). For 6-DoF, build the
  model from Kinova's URDF; menagerie's Gen3 is believed to be 7-DoF.
- Attach the menagerie `robotiq_2f85` at the flange; check its closed gap (the Panda pads left
  3 mm; see `closed_pad_gap`).
- The fallback in parallel, because it is cheap: `PandaLocked` (one joint fixed, tighter limits).
  It is a weak substitute, so label it in every result.
- Tasks: `kinova_hang_mug*.json` (and `panda_locked_*`), with the same scene and success, and the
  base placed so the stand is in comfortable reach. Test: each arm reaches the grasp and
  threading poses without `ik_failure`.

## 3. Per-robot parameters (contracts, then T2)
A sanctioned place for numbers that belong to one arm, so a parameter-only fix is visibly
robot-specific. The request is already in `HANDOVER.md`: an optional
`RobotSpec.params: dict[str, float]`. Skills read
`task.robot.params.get(name, class_default)`. Move `HangMug`'s arm-dependent constants
(`grasp_depth`, `wrist_home_yaw`, `wrist_limit`, approach offsets) behind it.

## 4. Naive seed skills (T2)
A handle-blind `HangMug` variant that grasps anywhere and turns nothing, so the general fixes
(grasp opposite the handle, plan backwards from the hanging pose, align with the peg axis) must
be discovered. Without this the library has little to scope. Record its success on each arm.

## 5. Scoped library (T3)
Extend `agents/library.py` (keep its version store):
- each accepted repair stores its diff against the parent, the robot it came from, and a
  **scope** (`general` / `robot:<name>`) with the evidence that decided it;
- **agent label**: the improver prompt asks for a scope and a reason, and the label is recorded
  even when overridden;
- **structural label**: a diff that touches only `RobotSpec.params` or class constants is
  robot-specific;
- **checking**: apply a candidate "general" fix to each other robot's current best skill and run
  the paired gate on that robot's gate seeds. Any significant loss makes it robot-specific.
  Record wins, losses and p per robot;
- **sharing**: building a starting skill for robot X = base skill + general fixes + X's own fixes.
  Diffs that no longer apply cleanly are recorded, not silently dropped.

## 6. Experiment runner (T3 + T4)
`skillforge experiment <config>`:
- runs the four conditions for robots A, B, C on one task with a fixed budget (rounds, $);
- logs rounds-to-target, cost, gate and held-out results, and every scope decision;
- runs the share-back step for negative transfer: robot A re-evaluated (paired, same seeds) after
  B's and C's fixes are shared;
- reports scope-label error: agent label vs checking result, as a confusion matrix.

Repeat each condition (at least 3 runs, more if the budget allows); the agent is not
deterministic. Results go in `results/<date>-scoped-<task>/` with a README that says what is
measured and what is not.

## 7. Second task
Repeat on dominoes (or another task) once 1-6 work on mugs.

## 8. Real arms (T6 + T7, after the course deliverable)
- Franka driver behind `real/core.py` `RobotIO` (`franka_ros2` or `libfranka`).
- An AprilTag mug pose: the skill needs the handle's yaw, and `perception/frames.py` maps one geom
  per object, so it sees only the mug's bottom disc.
- A task file matched to the measured mug and stand. The trial protocol goes in `HANDOVER.md`
  before the first scored trial.
- Then the Kinova, in spring.

## Notes
- `grasp_depth` 0.018 -> 0.028 from the 2026-11-25 mug forge round is the first scope test case:
  check it on the second arm.
- Keep each skill one file importing only `skillforge.*` and numpy, or the sandbox refuses it.
- The paired gate needs the same seeds on both versions, and the same seeds draw the same scene
  on every arm. Keep it that way when adding profiles.
