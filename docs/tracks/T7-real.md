# T7 · Real Robot (Phase 3)

**Goal:** a `SimEnv` that drives a real arm through ROS 2, so the same harness, skills and reports run on hardware. Then calibrate the twin from real logs and keep a sim2real scoreboard per skill.

**Phase 3 gate (this track measures it):** a sim-hardened skill succeeds on a real arm, measured. Fix the protocol before the first scored trial, and write it in HANDOVER first:
- 20 or more trials;
- object placements drawn from the task's randomization by seed;
- success labelled by a person;
- the result reported as n/N with a Wilson 95% CI, next to the sim prediction on the matched task.

## Before you start: the human answers these (record them in HANDOVER)
- **Which arm and gripper, and where?** The default plan is an xArm 6 with the xArm Gripper, and a Franka later.
- **Which cameras, and where are they mounted?** For example a fixed RealSense D435 plus a wrist D405.
- **How much arm time is available, and is someone always next to the e-stop?** No hardware motion happens without a person at the e-stop.
- **The local box:** ROS 2 Humble is installed at `/opt/ros/humble`. Confirm that the driver (`xarm_ros2` humble branch, or `franka_ros2`) builds there.

## Deliverables
1. **Robot model in sim (needs T1).** `MujocoEnv` hardcodes `PandaProfile`.
   - Ask T1 in HANDOVER for a robot-profile registry and an xArm 6 profile. As far as I know menagerie ships an xArm 7 and a Lite 6, not an xArm 6; check. If it has no xArm 6, use `urdf:` from the `xarm_ros2` description.
   - Add a matched task, `tasks/xarm6_pick_cube_to_bin.json`, using the measured table, bin and camera positions from the real cell.
   - The goal is to run the identical skill in sim and on the real arm.
2. **`real/core.py`: the ROS-free core**, all testable without hardware.
   - **`RealEnv(task, robot: RobotIO, frames: FrameSource, estimator)`** implements `SimEnv`.
   - **`RobotIO`** is a protocol: `read_state()`, `send_joint_targets(q, gripper)`, `stop()`, `home()`.
   - **IK:** `Action(mode="ee")` goes through the same DLS IK as sim, run on a kinematic-only MuJoCo model of the same robot. Sim and real then share one controller and the controller is not a source of sim2real gap. `mode="joint"` passes through.
   - **`reset(seed)` semantics on hardware:**
     1. The arm homes.
     2. The seed draws an object placement (xy and yaw) from the task's randomization.
     3. The operator places the object there, guided by a live camera overlay that shows the target pose against T6's estimate, and confirms.
     4. The env records the drawn values together with the measured placement. Physical parameters (friction, mass, gain) can't be drawn on hardware; `randomization_sample()` reports them as nominal and says so.
   - **Observations:** `step` sends targets at `control_hz` and returns an `Observation`. Joint state comes from the robot, and object poses come from T6's estimator on real frames. `contact_force` comes from the joint-torque estimate if the arm provides one.
   - **`StepInfo.success`** comes from T6's estimate, and only stops the episode early. It is never the recorded result (see 5).
3. **`real/safety.py`: safety in the env, never in the skill.** Agent-written skills return only `Action`s, and `RealEnv` enforces:
   - **Workspace:** a box derived from the table and regions. Out-of-box targets are clamped and logged as `safety:workspace`.
   - **Speed limits:** EE and joint speed well below sim. Start at 0.1 m/s; 0.25 m/s only after 10 clean trials.
   - **Force limit:** stop if `contact_force` exceeds it, logged as `safety:force`.
   - **Watchdog:** if no action arrives for 2 control periods, the arm holds.
   - **Exceptions:** an exception in the skill or the estimator means stop and hold.
   - **What may run:** only skill versions accepted into T3's library, and only after the human approves that version by name. Never run a candidate straight out of an improver.
4. **`real/ros/`: the ROS 2 layer.** A `rclpy` node implementing `RobotIO` and `FrameSource`: the xArm driver's servo/joint-command interface, camera topics, and TF for camera extrinsics. Import `rclpy` lazily; nothing outside `real/ros/` imports it.
   - **Dry-run mode:** `--dry-run` sends nothing to the robot, steps a MuJoCo twin with the same commands, and shows both. Run every new skill version dry first.
   - **Logging:** record a rosbag2 per session under `runs/real/` (git-ignored).
5. **`real/runner.py` and the CLI:** `skillforge real run tasks/xarm6_....json --skill <library version> --seeds 0-19 [--dry-run]`.
   - **Before each trial:** run the placement step from 2.
   - **After each trial:** the operator labels the result in the terminal: success y/n, a failure tag from `FailureTag`, and a note.
   - **What gets written:** a normal `Episode` (operator label = `success`), plus a sidecar `real.json` with the auto (estimator) success, the VLM-judge verdict (T6), the measured placement and the safety events. `.npz` arrays use the T5 dataset format, so real episodes can become training data.
   - **Video:** from the fixed camera.
6. **`real/sysid.py`: system ID from logs.** From commanded vs measured joint trajectories, fit:
   - action latency (cross-correlation, in control steps);
   - tracking gain (compare it to `actuator_gain_scale`);
   - gripper close time and measured width per object.

   Output `runs/real/calibration.json` and a proposed `RandomizationSpec` (centres from the fit, ranges from its spread) for the matched task. Record the fit and its uncertainty in HANDOVER before changing any task file.
7. **`real/scoreboard.py`: a sim2real scoreboard.** Per skill version on the matched task, show:
   - sim held-out success with its CI;
   - real n/N with its CI;
   - real failure tags;
   - estimator-vs-human and VLM-vs-human agreement;
   - links to videos.

   Write JSON; ask T4 to render it in the dashboard (T7 may write a minimal static HTML meanwhile).
8. **Close the loop.** Every real failure becomes a sim test case. Write a pinned task variant (`lo == hi` at the measured placement and the calibrated parameters) as a new file under `tasks/`, plus a regression baseline in T4's format. Real failures thus become regression tests and inputs for the improver.

## Tests (no hardware, no ROS in CI)
- **`FakeRobotIO`:** a fake robot that tracks targets with a set latency and gain. With it:
  - `RealEnv` satisfies `SimEnv`;
  - a skill runs end to end through `evaluate`;
  - a dry run against `MujocoEnv` matches a sim run with the same seed.
- **Safety:**
  - an out-of-workspace target is clamped;
  - over-speed is limited;
  - the force threshold stops the arm;
  - a missed action trips the watchdog;
  - an exception in the skill holds the arm.
- **sysid:** recovers a known latency and gain from synthetic logs.
- **Scoreboard and close-the-loop:** the scoreboard renders from a fixture, and a failure turns into a valid pinned `TaskSpec`.
- **Imports:** `real/ros/` tests skip when `rclpy` is missing.

## Contract requests to file in HANDOVER (work around them meanwhile)
- `Episode` has no field for a human label or for an auto vs labelled success. Request optional `label_source` and `auto_success` fields. Meanwhile use the sidecar `real.json`.
- `CameraFrame` / `capture()`: the same request as T6's. Share one type.
- `RandomizationSpec` can't pin an exact placement for a regression case except through `lo == hi` on a magnitude, so it can't express an exact signed xy offset. Request an optional per-object pose override on `TaskSpec` for replaying real failures. Meanwhile write the measured pose as the object's nominal `pose` with zero jitter.

## Notes
- Work in this order. 1–3 and the fakes need no hardware and can run alongside T6. 4–5 need the robot. 6–8 need logged sessions.
- First hardware session: no object, a dry run, then live motion at 0.1 m/s. Then a foam cube, then the real cube.
- The real result will be small-n. Report the CI and don't round 17/20 up to "works".
- Hardware drivers and their build output live outside the repo or in `runs/`. Only `real/` code and configs go in git.

## Don't
- Don't change contracts, skills (T2), the sim env (T1) or perception (T6). Request changes in HANDOVER.
- Don't move the arm without a person at the e-stop, and don't run unreviewed agent code on hardware.
- Don't overwrite an existing task's randomization from sysid. Add a new task file.
