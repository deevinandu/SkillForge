# T1 · Sim and Scenes (Phase 1)

**Goal:** a MuJoCo backend that turns any `SceneSpec` into a physics env implementing `contracts.SimEnv`.

## Deliverables
1. `sim/mujoco_env.py`: `MujocoEnv(task)`.
   - Robot from `RobotSpec.source`: `menagerie:franka_emika_panda` first (clone google-deepmind/mujoco_menagerie into `assets/cache/`, ignored by git; a `scripts/fetch_assets.py` does it). Then `mjcf:<path>`, then `urdf:<path>`.
   - Build the scene with `mujoco.MjSpec`: table, objects from `ObjectSpec`, regions as visual-only sites, a `tcp` site between the fingertips if the model lacks one.
   - Low-level controller for `Action(mode="ee")`: damped-least-squares differential IK on the `tcp` site → joint position targets, joint-limit clamping, a max EE speed. `mode="joint"` passes targets through. Gripper 0..1 maps to finger actuator range.
   - Physics at its own timestep, `control_hz` control steps (substeps = round(1 / (hz · dt))).
   - Every `RandomizationSpec` field applied for real: object xy/yaw jitter, friction and mass scale on objects, actuator gain scale, obs noise on object positions, action latency queue. `randomization_sample()` returns what was drawn.
   - `StepInfo.events`: `grasp:<obj>` when both fingers touch it, `grasp_slip` when contact is lost while the gripper is closed, `collision:<a>-<b>` for arm links hitting table or objects other than fingertip contacts, `ik_failure` when DLS error stays above tolerance.
   - `render(camera)`: offscreen `mujoco.Renderer`, default camera + `wrist` camera; return None if no GL.
2. Register it in `sim.make_env` for `menagerie:`/`mjcf:`/`urdf:` sources.
3. `tasks/franka_pick_cube_to_bin.json` (and a lift task) with realistic randomization ranges.
4. Tests: model loads; home pose holds still; EE reaches a target within 5 mm; same seed gives the same episode; each randomization field changes the model; success predicate matches ground truth; `PickPlace` from T2's current version gets > 0% on the Franka task (it may be low; that's what T3 fixes).

## Don't
- Don't change contracts. Don't write skills (T2). Keep the env fast: aim for > 2,000 control steps/s without rendering on one CPU core.
