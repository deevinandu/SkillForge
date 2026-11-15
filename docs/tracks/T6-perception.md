# T6 · Perception (Phase 3)

**Goal:** skills stop reading ground-truth object poses. A camera pipeline estimates them, in sim first and then on T7's real cameras, and the harness measures what that costs in success rate.

Today `MujocoEnv._observe` hands skills the true pose plus isotropic Gaussian noise (`obs_noise_std`). On the real robot there is no true pose, only pixels. T6 closes that gap in sim, where truth is still available to score against, so the error model and the success cost are measured before any hardware run.

## Deliverables
1. `perception/frames.py`: `CameraFrame(rgb, depth, K, T_world_cam, t)` in metres and the world frame, with quaternions (w, x, y, z). Add a `FrameSource` protocol with `capture(camera) -> CameraFrame | None`. Build `SimFrameSource(env: MujocoEnv)`, which renders RGB, depth and (sim only) a segmentation mask with `mujoco.Renderer`, and reads K from the camera's `fovy` and `T_world_cam` from `data.cam_xpos/xmat`. T7 implements the same protocol for its real cameras.
2. `perception/estimators.py`: a `PoseEstimator` protocol with `reset(task)` and `estimate(frames, obs) -> dict[str, Estimate]`, where `Estimate` is a pose, a confidence and a `found: bool`. Three implementations, in this order:
   - `OracleEstimator`: true pose plus configurable noise. It reproduces today's behaviour and is the reference point for every comparison.
   - `DepthEstimator`: classical and CPU-only. It segments the object (sim mask in sim; colour or depth-above-table on real), back-projects the depth pixels to a point cloud, and fits the known `ObjectSpec` geometry. Boxes use centroid plus PCA yaw, snapped to the box's symmetry. Cylinders and spheres use a centroid offset by the known radius, with no yaw. It must be deterministic. This is the estimator Phase 3 relies on.
   - `LearnedEstimator` (stretch, GPU, local RTX 5070 Ti): open-vocabulary detection (OWLv2 / Grounding DINO) from `ObjectSpec.name` or the task instruction, then SAM 2 masks, then depth fitting as above. Optionally add FoundationPose tracking with a mesh generated from `ObjectSpec`. First measure whether it fits in 12 GB and runs at a usable rate; if not, document that and stop.
3. `perception/env.py`: `PerceivedEnv(inner, estimator, cameras=("front", "wrist"), estimate_hz=None)` implementing `SimEnv`.
   - It steps `inner`, captures frames, and replaces `obs.objects` with estimates.
   - It never touches `StepInfo`: success is still judged on true sim state.
   - When an object is not found, it keeps the last estimate and appends `perception_miss:<obj>` to `StepInfo.events`.
   - `estimate_hz` below `control_hz` holds the last estimate between captures, which is how a slow real pipeline behaves.
   - It works with `evaluate`, `evaluate_parallel` and the sandbox, so registration has to be picklable/spawn-safe. Follow how T4's pool builds envs.
4. Visual domain randomization inside `PerceivedEnv`: light position and intensity, table and object colour/texture, and camera extrinsic jitter (a few mm, about 1°). Draw it from an RNG seeded by the `reset(seed)` value, so the same seed still gives the same episode. `RandomizationSpec` has no visual fields. Keep the ranges in `PerceivedEnv`'s config and log the values drawn alongside `randomization_sample()`.
5. `perception/error_model.py`: run an estimator over N seeds and record per-object position error, yaw error, bias, miss rate and latency against the true pose. Write a `PerceptionReport` (a model in `perception/`, not contracts). Then fit the closest `obs_noise_std` range, so the sim's noise axis is calibrated to real estimator error instead of guessed.
6. `perception/vlm_judge.py`: a success judge for the real robot, where no true state exists. It sends the final frame(s) and the task instruction to Claude (`claude-sonnet-5` by default, `claude-opus-5-5` optional) and returns `(success, reason)`. In sim it is scored only as a classifier against true success: a confusion matrix over 200+ episodes that include near-misses (object on the bin rim). It never decides success in sim. Log cost per call and add a `--budget-usd` cap. Key from `ANTHROPIC_API_KEY`.
7. Measurements to record in HANDOVER:
   - **Pose accuracy:** p50/p95 position error, yaw error and miss rate for `DepthEstimator` on each Franka task.
   - **Success cost:** the agent-hardened skill on `franka_pick_cube_to_bin_hard` under `OracleEstimator` vs `DepthEstimator`, on held-out seeds, as a paired comparison (T3's McNemar helper).
   - **Robustness:** one robustness sweep with perception in the loop.
   - **VLM judge:** its confusion matrix.

## Tests
- `SimFrameSource`: a known point back-projects to its true world position within 2 mm, and K/extrinsics agree with MuJoCo's own projection.
- `DepthEstimator` on box, cylinder and sphere: error within a few mm with no noise; box yaw is correct modulo its symmetry.
- `PerceivedEnv`: satisfies `SimEnv`; same seed gives the same episode; `StepInfo.success` matches the inner env exactly; a hidden object produces `perception_miss`; holds behave correctly under `estimate_hz`.
- `PerceivedEnv` + `OracleEstimator` with zero noise gives the same `EvalReport` as the bare env.
- The VLM judge runs offline against a fake client, with no network in CI.
- Rendering tests skip cleanly when GL is unavailable, the way T4's video tests do.

## Contract requests to file in HANDOVER (work around them meanwhile)
- **Camera frames:** `SimEnv.render` returns RGB only. Request an optional `capture(camera) -> CameraFrame | None` on `SimEnv`, with `CameraFrame` moved into contracts, so T6 and T7 share one type without cross-track imports. Meanwhile `SimFrameSource` reads `MujocoEnv.model/data` directly.
- **Failure tag:** request a `perception` `FailureTag`, so `perception_miss` can outrank `grasp_miss` in `_classify` (a T4 change once the tag exists). Meanwhile these episodes are tagged by their physical symptom, and the miss shows up in `events`.
- **Noise shape:** `obs_noise_std` is isotropic position noise only. Request yaw noise, a per-episode bias and a dropout probability, if the error model shows that real errors look like that.

## Notes
- Rendering every control step is expensive (osmesa is far slower than egl). Measure steps/s with perception on. Use `MUJOCO_GL=egl` locally, and use `estimate_hz` to keep held-out evals of 200+ seeds practical.
- The skills already low-pass filter the object pose (`skills/primitives.py` `Context`). Measure with that filter as it is. Don't retune it; that is T2's call.
- Keep heavy dependencies (torch, detection models) behind an optional `perception` extra, with lazy imports. `DepthEstimator` needs only numpy.
- Model weights go in `assets/cache/`, never in git.
- Be explicit about what is measured where. Everything in this track is sim until T7 points it at a real camera.

## Don't
- Don't change contracts, skills (T2) or the env (T1). If `MujocoEnv` needs a hook (e.g. segmentation IDs), ask T1 in HANDOVER.
- Don't let any estimate or the VLM judge decide sim success.
