# T6 perception: first measurements (2026-11-15)

Everything here is **sim only**. No real camera has been used.

What "depth perception" means in these runs: two simulated RGB-D cameras (a fixed `overview`
camera 0.9 m from the workspace and the `wrist` camera), 640x480, **the renderer's own
segmentation masks**, perfect camera calibration, and no visual randomization. `DepthEstimator`
fits the known object shape to the fused point cloud. Real cameras will be worse on every one of
those points, so read these numbers as a best case for this estimator.

Commands (held-out seeds 5000-5199, the same seeds the earlier results use):

```bash
MUJOCO_GL=egl python -m skillforge.perception compare <task> --skill <skill.py> --seeds 5000-5200 --workers 12 --noisy
MUJOCO_GL=egl python -m skillforge.perception error   <task> --skill <skill.py> --seeds 5000-5030
```

Skills: the agent-written versions from earlier results (`2026-11-04-claude-code/hard/pick_place_0.2.2.py`,
`2026-11-05-claude-code-repeats/cylinder_r1/pick_place_0.2.1.py`, `.../ball_r1/pick_place_0.2.1.py`).

## Success with and without perception (200 held-out seeds each)

| Task | Env poses | Depth | Depth + noise | Paired, env vs depth (env only / depth only) |
| --- | --- | --- | --- | --- |
| `franka_pick_cube_to_bin_hard` | 195 (97.5%) | 194 (97.0%) | 189 (94.5%) | 4 / 3 |
| `franka_pick_cylinder_to_bin` | 197 (98.5%) | 183 (91.5%) | 182 (91.0%) | 16 / 2 |
| `franka_pick_ball_to_bin` | 180 (90.0%) | 193 (96.5%) | 193 (96.5%) | 3 / 16 |

- **Env poses** is the bare `MujocoEnv`: true pose plus the task's own Gaussian `obs_noise_std`.
  It is not noise-free. The oracle wrapper reproduces it exactly on all three tasks (same
  episodes, seed for seed), which checks that the wrapper itself changes nothing.
- **Depth + noise** adds independent per-pixel depth noise (3 mm std at 1 m, growing with z squared).
- **Cube:** no measurable cost (4 seeds lost, 3 gained).
- **Cylinder:** a real cost, 16 seeds lost against 2 gained, 13 of the failures tagged `grasp_slip`.
  The can is hidden by the fingers for most of the episode (a confident fit on only 23% of steps),
  so the skill runs on the in-hand carry, which assumes the can does not move in the grasp. It does.
- **Ball:** depth is *better* than the env's poses. That is not perception beating the truth: the
  task injects observation noise into the env's poses, and the simulated depth pipeline is
  cleaner than that noise (0.3 mm median error). It says the task's noise range is pessimistic
  for an idealised camera, nothing more.

## Pose error against true sim state (30 episodes each, every control step)

| Task | Median | p95 | Max | Steps with a confident fit | Error on those (median / p95) | Error on other steps (median / p95) | Equivalent `obs_noise_std` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cube (hard) | 1.4 mm | 16.2 mm | 40.8 mm | 65% | 0.7 / 5.1 mm | 6.5 / 22.0 mm | 1.6-5.7 mm |
| cylinder | 6.7 mm | 32.7 mm | 216.7 mm | 23% | 0.9 / 5.5 mm | 8.0 / 32.9 mm | 2.6-11.2 mm |
| ball | 0.3 mm | 0.4 mm | 75.0 mm | 100% | 0.3 / 0.4 mm | 33.0 / 72.0 mm | 0.2-0.6 mm |

- Cube yaw error (modulo 90 degrees, cube upright): median 0.2, p95 1.1 degrees.
- "Other steps" are the ones where the object is hidden or does not look like its upright
  shape: the estimate is carried with the gripper, held, or a coarse centre of the visible points.
- "Equivalent `obs_noise_std`" is the isotropic Gaussian std with the same RMS error, 5th to 95th
  percentile over episodes. The hard cube task's range is 0-15 mm, so on these numbers it is wider
  than this estimator's error. But the real error is not Gaussian: it is near zero when the object
  is visible and centimetres when it is in the hand, which a single std does not describe.
- The large maxima are single episodes where the object left the cameras' view or was lost.

## Not done yet
Visual domain randomization, the learned (GPU) estimator, the VLM success judge, a robustness
sweep with perception in the loop, colour segmentation at scale (it has one test), camera
calibration error, and anything on a real camera.

## Files
`<task>_compare.json`: the four `EvalReport`s and the paired counts. `<task>_error.json`: the `PerceptionReport`.
