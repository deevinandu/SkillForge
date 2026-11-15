# Robustness sweep with perception in the loop (T6, sim only)

2026-11-15. T4's robustness sweep on `franka_pick_cube_to_bin_hard`, run three times with the same
skill and the same seeds: with the env's own poses, with clean depth perception, and with the
hardest perception setting in sim. Nothing here ran on hardware.

- Skill: `results/2026-11-04-claude-code/hard/pick_place_0.2.2.py` (agent-written, hardened on the env's own poses).
- Seeds: held-out 5000-5099, **100 per point**. A 95% interval at 90% success is roughly 83-95%, so differences under about 7 points between two cells are not evidence of anything.
- Each axis is pinned to one value at a time; the other axes keep the task's ranges (`others="task"`).
- Settings: `true` = true pose plus the task's obs noise; `depth` = overview + wrist depth cameras, sim masks, clean depth, perfect calibration; `dr-color` = colour segmentation, 3 mm depth noise, visual randomization incl. calibration error.

Command (one per setting):

    MUJOCO_GL=egl python -m skillforge.perception robustness franka_pick_cube_to_bin_hard \
      --skill results/2026-11-04-claude-code/hard/pick_place_0.2.2.py --seeds 5000-5100 --envs depth --workers 12

## Success per point (out of 100)

| Axis | Values | true | depth | dr-color |
| --- | --- | --- | --- | --- |
| object_xy_jitter (m) | 0, .02, .04, .06, .08, .10 | 99 98 98 99 95 96 | 100 100 98 95 96 91 | 98 93 96 90 90 87 |
| object_yaw_jitter (rad) | 0, .31, .63, .94, 1.26, 1.57 | 98 99 98 93 96 94 | 96 98 97 94 98 98 | 91 93 93 91 90 89 |
| friction_scale | .25, .5, .75, 1, 1.5, 2 | 71 97 97 97 98 97 | 69 96 97 95 97 97 | 68 89 86 92 94 90 |
| mass_scale | .5, 1, 1.5, 2, 3, 4 | 97 96 93 81 29 6 | 93 96 96 80 33 3 | 91 91 93 81 32 6 |
| actuator_gain_scale | .25, .5, .75, 1, 1.5, 2 | 36 94 99 98 87 74 | 36 92 99 90 90 93 | 36 92 99 84 84 89 |
| obs_noise_std (m) | 0, .002, .004, .006, .008, .010 | 99 97 96 98 95 95 | 97 97 97 97 97 97 | 89 89 89 89 89 89 |
| action_latency_steps | 0, 1, 2, 3, 4, 6 | 97 96 96 95 96 95 | 96 97 95 96 96 95 | 89 91 90 89 90 91 |
| stress x2 (every range widened) | | 37 | 60 | 55 |

Depth-noise curve (`depth` setting, std at 1 m, growing with z^2): 0 mm 97, 1.5 mm 97, 3 mm 94, 6 mm 89, 12 mm 96.

## What it shows

1. **The physical breaking points do not move.** Friction 0.25x (68-71), mass 2x/3x/4x (80-81 / 29-33 / 3-6) and gain 0.25x (36 in all three) fail the same way whether the skill reads true poses or camera estimates. These are grasp and actuation limits, not perception.
2. **`obs_noise_std` is inert under perception**, as expected: a perceived env throws the env's object poses away, so the curve is flat (97, and 89 for `dr-color`). The camera's counterpart is the depth-noise curve.
3. **The stress result is higher with perception (60 and 55 against 37), and observation noise is why.** The stress preset widens obs noise to 0-3 cm, which perception never sees. Rerunning the stress eval with the env's own poses but obs noise left at the task's range gives 58, and with no obs noise 63 (`stress_obs_noise_split.json`). So perception is not more robust; it is spared one stressor. Stress numbers with and without perception are not comparable until the stress preset also widens a camera axis.
4. **`dr-color` costs a roughly constant 5-9 points** on the axes that do not break the skill, in line with the 200-seed result (186 against 194). It grows with object position jitter (87 at 10 cm against 96): cubes further from the nominal position are seen worse or placed worse; not traced.
5. **High actuator gain is better with perception** (2x: 74 true, 93 depth, 89 dr-color). Not traced. A guess: the env's per-step pose noise excites an over-driven arm and the estimator's steadier poses do not. Untested.

## Not explained

- The depth-noise curve is not monotonic (89 at 6 mm, 96 at 12 mm). At n=100 the two intervals overlap, so this may be noise; it was not rerun.
- Item 5 above, and why position jitter hurts `dr-color` more.

## Caveats

- One task, one skill, 100 seeds per point. The cylinder and ball were not swept.
- Sim only; the limits of the visual model in `results/2026-11-15-perception-visual/README.md` apply.
- The `dr-color` sweep took 1 h 44 min on 12 workers against 30 min for `depth`. GPU memory sat at 11.5 of 12 GB during it: each sweep point is a new task, so every worker builds a new env and renderer and keeps up to 8 cached. Probably the cause of the slowdown; not confirmed.

## Files

- `franka_pick_cube_to_bin_hard_{true,depth,dr-color}.json`: `{reports: {<setting>: RobustnessReport}, depth_noise}`.
- `stress_obs_noise_split.json`: the three stress evals of item 3 (env's own poses).
