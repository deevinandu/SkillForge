# T6 perception: visual randomization and colour segmentation (2026-11-15)

Everything here is **sim only**. Held-out seeds 5000-5199, the agent-written skills from earlier
results, 200 episodes per cell.

## What was added
- **Visual randomization** (`VisualRandomization`, drawn once per episode from the env seed):
  light intensity and direction, table colour, object brightness and colour, and a **camera
  calibration error** (by default up to 3 mm per axis and 0.5 degrees per camera: the image is
  rendered from a slightly wrong pose while the frame reports the calibrated one). Textures are
  not randomized. Nothing is written to the env's model.
- **Colour segmentation measured at scale** (`segment="color"`): the estimator is told each
  object's nominal colour and has to find it in the image, instead of using the renderer's masks.
- Settings, by `--envs` name: `depth` (masks, clean), `noisy` (masks + depth noise), `color`
  (colour, clean), `dr` (masks + randomization; only the calibration error can matter to masks),
  `dr-color` (colour + depth noise + randomization: the hardest setting here).

```bash
MUJOCO_GL=egl python -m skillforge.perception compare <task> --skill <skill.py> --seeds 5000-5200 --workers 12 --envs true,depth,color,noisy,dr,dr-color
MUJOCO_GL=egl python -m skillforge.perception calibration franka_pick_cube_to_bin_hard --skill <skill.py> --seeds 5000-5200 --workers 12 --rot-deg 0.25,0.5,1.0,2.0
```

## Success, final code (200 held-out seeds each)

| Setting | Cube (hard) | Cylinder | Ball |
| --- | --- | --- | --- |
| Env poses (true pose + the task's obs noise) | 195 | 197 | 180 |
| `depth`: masks, clean | 194 | 189 | 193 |
| `noisy`: masks + depth noise | 191 | 190 | 193 |
| `color`: colour, clean | 191 | 185 | 193 |
| `dr`: masks + randomization | 194 | 193 | 190 |
| `dr-color`: colour + noise + randomization | 186 | 192 | 191 |

- Differences of 3-4 seeds between cells are within run-to-run variation at n = 200 (the 95%
  interval on 190/200 is roughly 91-97%). Read the table for large effects only.
- **Cube:** the hardest setting costs 9 seeds against env poses (11 lost, 2 gained), 12 of the 14
  failures tagged `wrong_placement`.
- **Cylinder:** no measurable cost from randomization. The gap to env poses (5-12 seeds) is the
  one already reported in `../2026-11-15-perception-cylinder/`.
- **Ball:** no measurable cost. (Depth beats env poses on the ball because the env's poses carry
  the task's observation noise; see the first results README.)

## Two faults this exposed, both fixed

The first full run of `dr-color` scored **173 on the cube and 150 on the ball**.

1. **A table the same colour as the object** (cube 173 -> 186). The randomized table is reddish
   in about 11% of episodes; 7 of the first 20 cube failures had one. Colour segmentation then
   also selects table pixels, and a height cut just above the table does not remove them once
   depth noise lifts some above it. Lighting/colour changes alone cost 2 seeds and depth noise
   alone 7, but together 16. `DepthEstimator` now finds the object by its upper part, which no
   table pixel reaches, and keeps only points within the object's own size of it
   (`cube_decomposition.json` is the factor-by-factor run after this fix).
2. **The fit-confidence score was too strict for small round objects** (ball 150 -> 191). On the
   ball, depth noise alone and lighting/colour alone cost nothing (193, the same seeds), and
   calibration error alone cost 7-18 seeds depending on the draw. Calibration error *with* depth
   noise cost 43, with masks or with colour alike (`ball_interaction_before_fix.json`). Cause,
   measured on the approach: the sphere fit's confidence fell below the 0.7 bar on 42% of steps
   with calibration error and 64% with noise added, so a usable fit was replaced by the coarse
   centre-of-visible-points estimate, and median error went from 0.3 mm to 7 mm. The score now
   tolerates up to 1 cm of RMS residual on small radii before reaching zero.

## Camera calibration error alone (cube, masks, clean depth, 3 mm translation error per axis)

| Max rotation error per camera | 0 (no error at all) | 0.25 deg | 0.5 deg | 1 deg | 2 deg |
| --- | --- | --- | --- | --- | --- |
| Successes of 200 | 194 | 190 | 188 | 191 | 187 |

There is no clean trend: any calibration error costs about 3-7 seeds, and 2 degrees is not
clearly worse than 0.25. One reason is that the rotation angle is drawn uniformly up to the
maximum, so most episodes get much less than it. The estimator also fuses two cameras, so a
skill sees an average of two errors until the wrist camera's close-up points dominate.

## What this does and does not show
- It shows the pipeline survives randomized lighting, colours and a modest calibration error in
  sim, after two real faults were found and fixed. Finding those was the point of the exercise.
- It does not show real-camera performance. Still idealised or missing: depth noise is
  independent per pixel (a real sensor's error is correlated), no textures or clutter, no motion
  blur, lens distortion or exposure changes, one object per scene, and the colour segmenter is
  told the object's exact nominal colour.
- The 0.7 confidence bar and the new 1 cm floor were both chosen from data on these three tasks.
  They are not validated on other objects.

## Files
`<task>_compare.json` and `franka_pick_cube_to_bin_hard_calibration.json` are from the final
code. `cube_decomposition.json` is after fix 1 (fix 2 does not touch boxes).
`ball_decomposition_before_fix.json` and `ball_interaction_before_fix.json` are the runs that
located fault 2, taken before it was fixed.
