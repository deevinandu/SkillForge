# T6 perception: cylinder follow-up (2026-11-15)

Everything here is **sim only**, with the same idealisations as `../2026-11-15-perception/`
(the renderer's own segmentation masks, perfect calibration, no visual randomization).
Same seeds (held-out 5000-5199), same agent-written skills, same commands.

The first run lost 14 seeds on the cylinder task (197 with the env's poses, 183 through the depth
cameras). The first guess was the in-hand carry, which assumes the can does not move in the
fingers. Tracing failures showed the carry was only part of it. Three things were measurably wrong:

1. **Mediocre fits were trusted just before the grasp.** As the open fingers come down over the
   can they hide part of it. Fits scoring 0.8 or more were within about 1 mm; fits scoring 0.3-0.6
   were about 5 mm off, and the last of those was held through the grasp. The confidence needed to
   count as a model fit is now 0.7 (was 0.3).
2. **A can on its side had no model.** After a failed grasp the can usually falls over. It then
   got a coarse centre-of-visible-points estimate, 8 mm off at the median and 20 mm at p95, and
   the estimate only updated after a 3 cm jump. `fit_cylinder` now also fits a lying cylinder
   (circle fit in the cross-section), and a coarse estimate is accepted at once whenever the
   gripper is more than 12 cm away, since the fingers cannot then be what hides the object.
3. **In the hand, the carry drifted when the can slid.** `fit_cylinder` now fits the side wall as
   a circle seen from above, which needs only an arc of the wall below the fingers. While the
   object is held, that horizontal position replaces the carried one; height is still carried.

## Estimate error on the cylinder task, before and after (40 episodes, by situation)

| Situation | Before (median / p95) | After (median / p95) |
| --- | --- | --- |
| Upright, gripper open | 0.9-5.5 / up to 7.5 mm | 0.5 / 0.5 mm |
| Upright, in the gripper | 7.9 / 32.5 mm | 1.1 / 9.7 mm |
| Lying, gripper open | 18.7 / 141 mm | 0.7 / 64 mm |

The remaining p95 for a lying can is episodes where it rolled out of the cameras' view.

## Success (200 held-out seeds each)

| Task | Env poses | Depth, first run | Depth, now | Depth + noise, now | Paired now, env vs depth (env only / depth only) |
| --- | --- | --- | --- | --- | --- |
| `franka_pick_cylinder_to_bin` | 197 | 183 | 190 | 190 | 9 / 2 |
| `franka_pick_cube_to_bin_hard` | 195 | 194 | 194 | 191 | 5 / 4 |
| `franka_pick_ball_to_bin` | 180 | 193 | 193 | 193 | 3 / 16 |

- **Cylinder:** half the gap is closed (14 seeds lost, now 7). The cube and ball are unchanged.
- **What is still unexplained:** 9 seeds fail through the cameras that pass with the env's poses.
  Two things were tested and neither accounts for it:
  - *The can's rotation.* The env reports the can's true yaw, which no camera can see on a plain
    cylinder; the estimator reports zero. Zeroing only the yaw in the env's own poses costs 6 seeds
    (197 to 191), because the skill picks its grasp angle from it. But giving the depth estimator
    the true yaw does not help (187 vs 190, within noise).
  - *Depth noise.* No effect on the total (190 both ways).
  The likeliest remainder is the skill itself: it was hardened by the agent on the env's noisy
  poses, and some of what it learned does not transfer to a different error profile. That is a
  guess. The test would be to run the forge loop with `make_depth_env`, which has not been done.
- The ball result still is not perception beating truth; see the first README.

## Pose error over whole episodes (30 episodes each)

| Task | Median | p95 | Steps with a confident fit | Error on those (median / p95) | Error on other steps (median / p95) | Equivalent `obs_noise_std` |
| --- | --- | --- | --- | --- | --- | --- |
| cylinder | 0.8 mm (was 6.7) | 11.3 mm (was 32.7) | 27% | 0.5 / 2.3 mm | 1.1 / 15.7 mm | 0.3-7.2 mm |
| cube (hard) | 0.9 mm (was 1.4) | 16.1 mm | 57% | 0.6 / 1.8 mm | 3.7 / 20.8 mm | 1.3-5.8 mm |
| ball | 0.3 mm | 0.4 mm | 99% | 0.3 / 0.4 mm | 15.2 / 56.3 mm | 0.2-0.6 mm |

## Files
`<task>_compare.json` and `<task>_error.json`, as in the first results folder.
