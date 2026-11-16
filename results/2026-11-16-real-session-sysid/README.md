# A dry-run session through the real-robot runner, and system ID on its logs (T7, sim only)

2026-11-16. **No hardware.** `python -m skillforge.real run ... --dry-run` drove a simulated
arm (`MujocoTwin`) through `RealEnv`, the safety limits and the session runner, exactly as a
real session would be driven, with true sim state standing in for the operator's label.

    MUJOCO_GL=egl python -m skillforge.real run franka_pick_cube_to_bin_hard \
      --skill-file results/2026-11-04-claude-code/hard/pick_place_0.2.2.py --seeds 0-19 --dry-run
    python -m skillforge.real sysid <session>
    python -m skillforge.real scoreboard --sessions <session>

- Task `franka_pick_cube_to_bin_hard`, seeds 0-19, tool speed limit 0.1 m/s, perfect perception.
- The twin applies each seed's physical randomization (latency 0-5 steps, gain 0.6-1.4, ...), so the logs have a known truth to check the fits against.

## Session

19/20 (Wilson 95%: 76%-99%), one timeout. The estimator's success call agreed with the label
in 20/20, which is trivial here because the estimator was given true poses. 9 trials had a
`safety:joint_speed` event (the wrist turning to the grasp angle at just over 1 rad/s).

## System ID against the twin's truth (`sysid.json`)

| Quantity | Fit | Truth | Verdict |
| --- | --- | --- | --- |
| Action latency, control steps | per trial, 0 to 5 | identical in 20/20 trials | recovered exactly |
| Tracking gain per step | 0.403 +- 0.003 | actuator gain scale 0.64 to 1.39 | **not identifiable** |
| Gripper close time | 0.29 +- 0.05 s (18 trials) | not known independently | plausible, unchecked |
| Held width | 35.9 +- 0.2 mm | 40 mm cube | 4 mm under: the pads sink into the cube in sim |

The gain result is the one to remember. The fit's gain does not move with the sim's
`actuator_gain_scale` (0.397 to 0.407 for scales 0.4 to 2.0), because the env scales the
position and velocity gains together and the tracking lag stays the same. Reading a scale off
that curve gives a median error of 0.15 on a range of 0.6-1.4, which is no information.
`propose_randomization` now refuses to touch the gain range when the curve is flat. A plain hold
test does not help either: the simulated arm is gravity compensated and holds with zero droop at
every scale (checked, 0.4 to 2.0). In this sim the gain scale only shows under load (contact,
payload), so identifying it would take a loaded test, not task logs.

## Dry run with depth perception in the loop (`depth_dry_run.json`)

The same path with T6's depth cameras instead of true poses, 200 held-out seeds (5000-5199),
0.1 m/s. True success is the twin's sim state; the estimator's call is what `RealEnv` reports.

| Perception | True success | Estimator said success | False success | Missed success |
| --- | --- | --- | --- | --- |
| True poses | 196 | 196 | 0 | 0 |
| Depth cameras, sim masks, clean | 196 | 197 | 1 | 0 |
| Colour segmentation, 3 mm depth noise | 197 | 198 | 1 | 0 |

- **At the real speed limit, depth perception cost nothing** on this task (196 and 197 against 196). At the sim's 0.5 m/s the same estimator cost up to 9 seeds in 200. Not traced; slower motion gives the pose filter more frames per centimetre, which is the likely reason.
- **The estimator's success call was wrong once in 200** in each camera setting, both times a false success. That is the error an operator's label exists to catch; 1 in 200 is too few to put a rate on.
- No visual randomization or calibration error was applied in these runs.

## Caveats

- A simulated arm. A real arm's tracking is not first order, so the latency fit has to be re-checked on real logs before it is trusted.
- One task, one skill, 20 trials.

## Files

`real.json` (the session sidecar), `report.json`, `sysid.json` (calibration, truth, gain curve),
`scoreboard.json` / `.html` (a dry-run row; no sim report was attached).
