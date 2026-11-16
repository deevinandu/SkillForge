# Dry run of the real-robot env at the real speed limits (T7, sim only)

2026-11-15. **No hardware was involved.** `RealEnv` drove a simulated arm (`MujocoTwin`) through
the same path a real arm would be driven: skill -> safety limits -> controller -> joint targets.
Object poses were the sim's true poses (perfect perception), so this isolates the control path
and the speed limits.

- Task: `franka_pick_cube_to_bin_hard` with observation noise set to 0 (the dry run sees true poses, so the sim reference does too). All other randomization as the task has it.
- Skill: `results/2026-11-04-claude-code/hard/pick_place_0.2.2.py`. Seeds: held-out 5000-5199 (200).
- Joint speed limit 1 rad/s, tool rotation limit 1 rad/s, except the 0.5 m/s row, which uses the sim's own speeds.

| Setting | Tool speed limit | Success | Median steps (successes) | Episodes with `safety:joint_speed` |
| --- | --- | --- | --- | --- |
| Plain sim (`MujocoEnv`) | 0.5 m/s | 199/200 | 91 | n/a |
| Dry run, sim speeds | 0.5 m/s | 199/200 | 91 | 0 |
| Dry run | 0.25 m/s | 199/200 | 135 | 128 |
| Dry run | 0.1 m/s | 196/200 | 263 | 92 |

## Reading

1. **The real-robot path reproduces the sim.** At the sim's speeds the dry run gives the same count and the same median length as plain sim. It is not step-identical: `MujocoEnv` reports the tool pose one physics substep stale and `RealEnv` computes it from the measured joints (0.5 mm apart while moving), so a few episodes differ by a step.
2. **At 0.1 m/s the skill uses most of its time budget.** Median 263 of the task's 300 steps (15 s). 196/200 against 199/200 is within noise at n=200, but the margin is thin: anything that slows the real arm further turns into timeouts. For the first hardware trials either raise `max_steps` in the matched task or expect timeouts that are not the skill's fault.
3. **The 1 rad/s joint limit is hit in about half the episodes.** The limiter scales the step and reports it; it did not cost successes here. Which motion triggers it was not traced (likely the wrist turning to the grasp yaw).

## Caveats

- Perfect perception and a simulated arm. This says nothing yet about a real arm's tracking, latency or gripper.
- One task, one skill.
