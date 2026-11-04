# Claude Code improver: first runs (2026-11-04)

A headless Claude Code agent (`--improver claude-code`, model `claude-opus-5-5`) improved scripted
pick-and-place skills for a simulated Franka Panda. It needed no human edits, and the gains hold
on seeds that no part of the loop ever used. **Everything here is measured in simulation
(MuJoCo 3.17). Nothing has run on hardware.**

## How the numbers were made
- **Gate seeds 0–99.** The forge loop scores every version on these and keeps a version only if
  its success rate is at least the lower 95% confidence bound of the best version so far.
- **Dev seeds 100000+.** The agent may evaluate its own edits on these, through `./dev_eval` in
  the sandbox.
- **Held-out seeds 5000–5199.** Neither the loop nor the agent ever used these. Every row below
  was re-measured after the T1 physics fixes (PR #7).
- Success is judged on true simulator state. CIs are Wilson 95%.

## Hard task: `tasks/franka_pick_cube_to_bin_hard.json` (seed: hand-written PickPlace v0.2)

| Version | Author | Gate 0–99 | Held-out 5000–5199 | 95% CI (held-out) | Agent cost |
| --- | --- | --- | --- | --- | --- |
| 0.1.0 | hand-written | — | 159/200 (79.5%) | 73.4–84.5% | — |
| 0.2.0 | hand-written (seed) | 69/100 | 152/200 (76.0%) | 69.6–81.4% | — |
| 0.2.1 | agent, round 1 | 73/100 | 153/200 (76.5%) | 70.2–81.8% | $0.70 |
| 0.2.2 | agent, round 2 | 95/100 | 195/200 (97.5%) | 94.3–98.9% | $0.85 |

- **Round 1 (0.2.1)** removed the table collisions. It grasps no lower than the cube's nominal
  resting height and averages noisy readings. Held-out success did not move, though: timeouts
  replaced the collisions. The gate accepted it because 73 was within the seed's confidence
  interval, not because it was better.
- **Round 2 (0.2.2)** found the real cause. Observation noise of 1–1.5 cm made a single-sample
  slip check fire falsely, so the skill kept re-picking until it ran out of time. A slip monitor
  on an 8-sample mean fixed it.

## Original task: `tasks/franka_pick_cube_to_bin.json` (seed: hand-written PickPlace v0.1)

This run used the pre-PR #7 physics. The gate column is from that run; the held-out column was
re-measured with the current physics.

| Version | Author | Gate 0–99 (old physics) | Held-out 5000–5199 (current physics) | Agent cost |
| --- | --- | --- | --- | --- |
| 0.1.0 | hand-written (seed) | 80/100 | 181/200 (90.5%, CI 85.6–93.8%) | — |
| 0.1.1 | agent, round 1 | 99/100 | 200/200 (100%, CI 98.1–100%) | $0.51 |
| 0.2.0 | hand-written | — | 200/200 (100%, CI 98.1–100%) | — |

- **0.1.1** turns the gripper to the cube's observed yaw (mod 90°), smooths the object position,
  limits carry speed and re-grasps after a slip.
- Under the current physics, which has a stronger, more realistic grip, this task no longer
  separates the good skills from each other. Use the hard task.

## Caveats
- The gate rate is optimistic, because the loop stops as soon as a version hits the target.
  Trust the held-out column.
- These are two runs on one robot and one object, so the evidence is thin. More tasks and
  repeat runs are needed before claiming anything general.
- I checked the diffs by reading them: no seed or randomization special-casing. A skill only
  sees observations and the task spec.

## Files
- `*/pick_place_*.py`: the agent-written skills, exactly as accepted.
- `*/history.json`: the forge loop's per-round records.
- `*/dashboard.html`: the T4 dashboard for each run, self-contained.

Reproduce:
`skillforge forge tasks/franka_pick_cube_to_bin_hard.json --seed-skill pick_place --improver claude-code --rounds 5 --rollouts 100 --target 0.9 --budget-usd 20`
