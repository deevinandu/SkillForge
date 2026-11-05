# Claude Code improver: repeats, new tasks and a paired gate (2026-11-05)

Five forge runs with the headless Claude Code improver (`--improver claude-code`, model
`claude-opus-5-5`): three independent repeats on the hard cube task and one run on each of two new
tasks. All runs used the new **paired gate**. **Everything here is measured in simulation
(MuJoCo, current main physics). Nothing has run on hardware.**

## What changed in the gate, and why
The 2026-11-04 hard run accepted round 1 (0.2.1) at 73/100 vs the seed's 69/100. The gate
accepted it because 73 was above the seed's 95% CI lower bound (59%), not because it was better.
On held-out seeds it was no better (76.0% vs 76.5%).

Every round is scored on the same gate seeds (0–99), so the comparison is now paired. A candidate
is accepted only if (a) it solves more gate seeds than the current best, and (b) a one-sided
exact McNemar test on the discordant seeds gives p < alpha (default 0.1, `--gate-alpha`). The
discordant seeds are the ones that only the best solves (`best_only`) or only the candidate solves
(`cand_only`). Under H0 (the two versions are equally good), `cand_only ~ Binomial(best_only +
cand_only, 1/2)`. Seeds both versions solve, or both fail, say nothing about the difference. With
alpha 0.1, the gate needs, for example, +4/-0, +7/-1 or +11/-4. The old rule is still available
as `--gate ci`. Every round's paired counts and p go into `history.json` (`paired`,
`success_seeds`) and onto the CLI round line, under either rule.

Why alpha 0.1 and not 0.05: the gate makes one decision per round, and wrongly rejecting a real
gain only costs a round. Wrongly accepting a no-gain version moves the parent for every later
round. With 100 seeds, 0.1 still rejects the case that motivated the change (+11/-7) by a wide
margin.

**Replay of the 2026-11-04 hard run.** I re-evaluated its three versions on gate seeds 0–99 with
every failure listed (the stored reports keep only 20 failures). The totals reproduce exactly:
69/73/95. Per-seed data: `tests/fixtures/hard_2026_11_04_gate_seeds.json`, checked by
`tests/test_agents_library.py::test_paired_gate_on_the_2026_11_04_hard_run`.

| Candidate vs best | Paired (cand-only / best-only) | p (one-sided) | Old rule | Paired rule |
| --- | --- | --- | --- | --- |
| 0.2.1 (73) vs 0.2.0 (69) | +11 / -7 | 0.24 | accept | **reject** |
| 0.2.2 (95) vs 0.2.0 (69) | +28 / -2 | < 1e-5 | accept | **accept** |

Because 0.2.1 would have been rejected, 0.2.2 is compared against the seed in this replay. In the
real run it was derived from 0.2.1. Against 0.2.1 it is +22/-0.

## How the numbers were made
- **Gate seeds 0–99**: the loop scores every round on these. The loop stops as soon as an accepted
  version reaches 90%, so the best version's gate rate is biased upward.
- **Dev seeds 100000+**: the agent may run `./dev_eval` on these inside its workspace.
- **Held-out seeds 5000–5199**: neither the loop nor the agent ever used these. Each row below
  comes from `agents.sandbox.evaluate_source`, run after the forge runs (`*/heldout.json`, with
  per-seed outcomes).
- Success is judged on true simulator state. CIs are Wilson 95%. "Paired" columns count seeds
  on the same seed set.
- Settings for every run: `--rounds 5 --rollouts 100 --target 0.9 --budget-usd 5
  --per-call-budget-usd 6 --gate paired --gate-alpha 0.1`, seeded from hand-written PickPlace v0.2
  (`--seed-skill pick_place`).

## New tasks and hand-written baselines (gate seeds 0–99)
All three new tasks keep the hard task's harsh randomization: low friction (scale 0.3–0.8 or
0.4–1.0), gains 0.6–1.4, observation noise up to 12–15 mm, and up to 5–6 steps of latency.

| Task | Object | v0.1 | v0.2 | Main v0.2 failures | Forge run? |
| --- | --- | --- | --- | --- | --- |
| `tasks/franka_pick_cylinder_to_bin.json` | cylinder r 2 cm, 8 cm tall, 0.15 kg | 35/100 | 44/100 (CI 35–54%) | grasp_slip 25, ik_failure 25 (after a slip and a botched re-grasp), collision 6 | yes |
| `tasks/franka_pick_ball_to_bin.json` | sphere r 2.2 cm, 0.1 kg | 53/100 | 49/100 (CI 39–59%) | grasp_slip 27, ik_failure 23 (after a slip) | yes |
| `tasks/franka_pick_cube_to_far_bin.json` | cube, bin on the far side | 8/100 | 55/100 (CI 45–64%) | collision 18, grasp_slip 14, timeout 8 | no (budget) |

I looked at the per-seed details: slips happen on friction scales up to 0.8 and are followed by
failed re-grasps. These are skill failures, not physics limits, and both forge runs confirmed that
they can be fixed. I did not add a lift task. Both seed skills (v0.1, v0.2) refuse
`object_lifted` tasks, and no lift skill exists in `skills/` (T2-owned). Writing one only to seed
the loop would have meant judging the agent against my own throwaway baseline.

## Results

### Hard task, 3 independent repeats (`tasks/franka_pick_cube_to_bin_hard.json`)

| Run | Rounds (accepted / evaluated) | Seed gate | Best gate | Seed held-out | Best held-out | Held-out paired vs seed | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hard_r1 | 1 / 1 | 69/100 | 0.2.1: 95/100 (CI 88.8–97.8%) | 152/200 (76.0%, CI 69.6–81.4%) | 185/200 (92.5%, CI 88.0–95.4%) | +42 / -9 | $1.24 |
| hard_r2 | 2 / 4 | 69/100 | 0.2.4: 97/100 (CI 91.5–99.0%) | 152/200 | 194/200 (97.0%, CI 93.6–98.6%) | +44 / -2 | $3.32 |
| hard_r3 | 1 / 1 | 69/100 | 0.2.1: 94/100 (CI 87.5–97.2%) | 152/200 | 184/200 (92.0%, CI 87.4–95.0%) | +40 / -8 | $1.19 |

- **Variance across repeats.** Held-out best was 92.5%, 97.0% and 92.0% (mean 93.8%, sd 2.8
  points). Every repeat beat the seed by 16–21 points held out, with p < 1e-5 paired. Two
  repeats needed one agent round; r2 needed four. The 2026-11-04 run (ci gate) reached 97.5% held
  out in two rounds.
- **hard_r2 is where the gate did its job.** Round 1 (70/100, +6/-5, p 0.5) and round 2 (73/100,
  +10/-6, p 0.23) were rejected. Held out, they were 78.5% and 77.5% vs the seed's 76.0% (+17/-12
  and +18/-15), so they really were no gain. Under the old rule both would have been accepted
  (70 and 73 are above 59%). Round 3 restarted from the seed and was accepted (+23/-3). Round 4
  built on it (+8/-0 vs 0.2.3; held out +12/-0).
- What the agent changed, from its rationales:
  - **r1 0.2.1**: the slip check uses the mean of the last 8 raw offsets; grasp and place heights come from known geometry with a table clearance floor; the gripper settles 10 steps (longer than the 6-step latency); an alignment hold before release.
  - **r2 0.2.1 (rejected)**: grasp z clamped to the resting height, settle 10 steps, an averaged placement check.
  - **r2 0.2.2 (rejected)**: running-mean grasp target with z snapped to the resting height, averaged offsets.
  - **r2 0.2.3**: running mean of the raw object position while it is still (z pinned when resting), an averaged in-hand offset for placing, an observe hold before verification.
  - **r2 0.2.4**: a hold monitor that measures drift on an EMA of the raw offset against a reference averaged over the first 8 carried samples; it removes false slip aborts.
  - **r3 0.2.1**: a low-pass-filtered hold monitor; grip and release wait for the measured width to settle; the grasp is planned from the median of 10 still observations and then frozen; the in-hand offset is re-measured above the bin; the placement check uses a median.
- All three repeats, and the 2026-11-04 run, found the same root cause: single noisy samples
  falsely triggered the slip monitor, plus noisy grasp heights. That is evidence the diagnosis is
  robust, not luck.

### Cylinder (`tasks/franka_pick_cylinder_to_bin.json`)

| Run | Rounds | Seed gate | Best gate | Seed held-out | Best held-out | Held-out paired | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cylinder_r1 | 1 / 1 | 44/100 (CI 34.7–53.8%) | 0.2.1: 99/100 (CI 94.6–99.8%) | 78/200 (39.0%, CI 32.5–45.9%) | 194/200 (97.0%, CI 93.6–98.6%) | +118 / -2 | $2.60 |

- **0.2.1** grasps 3.5 cm below the top (near the centre of mass) and picks the grasp yaw whose
  closing axis is across the carry direction. It ramps carry speed and acceleration
  (0.25 m/s, 0.5 m/s²), plans grasps from a median of resting observations and sets a TCP height
  floor. The slip thresholds are widened for 12 mm noise. One constant is tuned to this scene's
  geometry: `min_tcp_z = 0.028` m, an absolute height, not derived from the task spec.

### Ball (`tasks/franka_pick_ball_to_bin.json`)

| Run | Rounds | Seed gate | Best gate | Seed held-out | Best held-out | Held-out paired | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ball_r1 | 1 / 4 | 49/100 (CI 39.4–58.7%) | 0.2.1: 85/100 (CI 76.7–90.7%) | 93/200 (46.5%, CI 39.7–53.4%) | 180/200 (90.0%, CI 85.1–93.4%) | +92 / -5 | $4.90 |

- **0.2.1** carries the ball 3 cm above the place height instead of 10 cm. The grip waits at least
  8 steps and until the measured width settles. **Caveat:** the agent says it carries low
  *because the goal region has no walls in the scene*. Regions are visual-only sites. On a real
  bin with walls, this skill would hit them. This is a twin-fidelity gap that the agent
  exploited, not seed special-casing. The bin should get collision walls (T1).
- Rounds 2–4 were rejected: 88 (+8/-5, p 0.29), 83 (+3/-5) and 87 (+4/-2, p 0.34). Held out
  against 0.2.1 they were +13/-9 (p 0.26), +5/-14 and **+10/-3 (p 0.046)**. So round 4 (0.2.4,
  a longer grip settle, 93.5% held out vs 90.0%) was probably a small real gain that the gate
  missed. A paired gate with 100 seeds cannot reliably see a gain of about 3 points. That is the
  price of not accepting noise. More gate seeds, or a second-stage re-test on fresh seeds for
  borderline candidates, would help. In round 4 the agent also wrote that `./dev_eval` was denied
  to it, so that candidate was not tested on dev. I did not find out why; this was the last call
  of the run, capped at the run's remaining ~$0.37.

### Spend
Total actual spend: **$13.25** ($1.24 + $3.32 + $1.19 + $2.60 + $4.90), from the CLI's own
`total_cost_usd`. Allocated caps: 5 x `--budget-usd 5` = $25, the hard cap for this batch. No
more runs were launched after the allocation reached $25, so the far-bin task has baselines but
no forge run.

## Checks I made
- **I read every accepted diff** (hard_r1 0.2.1, hard_r2 0.2.3 and 0.2.4, hard_r3 0.2.1,
  cylinder 0.2.1, ball 0.2.1) against its parent, looking for seed or randomization
  special-casing. I found none. The skills only see observations and the TaskSpec. They use
  nominal geometry from the spec (resting height, object pose z), which is legitimate. Two
  scene-specific choices are flagged above: the cylinder's absolute `min_tcp_z` and the ball's
  low carry that relies on bins without walls.
- The secret scan (tokens, session ids, UUIDs, home and tmp paths) over this directory found
  nothing. `history.json` holds no session ids or workspace paths.
- Dashboards (`*/dashboard.html`) render from the new history.json, with the extra fields.

## Caveats
- Sim only, one robot (Franka), one controller, three small objects. "Held-out" means unseen
  seeds of the same randomization distribution, not a new distribution.
- 3 repeats on one task, 1 run each on two others. The repeat variance (sd ≈ 3 points) is from 3
  samples.
- The gate rates of the final versions are optimistic (early stop at 90%); use held-out.
- The gate's power is limited (see ball round 4). alpha 0.1 with 100 seeds trades missed small
  gains for no accepted noise.
- `version` strings inside some skill files lag the loop's version (the agent sometimes kept
  "0.2.1"); the loop's version (file name, history) is authoritative.

## Files
- `<run>/history.json`: per-round records, including `paired` and `success_seeds`.
  `<run>/forge.log`: the CLI round lines. `<run>/heldout.json`: held-out results for every
  evaluated version, with per-seed successes. `<run>/pick_place_<v>.py`: accepted agent skills,
  exactly as accepted. `<run>/dashboard.html`: the T4 dashboard.
- `baselines/`: per-seed failing seeds of v0.1/v0.2 on gate seeds 0–99 for the new tasks
  (`gate_<task>_<skill>.json`), held-out 5000–5199 for the v0.2 seed (`heldout_<task>_pick_place.json`),
  and the per-seed re-evaluation of the 2026-11-04 hard-run versions (`replay_2026-11-04_*.json`).

Reproduce one run:
`skillforge forge tasks/franka_pick_cube_to_bin_hard.json --seed-skill pick_place --improver claude-code --rounds 5 --rollouts 100 --target 0.9 --budget-usd 5 --per-call-budget-usd 6 --gate paired --gate-alpha 0.1`
