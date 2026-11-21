# One forge iteration on the ball task (2026-11-21)

One round of the agent loop, run to watch it work: score the current skill, let the headless
Claude Code improver rewrite it, score the rewrite on the same seeds, gate it. Sim only.

## Command
```bash
MUJOCO_GL=egl skillforge forge tasks/franka_pick_ball_to_bin.json \
  --seed-skill results/2026-11-05-claude-code-repeats/ball_r1/pick_place_0.2.1.py \
  --rounds 2 --rollouts 100 --improver claude-code --model claude-sonnet-5-5 \
  --budget-usd 3 --per-call-budget-usd 3 --gate paired \
  --library runs/library_iter --out runs/forge/ball_iter2
```
Run on the Pro subscription through the CLI's own login (no API key; extra usage disabled, so no
money spent). The CLI's cost figure for the one call: $0.30 (8,381 output tokens, about 10 min).
A first attempt (`runs/forge/ball_iter1`) did nothing: the CLI's login had expired and the
improver logged the error as "success" (fixed on branch `fix/claude-code-cli-error`).

## Scores (task's own randomization, gate seeds 0-99)
| Version | Success | 95% CI | Failures |
| --- | --- | --- | --- |
| 0.1.1 (seed: agent-written ball skill from 2026-11-05, renamed by the library) | 85/100 | 77-91% | grasp_slip 10, wrong_placement 2, timeout 2, ik_failure 1 |
| 0.1.2 (Claude Code's rewrite) | 85/100 | 77-91% | grasp_slip 11, timeout 2, ik_failure 1 |

Paired: the rewrite fixed seeds 23, 52, 61, 97 and broke 0, 9, 21, 82 (+4/-4, one-sided McNemar
p = 0.64). **Rejected** by the paired gate (alpha 0.1). The seed version stays the best.

## What Claude Code changed (`pick_place_0.1.1.py` -> `pick_place_0.1.2.py`, `RATIONALE.md`)
- Grip settle doubled: `grip_min_steps` 8 -> 16. The only change it meant to keep.
- Added a `_Gentle` move that caps the commanded step while carrying, and replaced the lift /
  transport / place primitives with it. It measured the cap causing IK failures on regrasps and
  set `carry_step = 1.0`, which disables the cap; the replacement moves stay in the code.
- Tried a lower carry height: no effect, not kept.
- Its own verdict: "no measurable improvement ... The slip cause is not fixed." On its 30 dev
  seeds both versions scored 96%; the gate seeds are harder.

Not checked: whether the swapped-in moves (which drop the primitives' own checks) explain the four
seeds it broke. The remaining failures are slips at low friction or high mass; a different grasp
(lower on the ball, more force) is the obvious next thing to try.

## GIFs (front camera, not in git: `runs/` is ignored)
Rendered with a scratch script that replays the same seed for both versions
(`eval.video.record_episode`, every 3rd step):
- `runs/forge/ball_iter2/gifs/seed23_{before,after}.gif`: failed before, succeeds after
- `runs/forge/ball_iter2/gifs/seed52_{before,after}.gif`: failed before, succeeds after
- `runs/forge/ball_iter2/gifs/seed8_{before,after}.gif`: fails in both
- `runs/forge/ball_iter2/gifs/seed0_{before,after}.gif`: succeeded before, fails after

## Files
`history.json` (both rounds with per-seed outcomes), the two skill versions, `RATIONALE.md`.
