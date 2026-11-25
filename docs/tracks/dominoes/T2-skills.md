# T2 · Domino skills

**Goal:** `ToppleChain` succeeds under every disturbance the tasks throw, and contains no
arm-specific numbers, so it runs on the Kinova unchanged.

## Done (`track/t2-dominoes`)
`ToppleChain` 0.1.0: survey (ok / misplaced / leaning / down / down_bad) -> repair farthest-first
with `ReplaceDomino` -> push the first domino (called off if anything moves first) -> watch ->
nudge a stalled or partly knocked chain. Kinematic test world in `tests/domino_kinematic.py`.

## Deliverables
1. **MuJoCo numbers.** 100 seeds per task, failures by tag, a GIF of a repair and of a nudge.
   Tune `ReplaceDomino.grasp_depth/drop/lift_step` and `ToppleChain.push_*` from what fails.
2. **Stand a fallen domino back up** (`down_bad` today is reported and left). Candidates: grasp
   the lying domino across its width near one end, lift that end and pivot on the other; or
   pick it lying, place it on its edge against a fixed fence, regrasp upright. Add a task with a
   sideways knock (`knock_dir` across the lane) to measure it.
3. **Nudge from above** when a fallen predecessor leans on the next domino (the push from
   behind can hit it): push the standing domino's top face edge instead.
4. **Arm-agnostic:** hand clearance, finger thickness and push gripper command come from the
   robot profile (T1 item 2), not constants.
5. Curved chains: allow a direction per condition (the predicate already has it); fit spots on
   a curve.

## Tests
On the kinematic world: sideways knock -> stand-up -> success. On MuJoCo: seed-0 success per task.
