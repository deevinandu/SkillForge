# T4 · Eval for recovery

**Goal:** numbers that show recovery, not only final success.

## Done (`track/t4-disturbance-eval`)
`unrecovered_disturbance` tag; `disturbance_scale` sweep axis and `scale_disturbances`.

## Deliverables
1. Regression baselines for the three domino tasks (and the Kinova copies when T1 has them).
2. Recovery metrics per episode, from events and step indices: disturbances applied, steps
   from each disturbance to the next `release:` of that object (time to repair), whether the
   chain fell in one push. Add them to the dashboard.
3. `disturbance_scale` sweep (0 to 3x) for `ToppleChain` on each task; find where it breaks.
4. Per-arm rows in the scoreboard: same task family, Franka vs Kinova, same seeds.
