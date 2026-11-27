# T7 · Dominoes on real arms

**Goal:** the domino task on the course-lab Franka and the AI Maker Space Kinova Gen3, with
disturbances applied by a person in a repeatable way.

## Deliverables
1. Real dominoes and their measured size and mass; update the task files to match (a real
   domino is about 48 x 24 x 7.5 mm, smaller than the sim's 80 x 40 x 10 mm: check that the
   grasp and push heights still clear the palm).
2. A disturbance protocol for the operator: the session runner prints "bump domino_3 by about
   2 cm now" at the seeded step; the operator does it; the runner logs the measured pose
   before and after (`real.json`). Same seed, same requested disturbance on both arms.
3. Safety: pushes are slow contact moves near the table; keep the force limits and stop on
   contact force above the push's expected value.
4. Kinova driver behind the same controller interface as the Franka (needs T1 items 2 and 4).
5. Scoreboard rows for both arms, with the operator's success label and the estimator's.
