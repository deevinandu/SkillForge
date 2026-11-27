# T5 · Learning recovery from demonstrations

**Goal:** a learned policy that recovers from disturbances, trained from `ToppleChain` rollouts.

## Deliverables
1. Demos from `ToppleChain` with disturbances on (the recoveries are the interesting part),
   with a phase label per step (survey / repair / push / watch) from the skill's log.
2. A policy (ACT or diffusion) on state first, images later (T6). Compare with and without
   recovery segments in the data, under the same seeded disturbances.
3. DAgger with the scripted skill as expert, so states off the demo distribution (a bump mid
   carry) get labels.
4. Cross-embodiment stretch: one policy conditioned on the arm, trained on Franka + Kinova demos.
