# T1 · Sim for dominoes and a second arm

**Goal:** the domino tasks behave like real dominoes in MuJoCo, and the same task file runs on a
Franka Panda and a Kinova Gen3 (Robotiq 2F-85), so T3 can measure skill transfer.

## Done (`track/t1-dominoes`)
`object_toppled` judged on true quaternions; `DisturbanceSchedule` (own RNG stream
`[seed, 2]`); pose teleports with an overlap check, knocks as an angular velocity, never on an
object the fingers touch; `toppled:<obj>` events.

## Deliverables
1. **Validate the physics.** Run `tests/test_sim_dominoes.py`. Check by eye (GIF) that a knocked
   domino cascades down a 5-domino chain and the fallen dominoes end leaning past 0.6 rad. If the
   chain stalls or dominoes jitter at rest, tune the domino geoms (`condim`, `solref`, friction)
   in the env, not in the task, and record the change.
2. **Robot profiles.** Replace the hardcoded `PandaProfile` with a registry keyed by
   `RobotSpec.source` (`menagerie:franka_emika_panda`, `menagerie:kinova_gen3` + `robotiq_2f85`
   attached at the flange). A profile names arm joints, actuators, gripper actuator, finger
   bodies, hand body, base body, home posture, TCP offset and the gripper's command range. The
   controller (IK, speed limits, gravcomp) and event detection read only the profile. This is
   also T7's open request ("`MujocoEnv` hardcodes `PandaProfile`").
3. **Kinova domino tasks.** `kinova_topple_domino_chain*.json`: the same scene and success with
   the Kinova robot spec and its base placed so the lane is in comfortable reach. Gen3 has 7
   DoF but a different wrist; the 2F-85 opens 85 mm, so a 4 cm domino still fits.
4. **Public controller** (T7's request): `reset(q)`, `solve(action) -> (q_des, events)`, `fk(q)`,
   so the real Kinova driver and the sim share one reference generator.

## Tests
Profiles: both arms reach the lane spots top-down at grasp and push heights without `ik_failure`.
A disturbance draws the same values on both arms for the same seed.
