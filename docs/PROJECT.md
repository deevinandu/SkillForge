# Project: a scoped skill library shared across robot arms

Team Echo, Robot Autonomy (CMU). This page is the research question this fork of SkillForge
exists for. Read it before working on anything cross-robot. The brief with ordered work items is
[`docs/tracks/scoped-library.md`](tracks/scoped-library.md).

## The question
SkillForge already has a coding agent that writes a robot skill as Python, tests it in simulation
under randomization, reads the failures and repairs it, and keeps a repair only if a paired
statistical gate says it helps. We add a **skill library shared across different robot arms in
which every repair is scoped**:

- **general**: a better strategy that should help every arm (e.g. "plan the grasp backwards from
  the hanging pose", "turn the handle toward the hook before approaching");
- **robot-specific**: an offset, angle, depth or timing that suits one arm's kinematics, gripper
  or controller (e.g. a grasp depth tuned to one gripper's pads).

Does scoping repairs, and checking a "general" fix on the other robots before sharing it, let a
new robot reach high success with fewer repairs, without breaking robots that already work?

We do not yet know whether arm differences cause enough harm (negative transfer) to matter.
Finding that out early is part of the work. If they don't, the broader framing is "where does a
fix apply": across robots, objects, tasks and setups.

## Closest prior work: ASPIRE
ASPIRE (Lu et al., NVIDIA et al., arXiv [2607.00272](https://arxiv.org/abs/2607.00272), 2026):
a coding agent with a library of skills learned from validated repairs. How it maps onto this repo:

| ASPIRE | SkillForge today |
| --- | --- |
| Coding agent repairs programs from execution feedback | Yes: the forge loop (`agents/`, `skillforge forge`), Claude Code or API improver |
| "Robot execution engine": per-primitive traces with keyframes | Partly: events, failure tags, the skill's own log; no per-call keyframes |
| Admission check: a fresh "verifier actor" re-runs the source task on some seeds | Stronger: paired McNemar gate on fixed gate seeds, separate dev and held-out seeds, regression baselines |
| "Skill library": text entries (failure signature, when-to-apply, repair strategy, optional code sketch) reused across tasks | No: `agents/library.py` `SkillLibrary` stores versions of whole skill files, per task |
| "Evolutionary search": K candidates per round conditioned on the top-3 previous programs | No: one candidate per round, keep-the-best |
| Coordinator: curriculum, backward-transfer rerun of low-scoring tasks | No |
| Cross-embodiment: three sim-discovered skills given as text hints to a different real robot | No cross-robot support yet (see the gaps below) |

Where we differ: ASPIRE admits a skill after checking it once, on the task it came from. As far
as we read it, it does not re-validate entries later or check them on other robots and tasks.
That is the gap this project targets. Its cross-embodiment result is narrow (in-context hints on
one other robot), and it does not measure negative transfer. We read the paper's main sections
directly. **Our claims about its limitations section come from a summary, so check them
against the paper's text before citing them.**

## Robots
- **Franka Panda** (7-DoF, Franka Hand, opens 80 mm). Several in the course lab. The robot for
  the course deliverable, and the only arm the sim supports today (`PandaProfile`).
- **Kinova Gen3 + Robotiq 2F-85** (opens 85 mm), at CMU's AI Maker Space; real-hardware work is
  for spring. **Open: 6-DoF or 7-DoF?** The project doc says 6-DoF (no redundant joint, so
  reachable wrist orientations differ, especially near the edge of reach).
  `docs/tracks/dominoes/T1-sim.md` says 7. mujoco_menagerie's Gen3 is, as far as we know, the
  7-DoF one, so a 6-DoF sim model would have to be built from Kinova's URDF. Check the real arm.
- **Fallback second type:** a Franka with one joint locked and tighter limits. A weak substitute;
  label it as such in every result.
- One external camera with AprilTags is planned for the real robots.

## Tasks
- **Mug hanging** (done in sim, Franka only): pick a mug with a randomly facing handle and hang it
  on a peg of a stand so the peg goes through the handle loop and the mug hangs freely.
  `tasks/franka_hang_mug*.json`, `skills/hang_mug.py`, `results/2026-11-25-hang-mug/`. Chosen
  because the right grasp depends on what happens after it, which should produce both kinds of
  fix.
- **Dominoes** (sim, Franka): `docs/tracks/dominoes/`, which already plans a Kinova copy of the
  tasks.
- **Open: which task carries the paper?** The dominoes briefs call dominoes "the project task";
  the project doc names mug hanging as the current focus. A workshop paper needs several tasks;
  pick the primary one.

## Experiment design (open to change)
Build a library on robot A, then give robot B (same type) and robot C (different type) a starting
point under each condition, and count repair rounds (and cost) until each reaches a success target
on the gate seeds; confirm on held-out seeds.

| Condition | Robot B / C starts from |
| --- | --- |
| **scoped** (ours) | shared skills plus only fixes verified as general |
| **share everything** | every fix from every robot |
| **from scratch** | nothing shared (the naive seed skill) |
| **copy unchanged** | robot A's final skills; no repairs allowed |

Also measure:
- **negative transfer**: does robot A get worse once B's or C's fixes are shared back?
- **scope-label error**: how often the agent's own scope label disagrees with what checking finds.

Reuse SkillForge's gate, dev and held-out seed split, and paired tests. Repeat runs: the agent is
not deterministic, so one run per condition is an anecdote.

### Ways to decide a fix's scope (evaluate or replace)
1. **Structure**: skills read per-robot parameters, so a parameter-only fix is robot-specific.
2. **Agent label**: the agent proposes a scope and a reason.
3. **Checking**: run a "general" fix on the other robots' sim twins with the paired gate; if it
   hurts any of them, keep it robot-specific.

First data point: the mug forge round's only change was `grasp_depth` 0.018 -> 0.028 m (gate
82 -> 99/100). It depends on finger-pad and wall geometry, so by (1) it is robot-specific, and
whether it holds on the Robotiq gripper is untested.

### Risk to watch
If the seed skill is already good, there is little to repair and nothing to scope. The mug seed
(`HangMug` 0.1.0, 82/100) already encodes the general strategy. A naive seed (handle-blind grasp)
gives the agent general fixes to discover.

## Status (2026-11-28)
Measured in sim on the Franka only. **Nothing in this project has run on a real arm.**

| Item | Status |
| --- | --- |
| Mug hanging task, success check, seed skill | done (sim) |
| One forge round on the mug | done: 82 -> 99/100 gate, 198/200 held-out |
| Robot profiles; second arm in sim | missing (`MujocoEnv` hardcodes `PandaProfile`) |
| Per-robot parameters for skills | missing (contract request in `HANDOVER.md`) |
| Scoped library: scope labels, cross-robot check, sharing | missing |
| Experiment runner for the four conditions | missing |
| Negative transfer and scope-label error measurements | missing |
| Naive mug seed skill | missing |
| Real Franka driver; AprilTag mug pose | missing (`real/` is tested against fakes only) |
| Real Kinova | spring |

## Timeline
- Course progress presentation: Nov 30 / Dec 2.
- Final report (2-3 pages, IEEE) and a 1-3 minute video of the robot doing the task
  autonomously: Dec 11.
- Then an ICRA 2027 workshop paper, possibly IROS or CoRL 2027. That needs several tasks, at least
  two real robot types, statistically sound comparisons, and a clear finding about negative
  transfer.
