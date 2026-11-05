"""A phase-synchronised expert for DAgger: keep a phase-machine teacher in step with the learner.

Why: a primitives-based teacher (PickPlace v0.2.x) is a `Retry(Sequence([...]))` whose phases end
on 6 mm waypoint tolerances. Run in the shadow of a learner that cuts corners, it stays in
"approach" while the learner is already descending, or in "transport" while the learner hovers
over the bin, and labels the wrong thing (see dataset.py, HANDOVER 2026-11-04 T5). This wrapper
re-synchronises it every step:

1. infer the phase the executed state is in, from signals that do not depend on the teacher's
   memory: the last *executed* gripper command and how long it has been held, the measured
   finger width (noise-free), and the teacher's own filtered object estimate relative to the
   TCP and to its grasp / place targets;
2. if the teacher's current phase is inconsistent with that (wrong category, or behind within
   the same category, or implausible such as "descend" 3 cm off the grasp axis), fast-forward
   or rewind its sequence to the inferred phase and patch the blackboard the phase needs (hold
   monitor armed with the current offset when carrying, a fresh grasp plan when picking, the
   current gripper orientation when it never planned one);
3. tick the teacher once; its action is the label.

A teacher whose tree has finished or failed is re-instantiated from scratch and then synced.
The result is the teacher's own targets (grasp point, hover point, place point, settle times),
evaluated at the learner's state: a state-based expert built from the phase machine, not a
re-implementation of it.

What it requires of the teacher: `runner` (primitives.Runner) whose root is a `Retry` around a
`Sequence` with nodes named disarm, plan_grasp, approach, descend, grip, lift, transport, place,
release (v0.2.0 / 0.2.2 naming). `PhaseSyncExpert.supports(teacher)` checks that.

Known approximations (all measured only in sim): "held" means gripper commanded closed, measured
width strictly between almost-closed and almost-open, object estimate within 4 cm of the TCP;
under heavy observation noise (stress x2: up to 3 cm) the filtered estimate can still put a held
cube > 4 cm away, which reads as a miss and labels "retreat and re-pick".
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from skillforge.contracts import Action, Observation, Skill, TaskSpec

REQUIRED = ("disarm", "plan_grasp", "approach", "descend", "grip", "lift", "transport", "place", "release")
PICK = ("disarm", "clear", "plan_grasp", "approach", "descend")
CARRY = ("lift", "transport", "place")


class PhaseSyncExpert:
    """Wraps a primitives-based teacher; `act(obs)` returns the synced teacher's action."""

    near = 0.04  # metres: object estimate this close to the TCP counts as "between the fingers"
    on_axis = 0.015  # metres: xy distance to the grasp/place point that counts as lined up
    off_axis = 0.03  # metres: beyond this a descend/place phase is implausible
    grace = 16  # steps after a gripper switch while fingers may still be moving (settle + latency)

    def __init__(self, make_teacher: Callable[[], Skill]):
        self.make_teacher = make_teacher
        self.teacher: Skill | None = None
        self.name = "phase_sync"
        self.version = "1"
        self.jumps: list[str] = []

    @staticmethod
    def supports(teacher: Skill) -> bool:
        try:
            seq = teacher.runner.root.node  # type: ignore[attr-defined]
            names = {n.name for n in seq.nodes}
        except AttributeError:
            return False
        return set(REQUIRED) <= names

    # -------------------------------------------------------------- lifecycle

    def reset(self, task: TaskSpec) -> None:
        self.task = task
        self.region = task.scene.region(task.success.region)
        self.max_width = float(task.scene.robot.max_gripper_width)
        self.g_cmd = 0.0
        self.since_switch = 10**6
        self.rest_z: float | None = None
        self.jumps = []
        self.n = 0
        self._fresh()

    def _fresh(self) -> None:
        t = self.make_teacher()
        t.reset(self.task)
        if not self.supports(t):
            raise TypeError(f"{type(t).__name__} is not a primitives pick-place phase machine; phase sync needs {REQUIRED}")
        self.teacher = t
        self.runner = t.runner  # type: ignore[attr-defined]
        self.ctx = self.runner.ctx
        self.seq = self.runner.root.node
        self.idx = {n.name: i for i, n in enumerate(self.seq.nodes)}
        self.node = {n.name: n for n in self.seq.nodes}
        self.post = {n.name for n in self.seq.nodes[self.idx["release"]:]}
        self.planned_at: np.ndarray | None = None
        plan = self.node["plan_grasp"]
        fn = plan.fn

        def planned(obs, c, _fn=fn):
            why = _fn(obs, c)
            if why is None and c.obj_pos is not None:
                self.planned_at = np.asarray(c.obj_pos, float).copy()
            return why

        plan.fn = planned

    def note_executed(self, a: Action) -> None:
        """Tell the expert what the env actually received this step (the learner's action)."""
        g = 1.0 if a.gripper >= 0.5 else 0.0
        self.since_switch = self.since_switch + 1 if g == self.g_cmd else 1
        self.g_cmd = g

    # -------------------------------------------------------------- phase logic

    @property
    def phase(self) -> str:
        if not self.runner.status.running:
            return "finished" if self.runner.status.done else "failed"
        return self.seq.nodes[self.seq.i].name if self.seq.i < len(self.seq.nodes) else "finished"

    def _target(self, name: str, obs: Observation) -> np.ndarray:
        """A node's current target, without side effects on the teacher's blackboard."""
        offs = self.ctx.bb.get("offsets")
        n_offs = len(offs) if isinstance(offs, list) else None
        try:
            return np.asarray(self.node[name].target(obs, self.ctx), float)
        finally:
            if n_offs is not None:
                del self.ctx.bb["offsets"][n_offs:]

    def _ensure_plan(self, obs: Observation) -> None:
        """Plan a grasp if the teacher has none, or if the object was moved (knocked) since."""
        c = self.ctx
        moved = self.planned_at is not None and c.obj_pos is not None and np.linalg.norm(c.obj_pos[:2] - self.planned_at[:2]) > 0.05
        if "grasp" not in c.bb or self.planned_at is None or moved:
            c.monitor.disarm()
            c.bb.pop("rest_z", None)
            c.bb["offsets"] = []
            self.node["plan_grasp"].tick(obs, c)

    def decide(self, obs: Observation) -> str | None:
        """The phase to move the teacher to, or None when its current phase fits the state.

        Only clear desyncs move it: a category mismatch (the teacher picks while the learner
        carries, carries while the gripper is open, ...), plus two corner cuts within a
        category (approach -> descend, transport -> place) and their reversals. Everything
        else, including misses and slips, is left to the teacher's own checks.
        """
        c = self.ctx
        p = c.obj_pos
        have = self.phase
        if p is None or have not in self.idx:
            return None
        post = have in self.post
        ee = np.asarray(obs.ee_pos, float)
        w = float(obs.gripper_width)
        near = float(np.linalg.norm(p - ee)) < self.near
        if self.g_cmd >= 0.5:
            if self.since_switch <= self.grace and near:
                return None if have in ("grip", *CARRY) else "grip"  # fingers closing on it
            if not (near and c.monitor.width_eps < w < 0.9 * self.max_width):
                return "disarm" if post else None  # closed on nothing: the teacher's checks see it
            # holding the object
            if not c.monitor.armed:
                c.monitor.arm(obs, p)
                c.bb["offsets"] = []
            place, above = self._target("place", obs), self._target("transport", obs)
            xy = float(np.linalg.norm(ee[:2] - place[:2]))
            placing = xy < self.on_axis or (xy < self.off_axis and ee[2] < above[2] - 0.02)
            if have not in CARRY:
                if placing:
                    return "place"
                return "lift" if ee[2] < above[2] - 0.02 and xy > 0.05 else "transport"
            if have != "place" and placing and ee[2] < above[2] - 0.02:
                return "place"  # the learner cut the corner and is already lowering
            if have == "place" and xy > 1.5 * self.off_axis:
                return "transport"
            return None
        # gripper open
        ctr, half = np.asarray(self.region.center), np.asarray(self.region.half_extents)
        in_goal = bool(np.all(np.abs(p - ctr) <= half + 0.01))
        resting = self.rest_z is not None and abs(p[2] - self.rest_z) < 0.025
        if in_goal or (near and not resting and self.since_switch <= self.grace):
            return None if post else "release"
        self._ensure_plan(obs)
        g, h = self._target("descend", obs), self._target("approach", obs)
        xy = float(np.linalg.norm(ee[:2] - g[:2]))
        lined_up = xy < self.on_axis and ee[2] < h[2] + 0.015
        if have == "grip":
            if float(np.linalg.norm(ee - g)) < self.on_axis:
                return None  # teacher closes at the grasp pose; the learner simply has not yet
            return "descend" if lined_up else "approach"
        if have not in PICK:
            if lined_up:
                return "descend"
            return "disarm" if ee[2] < g[2] + 0.04 and xy > self.off_axis else "approach"
        if have == "approach" and xy < self.off_axis and ee[2] < h[2] - 0.02:
            return "descend"  # the learner cut the corner and is already going down
        if have == "descend" and (xy > 2 * self.off_axis or ee[2] > h[2] + 0.03):
            return "approach"
        return None

    def _jump(self, want: str, obs: Observation) -> None:
        c = self.ctx
        have = self.phase
        target = want
        i = self.idx[target]
        for nd in self.seq.nodes[i:]:
            nd.reset()
        self.seq.i = i
        self.runner._park = None
        if target in CARRY or target == "grip":
            c.quat = np.asarray(obs.ee_quat, float).copy()
            if target in CARRY and not c.monitor.armed and c.obj_pos is not None:
                c.monitor.arm(obs, c.obj_pos)
                c.bb["offsets"] = []
            if self.rest_z is not None:
                c.bb.setdefault("rest_z", self.rest_z)
            if target == "grip":
                c.monitor.disarm()
                g = self.node["grip"]
                g.n = min(max(self.since_switch - 1, 0), g.steps)
        elif target in ("approach", "descend"):
            c.monitor.disarm()
            c.bb["offsets"] = []
        elif target == "release":
            nd = self.node["release"]
            nd.n = min(max(self.since_switch - 1, 0), nd.steps)
        self.jumps.append(f"{self.n}:{have}->{target}")

    def act(self, obs: Observation) -> Action:
        self.n += 1
        if not self.runner.status.running:
            self._fresh()
            self.jumps.append(f"{self.n}:restart")
        c = self.ctx
        c.update(obs)
        if self.rest_z is None and c.obj_pos is not None:
            self.rest_z = float(c.obj_pos[2])
        want = self.decide(obs)
        if want is not None:
            self._jump(want, obs)
        c.update = lambda _obs: None  # already filtered this step; Runner.act would do it again
        try:
            return self.runner.act(obs)
        finally:
            del c.update  # back to the class method
