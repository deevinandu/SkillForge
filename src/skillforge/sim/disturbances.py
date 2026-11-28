"""Mid-episode disturbances (`SceneSpec.disturbances`): which object, when, by how much.

Physics-free on purpose: every env builds one `DisturbanceSchedule` at reset and asks it each step
which disturbances are due; the env applies them (teleport, angular kick) its own way.

Randomness: everything is drawn at reset from `np.random.default_rng([seed, 2])`, a stream of its
own, so adding disturbances to a task changes none of its other draws (the scene stream is
`default_rng(seed)`, observation noise `default_rng([seed, 1])`). Same seed, same disturbances.

Timing: `update(step, events)` is called once per env step with the 1-based step index and the
events the env emitted this step (before any disturbance). A disturbance without `after_event` is
due at step `max(1, at_step)`; with one, `at_step` steps after the step whose events first match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from skillforge.contracts import Disturbance, TaskSpec

from .scatter import region_bounds

MAX_WAIT = 40  # steps a due disturbance waits for its object to be let go before it is skipped
RELOCATE_CANDIDATES = 12  # spots drawn per relocate; the env takes the first free one
RELOCATE_CLEARANCE = 0.09  # metres from every other object for a relocate spot


@dataclass
class Planned:
    """One disturbance of one object, drawn at reset."""

    index: int  # position in SceneSpec.disturbances
    sub: int  # which of its `count` objects
    spec: Disturbance
    obj: str | None  # None until the triggering event names it (target="event_object")
    fires: bool  # the `prob` draw
    delay: int  # steps after reset / after the trigger event
    dx: float
    dy: float
    dyaw: float
    ang_vel: float
    heading: float  # knock: world-xy direction (radians) the object's top is pushed toward
    face: str = "flat"  # relocate: the face it is put down on
    yaw_abs: float = 0.0  # relocate: its yaw once put down
    cands: list[np.ndarray] = field(default_factory=list)  # relocate: candidate xy spots
    due: int | None = None  # absolute step at which it is due (None: trigger not seen yet)
    state: str = "waiting"  # waiting | applied | skipped | cancelled
    waited: int = 0

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def key(self) -> str:
        return f"disturb{self.index}.{self.sub}"


class DisturbanceSchedule:
    def __init__(self, task: TaskSpec, seed: int):
        rng = np.random.default_rng([seed, 2])
        self.movable = [o.name for o in task.scene.objects if not o.fixed]
        self.plans: list[Planned] = []
        self.sample: dict[str, float] = {}
        self._history: list[str] = []
        for i, d in enumerate(task.scene.disturbances):
            fires = bool(rng.uniform() < d.prob)
            if d.target == "event_object":
                chosen: list[str | None] = [None]
            else:
                cands = list(d.objects) or self.movable
                chosen = [cands[j] for j in rng.permutation(len(cands))[: min(d.count, len(cands))]]
            for j, obj in enumerate(chosen):
                # a fixed number of draws per plan, whatever the kind, so the stream stays aligned
                delay = int(round(rng.uniform(d.at_step.lo, d.at_step.hi)))
                r, th = rng.uniform(d.xy.lo, d.xy.hi), rng.uniform(-math.pi, math.pi)
                yaw = rng.uniform(d.yaw.lo, d.yaw.hi) * (1.0 if rng.uniform() < 0.5 else -1.0)
                w = rng.uniform(d.ang_vel.lo, d.ang_vel.hi)
                head = rng.uniform(-math.pi, math.pi)
                if d.knock_dir is not None:
                    head = math.atan2(d.knock_dir[1], d.knock_dir[0])
                p = Planned(
                    index=i, sub=j, spec=d, obj=obj, fires=fires, delay=delay,
                    dx=float(r * math.cos(th)), dy=float(r * math.sin(th)), dyaw=float(yaw),
                    ang_vel=float(w), heading=float(head),
                )
                if d.kind == "relocate":  # extra draws only for relocate, so other kinds keep theirs
                    p.face = d.faces[int(rng.integers(len(d.faces)))]
                    p.yaw_abs = float(rng.uniform(-d.max_yaw, d.max_yaw))
                    for _ in range(RELOCATE_CANDIDATES if d.regions else 0):
                        lo, hi = region_bounds(task, d.regions[int(rng.integers(len(d.regions)))])
                        p.cands.append(rng.uniform(lo, hi))
                if d.after_event is None:
                    p.due = max(1, delay)
                self.plans.append(p)
                k = p.key
                self.sample[f"{k}.fires"] = float(fires)
                self.sample[f"{k}.obj"] = float(self.movable.index(obj)) if obj is not None else -1.0
                self.sample[f"{k}.step"] = float(delay)
                self.sample[f"{k}.dx"], self.sample[f"{k}.dy"] = p.dx, p.dy
                self.sample[f"{k}.dyaw"], self.sample[f"{k}.ang_vel"] = p.dyaw, p.ang_vel
                self.sample[f"{k}.heading"] = p.heading
                if d.kind == "relocate":
                    self.sample[f"{k}.face"] = float(("upright", "flat", "side").index(p.face))
                    self.sample[f"{k}.yaw_abs"] = p.yaw_abs

    def __bool__(self) -> bool:
        return bool(self.plans)

    def _happened(self, prefix: str) -> bool:
        return any(e.startswith(prefix) for e in self._history)

    def update(self, step: int, events: list[str]) -> list[Planned]:
        """Record this step's events and return the disturbances to apply now. The env must then
        call `applied(p)` or `blocked(p)` for each one it was given."""
        self._history.extend(events)
        for p in self.plans:
            if p.state != "waiting" or p.due is not None:
                continue
            ev = next((e for e in events if e.startswith(p.spec.after_event)), None)
            if ev is None:
                continue
            p.due = step + p.delay
            if p.spec.target == "event_object":
                name = ev.split(":", 1)[1] if ":" in ev else ""
                if name not in self.movable:
                    p.state = "cancelled"
                    continue
                p.obj = name
                self.sample[f"{p.key}.obj"] = float(self.movable.index(name))
        out = []
        for p in self.plans:
            if p.state != "waiting" or p.due is None or step < p.due:
                continue
            if not p.fires or (p.spec.before_event and self._happened(p.spec.before_event)):
                p.state = "cancelled"
                continue
            out.append(p)
        return out

    def applied(self, p: Planned) -> str:
        p.state = "applied"
        return f"disturbance:{p.kind}:{p.obj}"

    def blocked(self, p: Planned) -> str | None:
        """The object could not be disturbed this step (held, or no room). Returns the skip event
        once it has waited too long, else None (it is retried next step)."""
        p.waited += 1
        if p.waited > MAX_WAIT:
            p.state = "skipped"
            return f"disturbance_skipped:{p.obj}"
        return None

    def skipped(self, p: Planned) -> str:
        p.state = "skipped"
        return f"disturbance_skipped:{p.obj}"


def yaw_quat(dyaw: float) -> np.ndarray:
    return np.array([math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def knock_axis(heading: float) -> np.ndarray:
    """World-frame rotation axis that moves an upright body's top toward `heading`: z x dir."""
    return np.array([-math.sin(heading), math.cos(heading), 0.0])
