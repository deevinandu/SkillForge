"""Composable closed-loop skill primitives and a small phase machine to build skills from them.

Every primitive is a `Node`: each control step it is ticked with the latest `Observation` and a
shared `Context`, and returns `(action, status)`. Status is running, done or failed(reason).
A node that is done may return `action=None`, meaning "I have nothing more to command"; the
enclosing `Sequence` then ticks the next node in the same control step, so no step is wasted.

Composition:
- `Sequence(nodes)`: run in order; fail as soon as one fails.
- `Retry(node, budgets)`: on failure, reset and rerun the node while the failure's reason has
  budget left (`budgets` maps a reason prefix such as "grasp_miss" to a number of retries).
- `Fallback(nodes)`: try each until one succeeds.
- `Runner(root)`: turns a tree into the `act(obs) -> Action` a `Skill` needs.

Primitives use only `Observation` fields every env provides (ee_pos, gripper_width, objects)
plus `contact_force` when present, and command only `Action(mode="ee")`, so they run the same in
`StubEnv` and a physics env. Failure reasons start with a contracts `FailureTag` where one fits
("grasp_miss:...", "grasp_slip:...", "wrong_placement:...", "timeout:...").

Known limit of the signals: `StubEnv` reports gripper_width = 0 whenever the gripper is closed,
held object or not, so width alone cannot tell a grasp from a miss there. Hold checks therefore
combine (a) the object's observed position tracking the end effector and (b) a width collapse
relative to the width measured right after closing (informative in physics, inert in the stub).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from skillforge.contracts import Action, Observation

from .grasp import DOWN, quat_to_mat

# ---------------------------------------------------------------- status


@dataclass(frozen=True)
class Status:
    state: Literal["running", "done", "failed"]
    reason: str = ""

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def done(self) -> bool:
        return self.state == "done"

    @property
    def failed(self) -> bool:
        return self.state == "failed"


RUNNING = Status("running")
DONE = Status("done")


def fail(reason: str) -> Status:
    return Status("failed", reason)


# ---------------------------------------------------------------- shared state


@dataclass
class HoldMonitor:
    """Decides whether the gripper still holds the object.

    `arm()` right after closing records the width and the object-to-EE offset; `check()` then
    returns a reason string when the hold is lost, else None. `rise` is how far the (filtered)
    object rose above its height at `arm()`: if it clearly rose before being lost, the object was
    held and then slipped; if not, the grasp never took (a miss).
    """

    width_eps: float = 0.004  # metres: below this the fingers are (nearly) touching
    dev_tol: float = 0.03  # metres: offset drift that counts as lost, if sustained
    dev_hard: float = 0.05  # metres: offset drift that counts as lost immediately
    sustain: int = 3  # steps above dev_tol before declaring a loss
    rise_held: float = 0.02  # metres: object rise that proves it was held
    armed: bool = False
    w0: float = 0.0
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    z0: float = 0.0
    rise: float = 0.0
    _bad: int = 0

    def arm(self, obs: Observation, obj_pos: np.ndarray) -> None:
        self.armed = True
        self.w0 = float(obs.gripper_width)
        self.offset = np.asarray(obj_pos, dtype=float) - obs.ee_pos
        self.z0, self.rise = float(obj_pos[2]), 0.0
        self._bad = 0

    def disarm(self) -> None:
        self.armed = False
        self._bad = 0

    @property
    def was_held(self) -> bool:
        return self.rise > self.rise_held

    def check(self, obs: Observation, obj: str, obj_pos: np.ndarray | None = None) -> str | None:
        """`obj_pos`: filtered object position for the rise estimate (raw observation if None)."""
        if not self.armed:
            return None
        z = float((obs.objects[obj].pos if obj_pos is None else obj_pos)[2])
        self.rise = max(self.rise, z - self.z0)
        if self.w0 > self.width_eps and obs.gripper_width < self.width_eps:
            return "width_collapsed"
        dev = float(np.linalg.norm(obs.objects[obj].pos - obs.ee_pos - self.offset))
        if dev > self.dev_hard:
            return f"object_left_gripper({dev:.3f}m)"
        self._bad = self._bad + 1 if dev > self.dev_tol else 0
        if self._bad >= self.sustain:
            return f"object_drifting({dev:.3f}m)"
        return None


@dataclass
class Context:
    """Per-episode state shared by every node of a skill.

    `obj_pos` is a low-pass filtered copy of the target object's observed position (it snaps to
    the raw value on jumps larger than `snap`, e.g. after a drop). Nodes read it for targets;
    hold checks read the raw observation instead, since the filter lags a moving object.
    """

    obj: str | None = None
    quat: np.ndarray = field(default_factory=lambda: DOWN.copy())  # current gripper orientation target
    bb: dict = field(default_factory=dict)  # free-form blackboard for skill-specific values
    monitor: HoldMonitor = field(default_factory=HoldMonitor)
    alpha: float = 0.4
    snap: float = 0.02
    obj_pos: np.ndarray | None = None
    obj_quat: np.ndarray | None = None
    steps: int = 0
    log: list[str] = field(default_factory=list)  # phase transitions and retries, for debugging

    def update(self, obs: Observation) -> None:
        self.steps += 1
        if self.obj is None or self.obj not in obs.objects:
            return
        raw = obs.objects[self.obj]
        if self.obj_pos is None or np.linalg.norm(raw.pos - self.obj_pos) > self.snap:
            self.obj_pos = np.asarray(raw.pos, dtype=float).copy()
        else:
            self.obj_pos = (1 - self.alpha) * self.obj_pos + self.alpha * raw.pos
        self.obj_quat = np.asarray(raw.quat, dtype=float).copy()


Vec = np.ndarray | Callable[[Observation, Context], np.ndarray]
Check = Callable[[Observation, Context], str | None]


def _resolve(v, obs: Observation, ctx: Context) -> np.ndarray:
    return np.asarray(v(obs, ctx) if callable(v) else v, dtype=float)


# ---------------------------------------------------------------- nodes


class Node:
    name = "node"

    def reset(self) -> None:
        """Forget progress so the node can run again from the start."""

    def tick(self, obs: Observation, ctx: Context) -> tuple[Action | None, Status]:
        raise NotImplementedError


class MoveTo(Node):
    """Drive the EE toward `target` until within `tol` (or `until` holds), with `gripper` fixed.

    `target`/`quat` may be arrays or callables of (obs, ctx), re-evaluated every step (closed loop).
    `quat=None` uses `ctx.quat`. `check` may abort with a failure reason (e.g. a lost hold).
    """

    def __init__(
        self,
        target: Vec,
        gripper: float = 0.0,
        quat: Vec | None = None,
        tol: float = 0.006,
        timeout: int = 200,
        until: Callable[[Observation, Context], bool] | None = None,
        check: Check | None = None,
        name: str = "move_to",
    ):
        self.target, self.gripper, self.quat = target, gripper, quat
        self.tol, self.timeout, self.until, self.check, self.name = tol, timeout, until, check, name
        self.n = 0

    def reset(self) -> None:
        self.n = 0

    def tick(self, obs, ctx):
        self.n += 1
        if self.check is not None and (why := self.check(obs, ctx)):
            return None, fail(why)
        tgt = _resolve(self.target, obs, ctx)
        if np.linalg.norm(obs.ee_pos - tgt) < self.tol or (self.until is not None and self.until(obs, ctx)):
            return None, DONE
        if self.n > self.timeout:
            return None, fail(f"timeout:{self.name}")
        q = ctx.quat if self.quat is None else _resolve(self.quat, obs, ctx)
        return Action(ee_pos=tgt, ee_quat=q, gripper=self.gripper), RUNNING


class Hold(Node):
    """Keep the EE where it was on the first tick and command `gripper` for `steps` steps.

    `check` runs every step and may fail the node; `on_done` runs once at the end and may too.
    """

    def __init__(
        self,
        gripper: float,
        steps: int = 4,
        check: Check | None = None,
        on_done: Check | None = None,
        name: str = "hold",
    ):
        self.gripper, self.steps, self.check, self.on_done, self.name = gripper, steps, check, on_done, name
        self.n = 0
        self.at: np.ndarray | None = None

    def reset(self) -> None:
        self.n, self.at = 0, None

    def tick(self, obs, ctx):
        if self.at is None:
            self.at = obs.ee_pos.copy()
        if self.check is not None and (why := self.check(obs, ctx)):
            return None, fail(why)
        if self.n >= self.steps:
            if self.on_done is not None and (why := self.on_done(obs, ctx)):
                return None, fail(why)
            return None, DONE
        self.n += 1
        return Action(ee_pos=self.at, ee_quat=ctx.quat, gripper=self.gripper), RUNNING


class Do(Node):
    """An instantaneous step: run `fn(obs, ctx)`; a returned string fails the node."""

    def __init__(self, fn: Check, name: str = "do"):
        self.fn, self.name = fn, name

    def tick(self, obs, ctx):
        why = self.fn(obs, ctx)
        return None, fail(why) if why else DONE


class Sequence(Node):
    def __init__(self, nodes: list[Node], name: str = "sequence"):
        self.nodes, self.name = list(nodes), name
        self.i = 0

    def reset(self) -> None:
        self.i = 0
        for n in self.nodes:
            n.reset()

    def tick(self, obs, ctx):
        while self.i < len(self.nodes):
            node = self.nodes[self.i]
            act, st = node.tick(obs, ctx)
            if st.failed:
                return None, fail(st.reason)
            if st.running:
                return act, RUNNING
            ctx.log.append(f"{ctx.steps}:done:{node.name}")
            self.i += 1
        return None, DONE


class Retry(Node):
    """Rerun `node` after a failure whose reason starts with a key of `budgets` that has retries left.

    `budgets` may be an int (that many retries for any failure). `on_retry(ctx, reason)` runs
    before each rerun, e.g. to pick a different grasp.
    """

    def __init__(
        self,
        node: Node,
        budgets: int | dict[str, int] = 1,
        on_retry: Callable[[Context, str], None] | None = None,
        name: str = "retry",
    ):
        self.node, self.on_retry, self.name = node, on_retry, name
        self.budgets = {"": budgets} if isinstance(budgets, int) else dict(budgets)
        self.used: dict[str, int] = {}

    def reset(self) -> None:
        self.used = {}
        self.node.reset()

    def _key(self, reason: str) -> str | None:
        for k in sorted(self.budgets, key=len, reverse=True):  # longest matching prefix wins
            if reason.startswith(k) and self.used.get(k, 0) < self.budgets[k]:
                return k
        return None

    def tick(self, obs, ctx):
        for _ in range(1 + sum(self.budgets.values())):
            act, st = self.node.tick(obs, ctx)
            if not st.failed:
                return act, st
            k = self._key(st.reason)
            if k is None:
                return None, st
            self.used[k] = self.used.get(k, 0) + 1
            ctx.log.append(f"{ctx.steps}:retry:{st.reason}")
            ctx.bb.setdefault("retries", []).append(st.reason)
            if self.on_retry is not None:
                self.on_retry(ctx, st.reason)
            self.node.reset()
        return None, fail("retry_loop")


class Fallback(Node):
    """Try each node in turn until one finishes; fail with the last reason if all fail."""

    def __init__(self, nodes: list[Node], name: str = "fallback"):
        self.nodes, self.name = list(nodes), name
        self.i = 0

    def reset(self) -> None:
        self.i = 0
        for n in self.nodes:
            n.reset()

    def tick(self, obs, ctx):
        why = "empty_fallback"
        while self.i < len(self.nodes):
            act, st = self.nodes[self.i].tick(obs, ctx)
            if not st.failed:
                return act, st
            why = st.reason
            ctx.log.append(f"{ctx.steps}:fallback:{why}")
            self.i += 1
        return None, fail(why)


class Runner:
    """Ticks a tree once per control step. After the tree finishes (done or failed) it holds still.

    On failure the gripper keeps its last command, so a held object is not dropped on the floor.
    """

    def __init__(self, root: Node, ctx: Context):
        self.root, self.ctx = root, ctx
        self.status = RUNNING
        self.last: Action | None = None
        self._park: np.ndarray | None = None

    def act(self, obs: Observation) -> Action:
        self.ctx.update(obs)
        if self.status.running:
            act, st = self.root.tick(obs, self.ctx)
            self.status = st
            if act is not None:
                self.last = act
                return act
            if not st.failed:
                self.ctx.log.append(f"{self.ctx.steps}:finished")
            else:
                self.ctx.log.append(f"{self.ctx.steps}:failed:{st.reason}")
        if self._park is None:
            self._park = obs.ee_pos.copy()
        grip = self.last.gripper if self.last is not None else 0.0
        return Action(ee_pos=self._park, ee_quat=self.ctx.quat, gripper=grip)


# ---------------------------------------------------------------- primitive factories
#
# Thin, named constructors so skills read like the task: approach -> descend -> grip -> lift ...


def hold_check(label: str) -> Check:
    """A `check` that fails with '<label>:<why>' when the hold monitor reports a lost object."""

    def check(obs: Observation, ctx: Context) -> str | None:
        why = ctx.monitor.check(obs, ctx.obj, ctx.obj_pos) if ctx.obj else None
        return f"{label}:{why}" if why else None

    return check


def move_to(target: Vec, gripper: float = 0.0, **kw) -> MoveTo:
    return MoveTo(target, gripper=gripper, **kw)


def approach(target: Vec, hover: float = 0.10, direction: Vec | None = None, **kw) -> MoveTo:
    """Go to a pre-grasp point `hover` metres back along the approach direction (default: above).

    `direction` is the approach direction (world, unit); None reads it from `ctx.quat`'s z axis.
    """

    def pre(obs, ctx):
        d = _resolve(direction, obs, ctx) if direction is not None else quat_to_mat(ctx.quat)[:, 2]
        return _resolve(target, obs, ctx) - hover * d

    kw.setdefault("name", "approach")
    return MoveTo(pre, gripper=0.0, **kw)


def descend_until(
    target: Vec, contact_force: float | None = 5.0, gripper: float = 0.0, **kw
) -> MoveTo:
    """Move to `target` and stop on arrival, or earlier when |contact_force| exceeds the threshold
    (only if the env reports contact force; otherwise height/arrival alone decides)."""

    def touched(obs, ctx):
        f = obs.contact_force
        return contact_force is not None and f is not None and float(np.linalg.norm(f)) > contact_force

    kw.setdefault("name", "descend")
    return MoveTo(target, gripper=gripper, until=touched, **kw)


def grip(settle: int = 4, miss_dist: float = 0.03, **kw) -> Hold:
    """Close and wait `settle` steps, then arm the hold monitor.

    Fails 'grasp_miss:empty' only when the fingers closed fully AND the object is not at the TCP,
    because a closed width alone is ambiguous in envs (like the stub) that do not report it."""

    def done(obs: Observation, ctx: Context) -> str | None:
        if ctx.obj is None or ctx.obj_pos is None:
            return None
        far = float(np.linalg.norm(ctx.obj_pos - obs.ee_pos))
        if obs.gripper_width < ctx.monitor.width_eps and far > miss_dist:
            return f"grasp_miss:empty({far:.3f}m)"
        ctx.monitor.arm(obs, ctx.obj_pos)
        return None

    kw.setdefault("name", "grip")
    return Hold(1.0, steps=settle, on_done=done, **kw)


def lift(height: float | Callable[[Observation, Context], float], verify_steps: int = 2, **kw) -> Sequence:
    """Raise the EE straight up to world z = `height` with the gripper closed, then dwell
    `verify_steps` steps. A lost object here is reported as 'grasp_miss:not_lifted:...' if it never
    rose with the gripper, else 'grasp_slip:during_lift:...'."""
    xy: dict[str, np.ndarray] = {}

    def tgt(obs, ctx):
        if "xy" not in xy:
            xy["xy"] = obs.ee_pos[:2].copy()
        h = height(obs, ctx) if callable(height) else height
        return np.array([xy["xy"][0], xy["xy"][1], h])

    class _Lift(MoveTo):
        def reset(self):
            super().reset()
            xy.clear()

    def check(obs, ctx):
        why = ctx.monitor.check(obs, ctx.obj, ctx.obj_pos) if ctx.obj else None
        if not why:
            return None
        return f"grasp_slip:during_lift:{why}" if ctx.monitor.was_held else f"grasp_miss:not_lifted:{why}"

    kw.setdefault("name", "lift_up")
    return Sequence(
        [_Lift(tgt, gripper=1.0, check=check, **kw), Hold(1.0, steps=verify_steps, check=check, name="lift_verify")],
        name="lift",
    )


def transport(target: Vec, **kw) -> MoveTo:
    """Carry the held object to `target` (usually above the goal); a lost object fails 'grasp_slip:...'."""
    kw.setdefault("name", "transport")
    return MoveTo(target, gripper=1.0, check=hold_check("grasp_slip"), **kw)


def place(target: Vec, contact_force: float | None = 5.0, **kw) -> MoveTo:
    """Lower the held object to `target` (EE target), stopping early on contact."""
    kw.setdefault("name", "place")
    return descend_until(target, contact_force=contact_force, gripper=1.0, check=hold_check("grasp_slip"), **kw)


def release(settle: int = 4, **kw) -> Hold:
    """Open the gripper and wait `settle` steps; disarm the hold monitor."""

    def done(obs, ctx):
        ctx.monitor.disarm()
        return None

    kw.setdefault("name", "release")
    return Hold(0.0, steps=settle, on_done=done, **kw)


def retreat(height: float | Callable[[Observation, Context], float], gripper: float = 0.0, **kw) -> MoveTo:
    """Move straight up to world z >= `height` from wherever the EE is when this starts."""
    start: dict[str, np.ndarray] = {}

    def tgt(obs, ctx):
        if "p" not in start:
            start["p"] = obs.ee_pos.copy()
        h = height(obs, ctx) if callable(height) else height
        return np.array([start["p"][0], start["p"][1], max(h, start["p"][2])])

    class _Retreat(MoveTo):
        def reset(self):
            super().reset()
            start.clear()

    kw.setdefault("name", "retreat")
    return _Retreat(tgt, gripper=gripper, **kw)
