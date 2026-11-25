"""ToppleChain v0.1: stand dominoes on their spots, fix what gets disturbed, then topple the chain.

For a task whose success is `all_of` `object_toppled` conditions that share one `direction`, and
whose scene has a region `spot_<domino>` per domino (where it must stand, its centre at the
domino's resting centre), ToppleChain loops:

  park -> survey -> (repair one domino -> park -> survey)* -> push the first -> watch -> survey ...

- survey: average `survey_steps` observations, then sort every domino into
  ok (standing on its spot, yawed along the lane), misplaced (standing, off its spot or yaw),
  leaning (in between: wait and look again), down (fallen the right way) or down_bad (fallen
  any other way: this version cannot stand one back up, see Limits).
- repair (`ReplaceDomino`): top-down grasp across the domino's width, rate-limited lift, carry
  over everything standing, turn the wrist so the domino arrives at the lane's yaw, lower to
  `drop` above the table, open, retreat. Misplaced dominoes are repaired farthest-first; each at
  most `max_repairs` times. Dominoes still in a staging area are "misplaced" too, so the same
  skill sets up a chain from scratch.
- push: before the fingers reach the domino, every domino is watched (smoothed xy, yaw, tilt);
  if one moves away from what the survey saw, the push is called off and the skill surveys again.
  Then: fingers nearly closed (width ~8 mm, so neither a grasp nor a grasp_miss is reported) and
  lined up along the lane, behind the first standing domino at `push_frac` of its height; the
  reference advances at `push_speed` until that domino tips past `tipped` or `push_travel` is used.
- watch: wait for the chain to stop moving, then survey again. A chain that stalled (or was
  partly knocked over early) is nudged at its first standing domino, at most `max_nudges` times.

Disturbances (SceneSpec.disturbances) are handled by construction: every decision is taken on a
fresh survey, so a domino moved before, during or after a repair is found and repaired again.

Limits (v0.1): a domino knocked over sideways or backwards is reported and left (standing a
flat domino back up needs a regrasp/pivot primitive); a nudge behind a domino whose predecessor
leans on it may hit that predecessor. Units: metres, radians. Loadable as a `SkillSource` (imports
only `skillforge.*` and numpy).
"""

from __future__ import annotations

import math

import numpy as np

from skillforge.contracts import (
    Action,
    AllOf,
    ObjectToppled,
    Observation,
    TaskSpec,
    success_conditions,
)
from skillforge.skills.grasp import DOWN, quat_angle, quat_from_axis_angle, quat_mul, quat_to_mat
from skillforge.skills.primitives import (
    DONE,
    RUNNING,
    Context,
    Do,
    HoldMonitor,
    MoveTo,
    Node,
    Runner,
    Sequence,
    fail,
    grip,
    hold_check,
    release,
)

# ---------------------------------------------------------------- geometry


def wrap_half_pi(a: float) -> float:
    """Angle modulo pi, in (-pi/2, pi/2]: a domino looks the same turned half a revolution."""
    a = (a + math.pi / 2) % math.pi - math.pi / 2
    return math.pi / 2 if math.isclose(a, -math.pi / 2, abs_tol=1e-12) else a


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def up_of(q: np.ndarray) -> np.ndarray:
    return quat_to_mat(q)[:, 2]


def tilt(q: np.ndarray) -> float:
    return float(np.arccos(np.clip(up_of(q)[2], -1.0, 1.0)))


def heading_x(q: np.ndarray) -> float:
    """Heading of the body x axis on the world xy plane (a domino's face normal when upright)."""
    x = quat_to_mat(q)[:, 0]
    return math.atan2(x[1], x[0])


def hand_quat(yaw: float) -> np.ndarray:
    """Gripper pointing down, turned by `yaw` about world z from DOWN (DOWN closes along world y)."""
    return quat_mul(quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw), DOWN)


def nearest_turn(want: float, current: float, period: float) -> float:
    """The angle equal to `want` modulo `period` closest to `current` (keeps the wrist from spinning)."""
    return want + period * round((current - want) / period)


# ---------------------------------------------------------------- nodes


class Ramp(MoveTo):
    """MoveTo whose commanded point is a reference that advances toward the goal at most `step`
    metres per control step (the measured EE lags its target, so stepping from it would crawl)."""

    def __init__(self, goal, step: float, eps: float = 0.004, **kw):
        self.goal, self.step, self.eps = goal, step, eps
        self.ref: np.ndarray | None = None
        self.g: np.ndarray | None = None
        self.arrived = False
        super().__init__(self._target, tol=1e-9, until=self._near, **kw)

    def reset(self) -> None:
        super().reset()
        self.ref, self.g, self.arrived = None, None, False

    def _target(self, obs, ctx):
        # MoveTo evaluates the target before `until`, once per tick: the goal is read once here
        self.g = np.asarray(self.goal(obs, ctx) if callable(self.goal) else self.goal, dtype=float)
        if self.ref is None:
            self.ref = obs.ee_pos.copy()
        d = self.g - self.ref
        n = float(np.linalg.norm(d))
        self.arrived = n <= self.step
        self.ref = self.g.copy() if self.arrived else self.ref + d * (self.step / n)
        return self.ref

    def _near(self, obs, ctx):
        return self.arrived and float(np.linalg.norm(obs.ee_pos - self.g)) < self.eps


class Turn(Node):
    """Hold the EE still (gripper fixed) until its orientation is within `tol` of `ctx.quat`."""

    def __init__(self, gripper: float, tol: float = 0.04, timeout: int = 60, check=None, name: str = "turn"):
        self.gripper, self.tol, self.timeout, self.check, self.name = gripper, tol, timeout, check, name
        self.n, self.at = 0, None

    def reset(self) -> None:
        self.n, self.at = 0, None

    def tick(self, obs, ctx):
        if self.at is None:
            self.at = obs.ee_pos.copy()
        if self.check is not None and (why := self.check(obs, ctx)):
            return None, fail(why)
        if quat_angle(obs.ee_quat, ctx.quat) < self.tol:
            return None, DONE
        self.n += 1
        if self.n > self.timeout:
            return None, fail(f"timeout:{self.name}")
        return Action(ee_pos=self.at, ee_quat=ctx.quat, gripper=self.gripper), RUNNING


# ---------------------------------------------------------------- one repair


class ReplaceDomino:
    """Pick one standing domino from wherever it is and stand it on `spot` at `yaw` (mod pi)."""

    grasp_depth = 0.022  # fingertip centre this far below the domino's top (palm clears the top)
    hover = 0.10
    lift_step = 0.006  # metres per step (0.12 m/s at 20 Hz): an upright domino is held by friction
    carry_step = 0.012
    lower_step = 0.004
    drop = 0.003  # release with the domino's bottom this far above the table
    clearance = 0.03  # held domino's bottom above every standing domino's top during the carry
    settle_steps = 10  # > the largest action latency
    offset_window = 8

    def __init__(self, task: TaskSpec, obj: str, spot: np.ndarray, yaw: float, others: list[str]):
        o = task.scene.object(obj)
        self.obj, self.spot, self.yaw, self.others = obj, np.asarray(spot, dtype=float), float(yaw), list(others)
        self.hh = float(o.size[2])
        self.tallest = max((2 * float(task.scene.object(n).size[2]) for n in others), default=2 * self.hh)
        self.ctx = Context(obj=obj, monitor=HoldMonitor(dev_tol=0.03, dev_hard=0.05))
        self.runner = Runner(self._build(), self.ctx)

    @property
    def status(self):
        return self.runner.status

    def act(self, obs: Observation) -> Action:
        return self.runner.act(obs)

    def _build(self):
        hh, st = self.hh, self.settle_steps
        below = 2 * hh - self.grasp_depth  # domino length hanging below the fingertip centre
        carry_z = self.tallest + self.clearance + below

        def grasp_target(obs, c):
            # The domino stands on the table, so its grasp height is known; xy and yaw are observed
            # (live, so a domino moved while the hand comes down is followed).
            p = c.obj_pos.copy()
            p[2] = 2 * hh - self.grasp_depth
            # Wrist yaw: the first choice is the one closest to DOWN (keeps the last joint mid-range,
            # where the IK is happy); after that follow it, so noise near +-pi/2 does not flip it.
            dom = heading_x(c.obj_quat)
            ref = c.bb.setdefault("grasp_yaw", wrap_half_pi(dom))
            c.bb["grasp_yaw"] = nearest_turn(dom, ref, math.pi)
            c.quat = hand_quat(c.bb["grasp_yaw"])
            return p

        def above_grasp(obs, c):
            return grasp_target(obs, c) + [0.0, 0.0, self.hover]

        def clear_z(obs, c):
            return np.array([obs.ee_pos[0], obs.ee_pos[1], max(obs.ee_pos[2], carry_z)])

        def note(obs, c):
            c.bb["offsets"], c.bb["rise_xy"] = [], obs.ee_pos[:2].copy()
            return None

        def offset(obs, c):
            # domino centre minus EE, averaged over the carry; one sample per control step
            if c.monitor.armed and c.obj in obs.objects and c.bb.get("offset_step") != c.steps:
                c.bb["offset_step"] = c.steps
                c.bb.setdefault("offsets", []).append(np.asarray(obs.objects[c.obj].pos) - obs.ee_pos)
            offs = c.bb.get("offsets") or []
            return c.monitor.offset if len(offs) < 3 else np.mean(offs[-self.offset_window :], axis=0)

        def risen(obs, c):
            return np.array([*c.bb["rise_xy"], carry_z])

        def aim(obs, c):
            # Wrist yaw that brings the held domino to the lane's yaw: its in-hand yaw offset is
            # constant during the carry, so take it fresh every step.
            hy = _hand_yaw(obs)
            off = wrap_half_pi(heading_x(obs.objects[c.obj].quat) - hy)
            c.quat = hand_quat(wrap_half_pi(self.yaw - off))  # closest to DOWN, as for the grasp
            return None

        def over_spot(obs, c):
            o = offset(obs, c)
            return np.array([self.spot[0] - o[0], self.spot[1] - o[1], carry_z])

        def on_spot(obs, c):
            o = offset(obs, c)
            return np.array([self.spot[0] - o[0], self.spot[1] - o[1], hh + self.drop - o[2]])

        held = hold_check("grasp_slip")
        return Sequence(
            [
                Ramp(clear_z, self.carry_step, gripper=0.0, timeout=200, name="clear"),
                MoveTo(above_grasp, gripper=0.0, tol=0.006, timeout=250, name="approach"),
                Ramp(grasp_target, self.lower_step, eps=0.004, gripper=0.0, timeout=150, name="descend"),
                grip(settle=st),
                Do(note, name="note"),
                Ramp(risen, self.lift_step, gripper=1.0, check=held, timeout=200, name="lift"),
                Do(aim, name="aim"),
                Ramp(over_spot, self.carry_step, gripper=1.0, check=held, timeout=300, name="carry"),
                Do(aim, name="aim_again"),
                Turn(1.0, check=held, name="turn"),
                Ramp(on_spot, self.lower_step, eps=0.003, gripper=1.0, check=held, timeout=200, name="lower"),
                release(settle=st),
                Ramp(clear_z, self.lift_step, gripper=0.0, timeout=200, name="retreat"),
            ],
            name=f"replace_{self.obj}",
        )


def _hand_yaw(obs: Observation) -> float:
    """Yaw of the gripper relative to DOWN (0 = closing along world y)."""
    return heading_x(obs.ee_quat)


# ---------------------------------------------------------------- the skill


class ToppleChain:
    name = "topple_chain"
    version = "0.1.0"

    survey_steps = 8  # observations averaged per survey (observation noise is per step)
    lat_tol = 0.006  # metres across the lane that count as on the spot
    along_tol = 0.008  # metres along the lane
    yaw_tol = 0.12  # radians
    standing_tilt = 0.15  # below: standing
    down_tilt = 0.5  # above: fallen
    down_dir_tol = 0.8  # radians: a fallen domino whose top points within this of the lane fell right
    max_repairs = 3  # per domino
    park_height = 0.25
    push_frac = 0.75  # push at this fraction of the domino's height
    push_gripper = 0.9  # ~8 mm between the fingers: no grasp (only the front finger touches), no grasp_miss
    push_back = 0.03  # metres behind the domino's back face where the TCP comes down
    push_speed = 0.004  # metres per step (8 cm/s at 20 Hz)
    push_travel = 0.06
    tipped = 0.35  # radians: the pushed domino is going, stop pushing
    watch_steps = 80
    quiet_steps = 10
    quiet_tol = 0.01  # radians of tilt change per step that counts as still
    max_nudges = 4
    abort_xy = 0.008  # metres a smoothed domino position may drift from the survey before a push is called off
    abort_yaw = 0.08  # radians
    abort_alpha = 0.3

    def reset(self, task: TaskSpec) -> None:
        conds = success_conditions(task.success)
        if not isinstance(task.success, AllOf) or not all(isinstance(c, ObjectToppled) for c in conds):
            raise ValueError("ToppleChain needs an all_of task of object_toppled conditions")
        dirs = {c.direction for c in conds}
        if len(dirs) != 1 or None in dirs:
            raise ValueError("ToppleChain needs every object_toppled condition to share one direction")
        d = np.array([*next(iter(dirs)), 0.0])
        self.d = d / np.linalg.norm(d)
        self.n = np.array([-self.d[1], self.d[0], 0.0])  # across the lane
        self.yaw = wrap_half_pi(math.atan2(self.d[1], self.d[0]))  # a domino's face normal along the lane
        self.task = task
        self.spots: dict[str, np.ndarray] = {}
        for c in conds:
            o = task.scene.object(c.object)
            if o.shape != "box":
                raise ValueError(f"ToppleChain: '{o.name}' is not a box")
            try:
                self.spots[o.name] = np.asarray(task.scene.region(f"spot_{o.name}").center, dtype=float)
            except KeyError:
                raise ValueError(f"ToppleChain needs a region 'spot_{o.name}'") from None
        self.order = sorted(self.spots, key=lambda k: float(self.spots[k] @ self.d))  # first to fall first
        self.size = {k: np.asarray(task.scene.object(k).size, dtype=float) for k in self.order}
        self.park = self.spots[self.order[0]] - 0.05 * self.d  # above the lane, behind its first spot
        self.park[2] = self.park_height
        self.repairs = {k: 0 for k in self.order}
        self.nudges = 0
        self.log: list[str] = []
        self.phase, self.n_phase = "park", 0
        self.samples: list[Observation] = []
        self.state: dict[str, str] = {}
        self.job: ReplaceDomino | None = None
        self.push: dict | None = None
        self.last_tilts: dict[str, float] | None = None
        self.quiet = 0
        self.steps = 0
        self.park_quat: np.ndarray | None = None

    # -- dispatch --------------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        self.steps += 1
        self.n_phase += 1
        return getattr(self, f"_{self.phase}")(obs)

    def _go(self, phase: str, why: str = "") -> None:
        self.log.append(f"{self.steps}:{phase}{':' + why if why else ''}")
        self.phase, self.n_phase = phase, 0

    def _hold(self, obs: Observation, gripper: float = 0.0) -> Action:
        if self.park_quat is None:
            self.park_quat = hand_quat(nearest_turn(self.yaw, _hand_yaw(obs), math.pi))
        return Action(ee_pos=self.park.copy(), ee_quat=self.park_quat, gripper=gripper)

    # -- phases ----------------------------------------------------------------

    def _park(self, obs: Observation) -> Action:
        a = self._hold(obs)
        up = obs.ee_pos[2] < self.park_height - 0.02
        if up:  # straight up first, so the hand never sweeps through the chain
            a.ee_pos = np.array([obs.ee_pos[0], obs.ee_pos[1], self.park_height])
        elif float(np.linalg.norm(obs.ee_pos - self.park)) < 0.02 or self.n_phase > 200:
            self.samples = []
            self._go("survey")
        return a

    def _survey(self, obs: Observation) -> Action:
        self.samples.append(obs)
        if len(self.samples) >= self.survey_steps:
            self.state = self.classify(self.samples)
            self._decide(obs)
        return self._hold(obs)

    def classify(self, samples: list[Observation]) -> dict[str, str]:
        """ok / misplaced / leaning / down / down_bad per domino, from averaged observations."""
        out = {}
        for k in self.order:
            p = np.mean([s.objects[k].pos for s in samples], axis=0)
            q = samples[-1].objects[k].quat
            t = tilt(q)
            if t > self.down_tilt:
                top = up_of(q)[:2]
                ang = math.acos(float(np.clip(top @ self.d[:2] / max(np.linalg.norm(top), 1e-9), -1, 1)))
                out[k] = "down" if ang <= self.down_dir_tol else "down_bad"
            elif t > self.standing_tilt:
                out[k] = "leaning"
            else:
                e = p - self.spots[k]
                yaw_err = wrap_half_pi(heading_x(q) - self.yaw)
                off = abs(float(e @ self.d)) > self.along_tol or abs(float(e @ self.n)) > self.lat_tol
                out[k] = "misplaced" if off or abs(yaw_err) > self.yaw_tol else "ok"
        return out

    def _decide(self, obs: Observation) -> None:
        st = self.state
        summary = ",".join(f"{k}={v}" for k, v in st.items())
        if any(v == "leaning" for v in st.values()) and self.n_phase < 6 * self.survey_steps:
            self.samples = []  # something is still moving: look again
            return
        standing = [k for k in self.order if st[k] in ("ok", "misplaced")]
        fix = [k for k in standing if st[k] == "misplaced" and self.repairs[k] < self.max_repairs]
        fallen = [k for k in self.order if st[k] in ("down", "down_bad", "leaning")]
        if fix:
            k = fix[-1]  # farthest along the lane first
            self.repairs[k] += 1
            others = [o for o in self.order if o != k and st[o] in ("ok", "misplaced")]
            self.job = ReplaceDomino(self.task, k, self.spots[k], self.yaw, others)
            self._go("repair", f"{k}#{self.repairs[k]} [{summary}]")
            return
        if not standing:
            self._go("idle", f"nothing standing [{summary}]")
            return
        if fallen and self.nudges >= self.max_nudges:
            self._go("idle", f"out of nudges [{summary}]")
            return
        k = standing[0]
        if fallen:
            self.nudges += 1
        self._start_push(obs, k)
        self._go("push", f"{k} [{summary}]")

    def _repair(self, obs: Observation) -> Action:
        a = self.job.act(obs)
        st = self.job.status
        if not st.running:
            self._go("park", f"repair {'failed: ' + st.reason if st.failed else 'done'}")
        return a

    # -- push and watch -----------------------------------------------------------

    def _start_push(self, obs: Observation, k: str) -> None:
        p = np.mean([s.objects[k].pos for s in self.samples], axis=0)
        th, hh = float(self.size[k][0]), float(self.size[k][2])
        z = 2 * hh * self.push_frac
        back = p - self.d * (th + self.push_back)
        back[2] = z
        # fingers lined up along the lane: only the front one meets the domino
        q = hand_quat(nearest_turn(self.yaw + math.pi / 2, _hand_yaw(obs), math.pi))
        ref = {}
        for j in self.order:
            ref[j] = (np.mean([s.objects[j].pos for s in self.samples], axis=0)[:2],
                      heading_x(self.samples[-1].objects[j].quat), tilt(self.samples[-1].objects[j].quat))
        self.push = {"obj": k, "back": back, "quat": q, "stage": "above", "s": 0.0, "n": 0, "ref": ref,
                     "ema": {j: r[0].copy() for j, r in ref.items()}}

    def _moved(self, obs: Observation) -> str | None:
        """A domino that moved since the survey (smoothed, so observation noise does not count)."""
        P, a = self.push, self.abort_alpha
        for j, (xy, yaw, t0) in P["ref"].items():
            o = obs.objects[j]
            P["ema"][j] = (1 - a) * P["ema"][j] + a * o.pos[:2]
            if float(np.linalg.norm(P["ema"][j] - xy)) > self.abort_xy:
                return f"{j} moved"
            if abs(tilt(o.quat) - t0) > self.standing_tilt:
                return f"{j} tilted"
            if t0 < self.standing_tilt and abs(wrap_half_pi(heading_x(o.quat) - yaw)) > self.abort_yaw:
                return f"{j} turned"
        return None

    def _push(self, obs: Observation) -> Action:
        P, g = self.push, self.push_gripper
        P["n"] += 1
        above = P["back"] + [0.0, 0.0, 0.10]
        if P["stage"] in ("above", "down") and (why := self._moved(obs)):
            self._go("park", f"push called off: {why}")
            return Action(ee_pos=np.array([obs.ee_pos[0], obs.ee_pos[1], self.park_height]), ee_quat=P["quat"], gripper=g)
        if P["stage"] == "above":
            if float(np.linalg.norm(obs.ee_pos - above)) < 0.006 and quat_angle(obs.ee_quat, P["quat"]) < 0.05:
                P["stage"], P["tcp"] = "down", obs.ee_pos.copy()
            return Action(ee_pos=above, ee_quat=P["quat"], gripper=g)
        if P["stage"] == "down":
            d = P["back"] - P["tcp"]
            n = float(np.linalg.norm(d))
            P["tcp"] = P["back"].copy() if n <= 0.004 else P["tcp"] + d * (0.004 / n)
            if n <= 0.004 and float(np.linalg.norm(obs.ee_pos - P["back"])) < 0.004:
                P["stage"] = "push"
            return Action(ee_pos=P["tcp"], ee_quat=P["quat"], gripper=g)
        if P["stage"] == "push":
            P["s"] = min(P["s"] + self.push_speed, self.push_travel)
            k = P["obj"]
            if tilt(obs.objects[k].quat) > self.tipped or P["s"] >= self.push_travel:
                P["stage"] = "up"
                self.log.append(f"{self.steps}:pushed:{k}:tilt={tilt(obs.objects[k].quat):.2f}")
            return Action(ee_pos=P["back"] + self.d * P["s"], ee_quat=P["quat"], gripper=g)
        # up: straight up, then watch from the park pose
        tgt = np.array([obs.ee_pos[0], obs.ee_pos[1], self.park_height])
        if obs.ee_pos[2] > self.park_height - 0.03 or P["n"] > 400:
            self.last_tilts, self.quiet = None, 0
            self._go("watch")
        return Action(ee_pos=tgt, ee_quat=P["quat"], gripper=g)

    def _watch(self, obs: Observation) -> Action:
        tilts = {k: tilt(obs.objects[k].quat) for k in self.order}
        if self.last_tilts is not None:
            moving = max(abs(tilts[k] - self.last_tilts[k]) for k in self.order) > self.quiet_tol
            self.quiet = 0 if moving else self.quiet + 1
        self.last_tilts = tilts
        if self.quiet >= self.quiet_steps or self.n_phase > self.watch_steps:
            self._go("park", "chain still")
        return self._hold(obs)

    def _idle(self, obs: Observation) -> Action:
        return self._hold(obs)
