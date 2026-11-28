"""ToppleChain v0.2: stand dominoes up and on their spots, fix what gets disturbed, then topple.

For a task whose success is `all_of` `object_toppled` conditions that share one `direction`, and
whose scene has a region `spot_<domino>` per domino (where it must stand, its centre at the
domino's resting centre), ToppleChain loops:

  park -> survey -> (stand one up | repair one -> park -> survey)* -> push the first -> watch -> ...

- survey: average `survey_steps` observations, then sort every domino into
  ok (standing on its spot, yawed along the lane), misplaced (standing, off its spot or yaw),
  leaning (in between: wait and look again; still leaning after a while = fallen), down (fallen
  the right way, inside its region) or down_bad (lying or fallen any other way, or anywhere else).
- stand up (`StandUp`; `stand_up = False` turns it off and only reports lying dominoes), before
  the robot's own push, for every down_bad domino, top of a pile first: grasp near its +z end, tilted about the closing axis just enough that palm and wrist
  stay clear of the table both now and once it stands; lift; turn the hand so the domino hangs
  upright (+z end up, chosen from how it actually sits in the fingers); set it down standing on
  its own spot if that is clear, else on the nearest clear point of a grid beside the lane; open,
  back the hand out along its axis. It is then "misplaced" and repaired like any other.
- repair (`ReplaceDomino`): top-down grasp across the domino's width, rate-limited lift, carry
  over everything standing, turn the wrist so the domino arrives at the lane's yaw, lower to
  `drop` above the table, open, retreat. Misplaced dominoes are repaired farthest-first; each at
  most `max_repairs` times. Dominoes still in a staging area are "misplaced" too, so the same
  skill sets up a chain from scratch. A repair whose domino falls over before it is grasped stops.
- push: before the fingers reach the domino, every domino is watched (smoothed xy, yaw, tilt);
  if one moves away from what the survey saw, the push is called off and the skill surveys again.
  Then: fingers nearly closed (width ~8 mm, so neither a grasp nor a grasp_miss is reported) and
  lined up along the lane, behind the first standing domino at `push_frac` of its height; the
  reference advances at `push_speed` until that domino tips past `tipped` or `push_travel` is used.
- watch: wait for the chain to stop moving, then survey again. A chain that stalled (or was
  partly knocked over early) is nudged at its first standing domino, at most `max_nudges` times.

Disturbances (SceneSpec.disturbances) and random starts (SceneSpec.scatter) are handled by
construction: every decision is taken on a fresh survey of what the robot sees (no event from the
sim is used), so a domino moved, turned or laid down before, during or after a repair is found
and rebuilt.

Limits (v0.2): after the robot's own push nothing is rebuilt (dominoes that fell the right way
count, the rest are nudged); a domino standing on its head is treated as fallen; a nudge behind a
domino whose predecessor leans on it may hit that predecessor. Units: metres, radians. Loadable as a `SkillSource` (imports
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
from skillforge.skills.grasp import (
    DOWN,
    mat_to_quat,
    quat_angle,
    quat_from_axis_angle,
    quat_mul,
    quat_to_mat,
)
from skillforge.skills.primitives import (
    DONE,
    RUNNING,
    Context,
    Do,
    HoldMonitor,
    MoveTo,
    Node,
    Retry,
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


class TouchDown(Ramp):
    """Ramp down toward `goal` (and up to `extra` metres past it) and stop as soon as the fingers
    touch something: contact force above `force`, or the EE lagging its reference by `lag` (it is
    resting on the table). For picking a domino that lies flat, where the pads must get as low as
    the fingertips allow and a fixed target height is either above the domino or in the table."""

    def __init__(self, goal, step: float, extra: float = 0.012, force: float = 2.0, lag: float = 0.002,
                 slow: float = 0.001, slow_from: float = 0.02, **kw):
        self.extra, self.force, self.lag = extra, force, lag
        self.slow, self.slow_from = slow, slow_from  # step per tick within `slow_from` of the goal
        self.last_z: float | None = None
        super().__init__(goal, step, **kw)

    def reset(self) -> None:
        super().reset()
        self.last_z = None

    def _target(self, obs, ctx):
        g = np.asarray(self.goal(obs, ctx) if callable(self.goal) else self.goal, dtype=float).copy()
        g[2] -= self.extra
        self.g = g
        if self.ref is None:
            self.ref = obs.ee_pos.copy()
        d = g - self.ref
        n = float(np.linalg.norm(d))
        step = self.slow if self.ref[2] - (g[2] + self.extra) < self.slow_from else self.step
        self.arrived = n <= step
        self.ref = g.copy() if self.arrived else self.ref + d * (step / n)
        return self.ref

    def _near(self, obs, ctx):
        f = obs.contact_force
        if f is not None and float(np.linalg.norm(f)) > self.force:
            ctx.bb["touch"] = f"force {float(np.linalg.norm(f)):.1f}N at z={obs.ee_pos[2]:.4f}"
            return True
        # blocked: the reference is already down near the goal, the EE is above it and has stopped
        z, last = float(obs.ee_pos[2]), self.last_z
        self.last_z = z
        near_goal = self.ref is not None and self.ref[2] <= self.g[2] + self.extra + 0.002
        if near_goal and last is not None and z - self.ref[2] > self.lag and abs(z - last) < 0.0005:
            ctx.bb["touch"] = f"blocked at z={z:.4f}"
            return True
        return self.arrived and float(np.linalg.norm(obs.ee_pos - self.g)) < self.eps


class HandHoldMonitor(HoldMonitor):
    """HoldMonitor whose object-to-EE offset is kept in the hand frame, so turning the wrist with
    the object in it (a stand-up turns it ~90 deg) is not mistaken for the object drifting."""

    def arm(self, obs: Observation, obj_pos: np.ndarray) -> None:
        super().arm(obs, obj_pos)
        self.off_hand = quat_to_mat(obs.ee_quat).T @ self.offset

    def check(self, obs: Observation, obj: str, obj_pos: np.ndarray | None = None) -> str | None:
        if self.armed:
            self.offset = quat_to_mat(obs.ee_quat) @ self.off_hand
        return super().check(obs, obj, obj_pos)


class CarryRamp(Ramp):
    """Ramp that judges a stall: once the reference has reached the goal, if the EE has not got
    closer by `progress` metres in `window` steps, it is done when within `ok` of the goal (the
    next step finishes the job), and fails `stalled:<name>` otherwise (the arm cannot get there
    in this orientation: try another)."""

    def __init__(self, goal, step: float, ok: float = 0.012, window: int = 20, progress: float = 0.001, **kw):
        self.ok, self.window, self.progress = ok, window, progress
        self.errs: list[float] = []
        super().__init__(goal, step, **kw)

    def reset(self) -> None:
        super().reset()
        self.errs = []

    def tick(self, obs, ctx):
        act, st = super().tick(obs, ctx)
        if st.running and self.arrived and self.g is not None:
            e = float(np.linalg.norm(obs.ee_pos - self.g))
            self.errs.append(e)
            if len(self.errs) > self.window and self.errs[-self.window - 1] - e < self.progress:
                if e < self.ok:
                    ctx.bb[f"{self.name}_stalled_at"] = round(e, 4)
                    return None, DONE
                return None, fail(f"stalled:{self.name}({e:.3f}m)")
        return act, st


class Turn(Node):
    """Hold the EE still (gripper fixed) until its orientation is within `tol` of `ctx.quat`."""

    def __init__(self, gripper: float, tol: float = 0.04, timeout: int = 60, check=None, name: str = "turn",
                 rate: float | None = None):
        self.gripper, self.tol, self.timeout, self.check, self.name = gripper, tol, timeout, check, name
        self.rate = rate  # radians per step the commanded orientation may turn (None: command the goal)
        self.n, self.at, self.ref = 0, None, None

    def reset(self) -> None:
        self.n, self.at, self.ref = 0, None, None

    def _command(self, obs, goal: np.ndarray) -> np.ndarray:
        if self.rate is None:
            return goal
        if self.ref is None:
            self.ref = np.asarray(obs.ee_quat, dtype=float)
        ang = quat_angle(self.ref, goal)
        self.ref = goal.copy() if ang <= self.rate else _slerp(self.ref, goal, self.rate / ang)
        return self.ref

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
        return Action(ee_pos=self.at, ee_quat=self._command(obs, ctx.quat), gripper=self.gripper), RUNNING


class StandTurn(Node):
    """Turn the hand (holding position) to the first of `ctx.bb["stand_cands"]` (quaternions,
    best first) it can reach: rate-limited; if the turn stalls (error not down by `progress` rad
    in `window` steps) it accepts the pose when within `loose` rad, else moves on to the next
    candidate. Sets `ctx.quat` to the pose it ends on."""

    def __init__(self, gripper: float, rate: float, tol: float = 0.04, loose: float = 0.12,
                 window: int = 20, progress: float = 0.01, timeout: int = 400, check=None, name: str = "stand"):
        self.gripper, self.rate, self.tol, self.loose = gripper, rate, tol, loose
        self.window, self.progress, self.timeout, self.check, self.name = window, progress, timeout, check, name
        self.reset()

    def reset(self) -> None:
        self.i, self.n, self.at, self.ref, self.hist = 0, 0, None, None, []

    def tick(self, obs, ctx):
        cands = ctx.bb.get("stand_cands") or []
        if self.at is None:
            self.at = obs.ee_pos.copy()
            self.i = int(ctx.bb.get("stand_start", 0))  # after a failed carry: the next pose
        if self.check is not None and (why := self.check(obs, ctx)):
            return None, fail(why)
        self.n += 1
        if self.i >= len(cands) or self.n > self.timeout:
            return None, fail(f"timeout:{self.name}")
        goal = cands[self.i]
        err = quat_angle(obs.ee_quat, goal)
        if err < self.tol:
            ctx.quat = goal
            return None, DONE
        self.hist.append(err)
        if len(self.hist) > self.window and self.hist[-self.window - 1] - err < self.progress:
            if err < self.loose:  # nearly upright: it settles when set down
                ctx.quat = goal
                ctx.bb["stand_loose"] = round(err, 3)
                return None, DONE
            self.i, self.hist, self.ref = self.i + 1, [], None  # stuck: try the next pose
            ctx.bb["stand_tries"] = self.i + 1
            return Action(ee_pos=self.at, ee_quat=obs.ee_quat, gripper=self.gripper), RUNNING
        if self.ref is None:
            self.ref = np.asarray(obs.ee_quat, dtype=float)
        ang = quat_angle(self.ref, goal)
        self.ref = goal.copy() if ang <= self.rate else _slerp(self.ref, goal, self.rate / ang)
        return Action(ee_pos=self.at, ee_quat=self.ref, gripper=self.gripper), RUNNING


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


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
        self.ctx = Context(obj=obj, monitor=HandHoldMonitor(dev_tol=0.03, dev_hard=0.05))
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

        def standing(obs, c):  # it fell over while the hand was on its way: survey again
            return "target_fell" if c.obj_quat is not None and tilt(c.obj_quat) > 0.3 else None

        held = hold_check("grasp_slip")
        return Sequence(
            [
                Ramp(clear_z, self.carry_step, gripper=0.0, timeout=200, name="clear"),
                MoveTo(above_grasp, gripper=0.0, tol=0.006, timeout=250, check=standing, name="approach"),
                Ramp(grasp_target, self.lower_step, eps=0.004, gripper=0.0, timeout=150, check=standing,
                     name="descend"),
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


# ---------------------------------------------------------------- stand a fallen domino up


def _rot(axis: int, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _axis_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about a unit axis (Rodrigues)."""
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def lying_grasp(size: np.ndarray, pos: np.ndarray, quat: np.ndarray, base_xy: np.ndarray, max_width: float,
                end_offset: float, min_z: float) -> tuple[np.ndarray, float]:
    """Top-down grasp of a box that is not standing: fingers close across its most horizontal short
    axis, `end_offset` from the centre toward the +end of its long axis. That end goes up when it
    is stood up, so the box's own z axis points up again: `object_toppled` judges which way a
    domino fell by where its +z end went. Returns (TCP point, hand yaw from DOWN)."""
    R = quat_to_mat(quat)
    long = int(np.argmax(size))
    shorts = [k for k in range(3) if k != long and 2 * size[k] + 0.01 <= max_width] or [k for k in range(3) if k != long]
    k = min(shorts, key=lambda j: abs(R[2, j]))
    c = R[:, k][:2]
    c = c / max(float(np.linalg.norm(c)), 1e-9)
    pt = np.asarray(pos, dtype=float) + R[:, long] * end_offset
    pt[2] = max(pt[2], min_z)
    return pt, wrap_half_pi(math.atan2(c[0], -c[1]))  # hand_quat(yaw) closes along (sin, -cos)


class StandUp:
    """Pick a domino that is not standing (face down, on its side, leaning) near one end, turn the
    hand so it hangs upright with the gripped end on top, and set it down standing at `spot`.

    The upright pose is chosen once it is in the air, from how it actually sits in the fingers:
    among all upright orientations (its own z axis up, yaw on a 10 deg grid) keep those where the
    gripped end is the top, the hand does not point up, and the palm and wrist stay above the
    table when the domino stands on it; prefer the hand pointing down and away from the robot,
    the domino's face across the lane, and the least wrist turn.

    A domino gripped straight from above stands up with the hand horizontal, which puts the wrist
    at the table. So the grasp is tilted about the closing axis (by `tilts`), the smallest tilt
    for which both the grasp and the stand-up pose keep the palm and wrist clear, checked before
    the hand goes down."""

    end_offset = 0.026  # grasp this far from the centre toward one end (8.5 mm pad half-height inside)
    min_grasp_z = 0.006  # aim for the pad centre here; TouchDown stops on contact (fingertips touch at ~9 mm)
    ease_up = 0.001  # after touching down, back off this much before closing
    lift_z = 0.24
    hover = 0.10
    lift_step = 0.006
    carry_step = 0.012
    lower_step = 0.004
    drop = 0.004
    back_off = 0.08  # metres the hand backs out along its own axis after letting go
    settle_steps = 10
    # Hand box in the TCP frame (its z axis is the approach, pointing out of the fingers, so the
    # palm sits at negative z): |x|, |y|, z from, z to. From menagerie's collision geoms, see
    # pick_clutter.py. Plus a point on the hand axis behind the palm (the flange/link 7) and the
    # arm's radius there.
    palm = (0.032, 0.105, -0.13, -0.037)
    wrist = (-0.21, 0.065)
    min_clear = 0.008
    tilts = (0.5, 0.65, 0.8, 1.0)  # radians of grasp tilt about the closing axis, tried in order
    max_up = -0.45  # the standing pose's hand axis must point down at least ~27 deg (reachable, wrist high)
    turn_rate = 0.04  # radians per step (0.8 rad/s) while turning the domino upright
    turn_z = 0.30  # height of the point in front of the robot where the domino is turned upright

    def __init__(self, task: TaskSpec, obj: str, spot: np.ndarray, lane_dir: np.ndarray, variant: int = 0):
        o = task.scene.object(obj)
        self.variant = variant  # which grasp option (the skill passes the attempt number)
        self.obj, self.spot, self.lane = obj, np.asarray(spot, dtype=float), np.asarray(lane_dir, dtype=float)[:2]
        self.size = np.asarray(o.size, dtype=float)
        self.h = float(self.size[np.argmax(self.size)])  # half-length: centre height when upright
        self.max_width = float(task.scene.robot.max_gripper_width)
        self.base = np.asarray(task.scene.robot.base_pose.pos[:2], dtype=float)
        self.end_offset = min(self.end_offset, self.h - 0.009)  # pads (8.5 mm half-height) stay on the domino
        self.ctx = Context(obj=obj, monitor=HandHoldMonitor(dev_tol=0.03, dev_hard=0.05))
        self.runner = Runner(self._build(), self.ctx)

    @property
    def status(self):
        return self.runner.status

    def act(self, obs: Observation) -> Action:
        return self.runner.act(obs)

    def _clear(self, R_h: np.ndarray, tcp_z: float) -> bool:
        """Palm box and wrist above the table for a hand at orientation R_h with its TCP at tcp_z."""
        px, py, z0, z1 = self.palm
        corners = np.array([[sx * px, sy * py, z] for sx in (-1, 1) for sy in (-1, 1) for z in (z0, z1)])
        if tcp_z + float(np.min(corners @ R_h[2, :])) < self.min_clear:
            return False
        return tcp_z + float(R_h[2, 2]) * self.wrist[0] - self.wrist[1] >= self.min_clear

    def plan_grasp(self, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
        """(TCP point, hand orientation matrix, tilt) for grasp option number `variant` (wrapping
        around) among those clear of the table both now and once the domino is stood up, in the
        order: least tilt first, tilt one way then the other, wrist as planned then turned half a
        revolution (same pinch, the arm's joints the other way round). None if none is clear."""
        opts = self.grasp_options(pos, quat)
        return opts[self.variant % len(opts)] if opts else None

    def grasp_options(self, pos: np.ndarray, quat: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
        pt, yaw = lying_grasp(self.size, pos, quat, self.base, self.max_width, self.end_offset, self.min_grasp_z)
        R_d = quat_to_mat(quat)
        out = []
        for t in self.tilts:
            for sgn in ((1.0,) if t == 0 else (1.0, -1.0)):
                for flip in (0.0, math.pi):
                    R0 = quat_to_mat(hand_quat(yaw + flip))
                    R_h = _axis_rot(R0[:, 1], sgn * t) @ R0  # tilt about the closing axis
                    if self._clear(R_h, float(pt[2])) and self.upright_hand(R_h, R_d, pt, pos) is not None:
                        out.append((pt, R_h, sgn * t))
        return out

    def upright_hand(self, R_h: np.ndarray, R_d: np.ndarray, p_e: np.ndarray, p_d: np.ndarray) -> np.ndarray | None:
        """Hand orientation (matrix) that stands the gripped domino up at `spot`, or None."""
        c = self.upright_hands(R_h, R_d, p_e, p_d)
        return c[0] if c else None

    def upright_hands(self, R_h: np.ndarray, R_d: np.ndarray, p_e: np.ndarray, p_d: np.ndarray,
                      k: int = 6, min_sep: float = 0.35) -> list[np.ndarray]:
        """Up to `k` feasible standing hand orientations, best first, at least `min_sep` rad apart."""
        R_hd = R_h.T @ R_d
        p_hd = R_h.T @ (np.asarray(p_d, dtype=float) - p_e)
        radial = self.spot[:2] - self.base
        radial = radial / max(float(np.linalg.norm(radial)), 1e-9)
        scored: list[tuple[float, np.ndarray]] = []
        for flip in (np.eye(3),):  # +z up only (see lying_grasp)
            for i in range(36):
                R_ds = _rot(2, -math.pi + i * math.pi / 18) @ flip
                R_hs = R_ds @ R_hd.T
                tcp_rel = -R_hs @ p_hd  # TCP minus the domino's centre
                if tcp_rel[2] < 0.012:
                    continue  # the gripped end must be the top one
                a = R_hs[:, 2]
                if a[2] > self.max_up:
                    continue  # hand level or pointing up: hard for the arm, wrist low
                if not self._clear(R_hs, self.h + self.drop + tcp_rel[2]):
                    continue  # palm or wrist in the table
                face = R_ds[:2, 0]  # the standing domino's face normal
                turn = math.acos(float(np.clip((np.trace(R_h.T @ R_hs) - 1) / 2, -1, 1)))
                score = -a[2] + 0.6 * float(a[:2] @ radial) + 0.3 * abs(float(face @ self.lane)) - 0.2 * turn
                scored.append((score, R_hs))
        out: list[np.ndarray] = []
        for _, R in sorted(scored, key=lambda x: -x[0]):
            sep = [math.acos(float(np.clip((np.trace(R.T @ Q) - 1) / 2, -1, 1))) for Q in out]
            if all(x >= min_sep for x in sep):
                out.append(R)
            if len(out) >= k:
                break
        return out

    def _build(self):
        st = self.settle_steps

        def grasp_target(obs, c):
            # re-planned only when the domino has moved, so the hand does not chase noise
            key = c.bb.get("grasp_key")
            if key is None or float(np.linalg.norm(c.obj_pos - key[0])) > 0.004 or quat_angle(c.obj_quat, key[1]) > 0.06:
                g = self.plan_grasp(c.obj_pos, c.obj_quat)
                c.bb["grasp_key"] = (c.obj_pos.copy(), c.obj_quat.copy())
                c.bb["grasp"] = g
                if g is not None:
                    c.quat = mat_to_quat(g[1])
            g = c.bb["grasp"]
            if g is None:
                return obs.ee_pos.copy()  # check_grasp fails the node
            return g[0] + (c.obj_pos - c.bb["grasp_key"][0])

        def check_grasp(obs, c):
            return "no_clear_grasp" if c.bb.get("grasp_key") is not None and c.bb.get("grasp") is None else None

        def above_grasp(obs, c):
            # back off along the (possibly tilted) approach, so the hand comes in along it
            pt = grasp_target(obs, c)
            return pt - quat_to_mat(c.quat)[:, 2] * self.hover

        def up_here(obs, c):
            return np.array([obs.ee_pos[0], obs.ee_pos[1], max(obs.ee_pos[2], self.lift_z)])

        def note(obs, c):
            c.bb["rise_xy"] = obs.ee_pos[:2].copy()
            return None

        def note_touch(obs, c):
            c.bb["eased"] = obs.ee_pos + [0.0, 0.0, self.ease_up]
            return None

        def eased(obs, c):
            return c.bb["eased"]

        def risen(obs, c):
            return np.array([*c.bb["rise_xy"], self.lift_z])

        def plan(obs, c):
            o = obs.objects[c.obj]
            cands = self.upright_hands(quat_to_mat(obs.ee_quat), quat_to_mat(o.quat), obs.ee_pos, o.pos)
            if not cands:
                return "no_upright_pose"  # none keeps the palm and wrist clear of the table
            c.bb["stand_cands"] = [mat_to_quat(R) for R in cands]
            c.quat = c.bb["stand_cands"][0]
            c.bb["offsets"] = []
            return None

        def next_pose(c, reason):  # the carry could not get there in this pose: back, turn, retry
            c.bb["stand_start"] = int(c.bb.get("stand_start", 0)) + 1
            c.bb["offsets"] = []

        def turn_point(obs, c):
            # turn in front of the robot, high: the wrist has the most room there
            mid = 0.5 * (np.asarray(c.bb["rise_xy"]) + self.spot[:2])
            return np.array([float(np.clip(mid[0], 0.40, 0.50)), 0.5 * float(mid[1]), self.turn_z])

        def offset(obs, c):
            if c.obj in obs.objects and c.bb.get("offset_step") != c.steps:
                c.bb["offset_step"] = c.steps
                c.bb.setdefault("offsets", []).append(np.asarray(obs.objects[c.obj].pos) - obs.ee_pos)
            return np.mean(c.bb["offsets"][-8:], axis=0)

        def over_spot(obs, c):
            o = offset(obs, c)
            return np.array([self.spot[0] - o[0], self.spot[1] - o[1], self.h + self.drop - o[2] + self.hover])

        def on_spot(obs, c):
            o = offset(obs, c)
            return np.array([self.spot[0] - o[0], self.spot[1] - o[1], self.h + self.drop - o[2]])

        def note_out(obs, c):
            c.bb["out"] = obs.ee_pos - quat_to_mat(obs.ee_quat)[:, 2] * self.back_off
            return None

        held = hold_check("grasp_slip")
        return Sequence(
            [
                Ramp(up_here, self.carry_step, gripper=0.0, timeout=200, name="clear"),
                MoveTo(above_grasp, gripper=0.0, tol=0.006, timeout=250, check=check_grasp, name="approach"),
                TouchDown(grasp_target, self.lower_step, eps=0.003, gripper=0.0, timeout=200, check=check_grasp,
                          name="descend"),
                Do(note_touch, name="note_touch"),
                Ramp(eased, 0.001, eps=0.0015, gripper=0.0, timeout=20, name="ease"),
                grip(settle=st),
                Do(note, name="note"),
                Ramp(risen, self.lift_step, gripper=1.0, check=held, timeout=250, name="lift"),
                Ramp(turn_point, self.carry_step, gripper=1.0, check=held, timeout=300, name="to_turn"),
                Do(plan, name="plan"),
                Retry(
                    Sequence([
                        CarryRamp(turn_point, self.carry_step, gripper=1.0, check=held, timeout=300, name="to_turn"),
                        StandTurn(1.0, rate=self.turn_rate, check=held, name="stand"),
                        CarryRamp(over_spot, self.carry_step, gripper=1.0, check=held, timeout=300, name="carry"),
                    ], name="stand_and_carry"),
                    budgets={"stalled:carry": 5, "timeout:carry": 5},
                    on_retry=next_pose,
                    name="other_pose",
                ),
                CarryRamp(on_spot, self.lower_step, eps=0.003, ok=0.006, gripper=1.0, check=held, timeout=200,
                          name="lower"),
                release(settle=st),
                Do(note_out, name="note_out"),
                Ramp(lambda obs, c: c.bb["out"], self.lower_step, gripper=0.0, timeout=150, name="back_out"),
                Ramp(up_here, self.lift_step, gripper=0.0, timeout=200, name="retreat"),
            ],
            name=f"stand_up_{self.obj}",
        )


def _hand_yaw(obs: Observation) -> float:
    """Yaw of the gripper relative to DOWN (0 = closing along world y)."""
    return heading_x(obs.ee_quat)


# ---------------------------------------------------------------- the skill


class ToppleChain:
    name = "topple_chain"
    version = "0.2.0"

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
    push_travel = 0.045  # less than the spacing plus a thickness, so the fingers never reach the next domino
    tipped = 0.35  # radians: the pushed domino is going, stop pushing
    watch_steps = 80
    quiet_steps = 10
    quiet_tol = 0.01  # radians of tilt change per step that counts as still
    max_nudges = 4
    max_standups = 4  # per domino (each with a different grasp option)
    # Stand a fallen domino back up (StandUp). Taped 15.7 mm dominoes, foam pads, MuJoCo: the
    # all-fallen task went 0/10 -> 4/10 over rounds 4-6 (see HANDOVER). On by default: it only acts
    # on dominoes that are otherwise unrecoverable. False only reports them.
    stand_up = True
    # where a fallen domino is stood up before it is carried to its spot: a grid on both sides of
    # the lane, close to the robot (picks from far out slipped in MuJoCo)
    # (MuJoCo: standing up at (0.40, 0.22) or (0.34, -0.16) failed while carrying or lowering;
    # near the lane centre it works, so the grid hugs the lane)
    standup_xs = (0.40, 0.46, 0.52)
    standup_ys = (0.09, 0.13)
    standup_ys_far = (0.17, 0.21)  # used only when no point of the near rows is clear
    standup_clear = 0.08  # metres from every other domino
    standup_off_spot = 0.07  # a grid point this close to any lane spot is not used
    rehome_step = 0.03  # rad per step when returning the joints to their start after a failed job
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
        self.lane = {c.object: c.region for c in conds}
        # object_toppled(restood=True): a domino must stand again after it fell before its fall
        # counts, so until then a fallen one is down_bad (to be stood up), whichever way it lies
        self.restand = {c.object for c in conds if c.restood}
        self.fell_seen: set[str] = set()
        self.restood_seen: set[str] = set()
        self.home_q: np.ndarray | None = None  # joint posture at the first step
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
        self.standups = {k: 0 for k in self.order}
        self.pushed = False  # once the robot has pushed, fallen dominoes are left to the chain
        self.fell_right: dict[str, bool] = {}
        self.nudges = 0
        self.log: list[str] = []
        self.phase, self.n_phase = "park", 0
        self.samples: list[Observation] = []
        self.state: dict[str, str] = {}
        self.job: ReplaceDomino | StandUp | None = None
        self.push: dict | None = None
        self.last_tilts: dict[str, float] | None = None
        self.quiet = 0
        self.steps = 0
        self.park_quat: np.ndarray | None = None

    # -- dispatch --------------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        if self.home_q is None:
            self.home_q = np.asarray(obs.qpos, dtype=float).copy()
        for k in self.restand:  # the same history the env keeps, from what the robot sees
            t = tilt(obs.objects[k].quat)
            if t >= 0.6:
                self.fell_seen.add(k)
            elif t <= 0.15 and k in self.fell_seen:
                self.restood_seen.add(k)
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
            if t > self.standing_tilt:
                top = up_of(q)[:2]
                ang = math.acos(float(np.clip(top @ self.d[:2] / max(np.linalg.norm(top), 1e-9), -1, 1)))
                lane = self.lane.get(k)
                inside = lane is None or bool(np.all(
                    np.abs(p - np.asarray(self.task.scene.region(lane).center))
                    <= np.asarray(self.task.scene.region(lane).half_extents)))
                self.fell_right[k] = ang <= self.down_dir_tol and inside and (
                    k not in self.restand or k in self.restood_seen)
            if t > self.down_tilt:
                out[k] = "down" if self.fell_right[k] else "down_bad"
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
        for k in self.order:  # leaning for a while now: it rests on something; judge it as fallen
            if st[k] == "leaning":
                st[k] = "down" if self.fell_right.get(k) else "down_bad"
        # Before the robot's own push, a domino that is not standing and did not fall the chain's
        # way is rebuilt: stood up in a clear spot (then repaired onto its spot like any other).
        # Top of a pile first, so nothing is resting on the one being picked.
        bad = [k for k in self.order if st[k] == "down_bad" and self.standups[k] < self.max_standups]
        if bad and not self.pushed and not self.stand_up:
            new = [k for k in bad if self.standups[k] == 0]
            for k in new:
                self.standups[k] = self.max_standups  # report once
                self.log.append(f"{self.steps}:cannot stand {k} up (stand_up is off) [{summary}]")
            bad = []
        if bad and not self.pushed:
            pos = {k: np.mean([s.objects[k].pos for s in self.samples], axis=0) for k in self.order}
            for k in sorted(bad, key=lambda j: -float(pos[j][2])):
                fallen = [j for j in self.order if st[j] in ("down", "down_bad")]
                spot = self.standup_spot(k, pos, fallen)
                if spot is None:
                    self.log.append(f"{self.steps}:no clear spot to stand {k} up")
                    continue
                self.standups[k] += 1
                # each attempt on this domino uses the next grasp option (tilt way, wrist turned)
                self.job = StandUp(self.task, k, spot, self.d, variant=self.standups[k] - 1)
                self._go("standup", f"{k}#{self.standups[k]} at ({spot[0]:.2f},{spot[1]:.2f}) [{summary}]")
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

    def standup_spot(self, k: str, pos: dict[str, np.ndarray], fallen: list[str] | None = None) -> np.ndarray | None:
        """Where to stand domino `k` up. While other dominoes are still lying (`fallen`), on a
        parking grid beside the lane, not on the lane: the lane stays empty, so nothing stands
        between the arm and the next fallen domino; the side away from the ones still lying is
        preferred, then the point nearest to `k`. They are carried into the lane (repairs) once
        all stand. The last fallen domino goes to its own spot if that is clear. Clear:
        `standup_clear` from every other domino, room behind it (toward the robot) for the hand."""
        base = np.asarray(self.task.scene.robot.base_pose.pos[:2], dtype=float)
        others = [pos[j][:2] for j in self.order if j != k]
        lying = [pos[j][:2] for j in (fallen or []) if j != k]

        def clear(pt: np.ndarray) -> bool:
            back = pt - 0.10 * (pt - base) / max(float(np.linalg.norm(pt - base)), 1e-9)
            return not any(float(np.linalg.norm(pt - o)) < self.standup_clear or float(np.linalg.norm(back - o)) < 0.07
                           for o in others)

        own = self.spots[k][:2]
        if not lying and clear(own):
            return np.array([own[0], own[1], 0.0])
        best, best_key = None, None
        spots = [self.spots[j][:2] for j in self.order]
        for x, y0 in [(x, y) for x in self.standup_xs for y in self.standup_ys] + [
                (x, y) for x in self.standup_xs for y in self.standup_ys_far]:
            if best is not None and y0 in self.standup_ys_far:
                break  # a near point was found
            for y in (y0, -y0):
                pt = np.array([x, y])
                if any(float(np.linalg.norm(pt - sp)) < self.standup_off_spot for sp in spots):
                    continue
                if not clear(pt):
                    continue
                away = min((float(np.linalg.norm(pt - q)) for q in lying), default=1.0)
                key = (-round(away / 0.05), float(np.linalg.norm(pt - pos[k][:2])))  # far from the lying, then near
                if best_key is None or key < best_key:
                    best, best_key = np.array([x, y, 0.0]), key
        return best

    def _standup(self, obs: Observation) -> Action:
        a = self.job.act(obs)
        st = self.job.status
        if not st.running:
            self._after_job(obs, f"standup {'failed: ' + st.reason if st.failed else 'done'}", st.failed)
        return a

    def _repair(self, obs: Observation) -> Action:
        a = self.job.act(obs)
        st = self.job.status
        if not st.running:
            self._after_job(obs, f"repair {'failed: ' + st.reason if st.failed else 'done'}", st.failed)
        return a

    def _after_job(self, obs: Observation, why: str, failed: bool) -> None:
        # A failed job can leave the arm in a posture from which the IK cannot reach the next
        # targets (MuJoCo: every later move failed). So after a failure, rehome the joints first.
        if failed and self.home_q is not None:
            self.q_cmd = np.asarray(obs.qpos, dtype=float).copy()
            self._go("rehome", why)
        else:
            self._go("park", why)

    def _rehome(self, obs: Observation) -> Action:
        d = self.home_q - self.q_cmd
        self.q_cmd = self.q_cmd + np.clip(d, -self.rehome_step, self.rehome_step)
        if float(np.max(np.abs(np.asarray(obs.qpos) - self.home_q))) < 0.05 or self.n_phase > 250:
            self._go("park", "rehomed")
        return Action(mode="joint", joint_targets=self.q_cmd.copy(), gripper=0.0)

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
                self.pushed = True  # from here on the chain may be falling: no more rebuilding
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
