"""StackWithRecovery v0.1: build a stack of exact spots, and rebuild it when it gets knocked over.

For the same tasks as `PlaceInOrder` (an `all_of` of tight `object_in_region` spots, bottom row
listed first), but it does not follow a fixed script. Every time its current action ends it looks
at the scene and picks the next one:

1. If an object lies on its side, stand it back up (`UprightCylinder`) on a staging spot: where it
   started if that is free, else a free patch nearby.
2. Else, take the first spot (in the task's order) whose object is not in it. If another object
   stands on or right next to that spot, move that one to its staging spot first. Then place the
   spot's object with PlaceInOrder's placer for its shape.
3. Else everything is in place: hold still, gripper open.

So a knocked-over pyramid is rebuilt from whatever state it is in. A failed action is simply
re-planned; each object gets at most `max_attempts` actions before the skill gives up and holds.

`UprightCylinder`: grip the lying cylinder from above across its body (the fingers close
perpendicular to its axis), lift it, turn the hand to point level and straight out from the robot base (which leaves the held
cylinder vertical), lower it onto the staging spot, open, back away horizontally, then rise. Of
the two such poses (the hand turned 180 degrees about its axis) it tries the one needing less wrist
rotation first, and the other if it cannot set the cylinder down.

Loadable as a `SkillSource` (imports only `skillforge.*` and numpy).
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import (
    Action,
    AllOf,
    ObjectInRegion,
    ObjectSpec,
    Observation,
    Region,
    TaskSpec,
    success_conditions,
)
from skillforge.skills.grasp import DOWN, frame_quat, quat_angle, quat_to_mat
from skillforge.skills.place_in_order import PlaceInOrder, SpotPlace
from skillforge.skills.primitives import (
    DONE,
    RUNNING,
    Context,
    Do,
    Hold,
    MoveTo,
    Node,
    Runner,
    Sequence,
    fail,
)

WORLD_Z = np.array([0.0, 0.0, 1.0])


def upright(quat: np.ndarray) -> bool:
    """The object's own z axis (a cylinder's axis) is near vertical."""
    return abs(float(quat_to_mat(quat)[2, 2])) > 0.9


def footprint_gap(xy: np.ndarray, o: ObjectSpec) -> float:
    """Horizontal distance from `xy` to a fixed box's footprint (0 inside it), using the box's turn
    about z and its local x/y half extents (close enough for a ramp tilted a few degrees)."""
    w, z = float(o.pose.quat[0]), float(o.pose.quat[3])
    yaw = 2.0 * np.arctan2(z, w)
    c, s = np.cos(yaw), np.sin(yaw)
    d = np.asarray(xy, dtype=float) - np.asarray(o.pose.pos[:2], dtype=float)
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    return float(np.linalg.norm(np.maximum(np.abs(local) - np.asarray(o.size[:2]), 0.0)))


def lying(quat: np.ndarray) -> bool:
    return abs(float(quat_to_mat(quat)[2, 2])) < 0.5


class SwingUpright(Node):
    """Hold the EE still and loosen the grip step by step until the held object hangs vertical.

    Gripped across its body near one end, a lying cylinder is held level by the pads' friction. A
    lighter grip lets gravity swing it down about the fingers' closing axis until it hangs
    vertical; too light and it drops. The right force depends on friction and mass (a fixed
    command worked only for some), so this closes the loop on what it sees: loosen slowly, tighten a
    little whenever the object slides down in the fingers, and grip firmly again once it has hung
    vertical for `still_steps` steps.
    """

    def __init__(self, obj: str, start: float = 0.56, step: float = 0.002, floor: float = 0.40,
                 firm: float = 0.6, still_steps: int = 15, timeout: int = 300, name: str = "swing_upright"):
        self.obj, self.start, self.step, self.floor, self.firm = obj, start, step, floor, firm
        self.still_steps, self.timeout, self.name = still_steps, timeout, name
        self.reset()

    def reset(self) -> None:
        self.n, self.g, self.dz0, self.up, self.at = 0, self.start, None, 0, None

    def tick(self, obs, ctx):
        self.n += 1
        if self.at is None:
            self.at = obs.ee_pos.copy()
        s = obs.objects.get(self.obj)
        if s is None:
            return None, fail(f"other:'{self.obj}' not observed")
        if self.up >= self.still_steps:
            return None, DONE
        if self.n > self.timeout:
            return None, fail("timeout:swing_upright")
        if float(np.linalg.norm(np.asarray(s.pos) - obs.ee_pos)) > 0.15:
            return None, fail("grasp_slip:dropped while swinging")
        dz = float(s.pos[2] - obs.ee_pos[2])
        self.dz0 = dz if self.dz0 is None else self.dz0
        if abs(float(quat_to_mat(s.quat)[2, 2])) > 0.97:
            self.g, self.up = self.firm, self.up + 1
        elif self.dz0 - dz > 0.004:  # sliding down in the fingers: a little tighter
            self.g, self.dz0, self.up = min(self.firm, self.g + 0.01), dz, 0
        else:
            self.g, self.up = max(self.floor, self.g - self.step), 0
        return Action(ee_pos=self.at, ee_quat=ctx.quat, gripper=self.g), RUNNING


# Franka hand, measured from the sim's collision geometry in the TCP frame (pointing down):
HAND_HALF_WIDTH = 0.104  # metres: the hand body's half width along the fingers' opening direction
HAND_HALF_DEPTH = 0.032  # metres: its half depth across it
HAND_BOTTOM = 0.037  # metres from the TCP up to the hand body's underside
FINGER_REACH = 0.066  # metres from the TCP to a fully open finger's outer face
FINGER_HALF_DEPTH = 0.011  # metres: a finger's half depth across the opening direction


def _seg_gap(q: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> float:
    """Horizontal distance from `q` to a body of radius `r` around the segment a-b."""
    ab = b - a
    t = 0.0 if float(ab @ ab) < 1e-12 else float(np.clip((q - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(q - (a + t * ab))) - r


def pick_clearance(g: np.ndarray, closing: np.ndarray, axis: np.ndarray, radius: float, obs: Observation,
                   others: dict) -> float:
    """The smallest of `pick_gaps` (inf if there is nothing else)."""
    return min(pick_gaps(g, closing, axis, radius, obs, others).values(), default=np.inf)


def pick_gaps(g: np.ndarray, closing: np.ndarray, axis: np.ndarray, radius: float, obs: Observation,
              others: dict) -> dict[str, float]:
    """Per other object, the smallest horizontal gap (metres; negative = overlap) to the open
    hand about to grip a lying cylinder at TCP `g`: the two fingers beside the cylinder, and the
    hand body above them (only objects taller than its underside). `others` maps names to specs."""
    hand_z = float(g[2]) + HAND_BOTTOM
    fingers = [g[:2] + sgn * closing[:2] * (radius + f * (FINGER_REACH - radius)) + v * axis[:2]
               for sgn in (1, -1) for f in np.linspace(0.3, 1, 4) for v in (-FINGER_HALF_DEPTH, FINGER_HALF_DEPTH)]
    body = [g[:2] + u * closing[:2] + v * axis[:2]
            for u in np.linspace(-HAND_HALF_WIDTH, HAND_HALF_WIDTH, 9) for v in (-HAND_HALF_DEPTH, 0.0, HAND_HALF_DEPTH)]
    gaps: dict[str, float] = {}
    for n, o in others.items():
        st = obs.objects.get(n)
        if st is None:
            continue
        p = np.asarray(st.pos, dtype=float)
        r = float(o.size[0]) if o.shape == "cylinder" else float(np.linalg.norm(o.size[:2]))
        h = float(o.size[1]) if o.shape == "cylinder" else float(o.size[2])
        if o.shape == "cylinder" and lying(st.quat):
            ax = quat_to_mat(st.quat)[:, 2][:2]
            ax = ax / max(float(np.linalg.norm(ax)), 1e-9)
            seg, top = (p[:2] - h * ax, p[:2] + h * ax), p[2] + r
        else:
            seg, top = (p[:2], p[:2]), p[2] + h
        pts = fingers + (body if top > hand_z - 0.005 else [])
        gaps[n] = min(_seg_gap(q, *seg, r) for q in pts)
    return gaps


class WaitStill(Node):
    """Hold the EE until `obj` has stopped moving (it may still be rolling): DONE when the mean of
    its last `window` observed positions moved less than `tol` since the previous window, or after
    `timeout` steps either way. Averaging rides out observation noise."""

    def __init__(self, obj: str, window: int = 5, tol: float = 0.0015, timeout: int = 80, name: str = "wait_still"):
        self.obj, self.window, self.tol, self.timeout, self.name = obj, window, tol, timeout, name
        self.reset()

    def reset(self) -> None:
        self.seen: list[np.ndarray] = []
        self.at: np.ndarray | None = None

    def tick(self, obs, ctx):
        if self.at is None:
            self.at = obs.ee_pos.copy()
        s = obs.objects.get(self.obj)
        if s is not None:
            self.seen.append(np.asarray(s.pos, dtype=float))
        w = self.window
        if len(self.seen) >= 2 * w:
            moved = float(np.linalg.norm(np.mean(self.seen[-w:], axis=0) - np.mean(self.seen[-2 * w : -w], axis=0)))
            if moved < self.tol:
                return None, DONE
        if len(self.seen) >= self.timeout:
            return None, DONE
        return Action(ee_pos=self.at, ee_quat=ctx.quat, gripper=0.0), RUNNING


class UprightCylinder:
    """A one-object skill: stand a lying cylinder up, letting gravity turn it.

    Grip it from above across its body near one end (an end the open hand fits at; on a retry the
    other one), lift it slowly to just 10 cm, then loosen the grip under feedback until it swings
    down to hang vertical (`SwingUpright`), its bottom ~2.6 cm above the table: if it slides in
    the fingers, the table catches it standing. Grip firmly, lower it until it rests, open, rise.
    By default it is set down where it was picked (`stage_xy` None). The hand points down throughout.

    What did not work (sim, 2 in x 3.3 in cylinder): turning the wrist with the cylinder in hand
    (the level hand pose is out of the arm's reach above ~15 cm, and low down the cylinder turns
    in the fingers), and gripping it end-on with the hand level (the hand body would go into the
    table at the cylinder's axis height).
    """

    name = "upright_cylinder"
    version = "0.1.0"

    lift_z = 0.10  # metres: TCP height while it swings (vertical, its bottom hangs ~2.6 cm above the table)
    # metres from the fingertip centre to the cylinder's end: the 17 mm pads must sit fully on the
    # side (at 4 mm, half a pad hung past the end, caught its corner and squirted the cylinder out)
    end_margin = 0.010
    # metres the TCP sits above the cylinder's axis when it grips. 0: at +6 mm (to lift the hand
    # body over standing neighbours) the swing dropped the cylinder in 23 of 24 grid cases at x 0.64.
    grip_lift = 0.0
    settle_steps = 10
    move_step = 0.006  # metres the set-down leads the hand by (it slows to a stop on contact)
    lift_step = 0.0015  # metres per step for the first lift, held level by the end (0.03 m/s)
    carry_step = 0.003  # metres per step carrying it hanging to the staging spot (0.06 m/s)
    # gripper command carrying it vertical (it no longer has to turn in the fingers): at 0.6 it
    # slid out on the carry
    carry_grip = 1.0
    # metres per step, per step: the carry speeds up and slows down at this rate (starting at full
    # speed, the jolt shook it out of the fingers); the rate PlaceInOrder carries upright cylinders at
    carry_accel = 0.00012

    def __init__(self, obj: str, half_len: float, radius: float, stage_xy: np.ndarray | None, walls: list = (),
                 others: dict | None = None, retry: int = 0):
        self.obj, self.half_len, self.radius = obj, half_len, radius
        # retry k > 0 (the swing dropped it before): grip the other end (the far one from the base
        # on odd retries), and loosen more gently from a higher floor, so a retry is not a repeat
        self.retry = retry
        # None: set it down where it was picked (carrying a cylinder that has slid to hang by its
        # very end dropped it, at friction 0.55x and mass 1.5x, even with a gentle ramp)
        self.stage_xy = None if stage_xy is None else np.asarray(stage_xy, dtype=float)
        self.walls = list(walls)
        self.others = others or {}  # name -> ObjectSpec of every other movable object
        self.clearance = np.inf  # of the chosen pick, set by the plan

    def reset(self, task: TaskSpec) -> None:
        self.ctx = Context(obj=self.obj)
        self.ctx.quat = DOWN.copy()
        self.runner = Runner(self._build(), self.ctx)

    def act(self, obs: Observation) -> Action:
        return self.runner.act(obs)

    def _build(self):
        bb = self.ctx.bb
        d_end = self.half_len - self.end_margin  # grip point's distance from the centre, along the axis

        def plan(obs, c):
            if c.obj_pos is None or c.obj_quat is None:
                return f"other:'{self.obj}' not observed"
            if not lying(c.obj_quat):
                return "other:not lying"
            axis = quat_to_mat(c.obj_quat)[:, 2]
            axis = axis - axis[2] * WORLD_Z
            axis /= np.linalg.norm(axis)
            closing = np.cross(WORLD_Z, axis)
            closing = min((closing, -closing), key=lambda cl: quat_angle(frame_quat(-WORLD_Z, cl), obs.ee_quat))
            c.quat = frame_quat(-WORLD_Z, closing)
            p = np.asarray(c.obj_pos, dtype=float)

            def clear(e):  # gap between the open hand at this end and every other object
                tcp = np.array([e[0], e[1], self.radius + self.grip_lift])
                return pick_clearance(tcp, closing, axis, self.radius, obs, self.others)

            def score(e):  # an end the hand fits at; then not against the border; then nearer the base
                near_wall = min((footprint_gap(e[:2], w) for w in self.walls), default=1.0) < 0.04
                far = -1.0 if self.retry % 2 else 1.0  # odd retry: prefer the end away from the base
                return (min(clear(e), 0.01) < 0.01, near_wall, -min(clear(e), 0.01), far * float(np.linalg.norm(e[:2])))

            g = min((p + d_end * axis, p - d_end * axis), key=score)
            self.clearance = clear(g)
            bb["pick"] = np.array([g[0], g[1], self.radius + self.grip_lift])  # TCP just above the axis
            if self.stage_xy is None:
                self.stage_xy = bb["pick"][:2].copy()
            return None

        def ramp(goal, step=self.move_step, key=None, accel=0.0):
            # key None: lead the hand's actual position by `step` (it slows when the hand lags, so it
            # stops gently on contact). With a key: a reference that advances by `step` from where it
            # was, so the hand really moves at that speed (leading a loaded hand by 3 mm, it climbed
            # only ~0.65 mm per step and the lift alone took ~330 steps).
            def f(obs, c):
                t = goal(obs, c)
                ref = obs.ee_pos if key is None else bb.get(key, obs.ee_pos)
                dlt = t - ref
                n = float(np.linalg.norm(dlt))
                st = step
                if key is not None and accel:  # speed up and slow down gently (a hanging load swings)
                    v = min(bb.get(key + "_v", 0.0) + accel, step, float(np.sqrt(2 * accel * n)))
                    bb[key + "_v"] = st = max(v, 1e-5)
                nxt = t if n <= st else ref + dlt * (st / n)
                if key is not None:
                    bb[key] = nxt
                return nxt

            return f

        above_pick = lambda obs, c: bb["pick"] + [0.0, 0.0, 0.10]
        pick = lambda obs, c: bb["pick"]
        up = lambda obs, c: np.array([*bb["pick"][:2], self.lift_z])
        over_stage = lambda obs, c: np.array([*self.stage_xy, self.lift_z])

        def on_stage(obs, c):  # land the CYLINDER on the spot, wherever it sits in the grip
            s = obs.objects.get(self.obj)
            off = np.zeros(3) if s is None else np.asarray(s.pos) - obs.ee_pos
            return np.array([self.stage_xy[0] - off[0], self.stage_xy[1] - off[1], self.half_len + 0.003 - off[2]])

        risen = lambda obs, c: np.array([*self.stage_xy, self.lift_z])

        def resting(obs, c):
            s = obs.objects.get(self.obj)
            down = s is not None and float(s.pos[2]) < self.half_len + 0.008
            f = obs.contact_force
            return down or (f is not None and float(np.linalg.norm(f)) > 5.0)

        def standing(obs, c):
            s = obs.objects.get(self.obj)
            return None if s is not None and upright(s.quat) else "other:not upright after the swing"

        def in_hand(obs, c):  # hanging vertical, its centre ~ half a length below the fingers
            st = obs.objects.get(self.obj)
            if st is None or float(np.linalg.norm(np.asarray(st.pos) - obs.ee_pos)) > self.half_len + 0.04:
                return "grasp_slip:dropped while carrying it upright"
            return None

        def let_go(obs, c):
            c.monitor.disarm()
            return None

        lifted = lambda o, c: o.ee_pos[2] > self.lift_z - 0.005
        over = lambda o, c: float(np.linalg.norm(o.ee_pos[:2] - self.stage_xy)) < 0.004
        # The hold monitor stays unarmed: the cylinder moves relative to the hand by design.
        return Sequence(
            [
                WaitStill(self.obj),  # a cylinder just knocked, or back off the border, may still be rolling
                Do(plan, name="plan_upright"),
                MoveTo(above_pick, gripper=0.0, tol=0.006, timeout=200, name="above_pick"),
                MoveTo(pick, gripper=0.0, tol=0.004, timeout=120, name="down_to_pick"),
                Hold(1.0, steps=self.settle_steps, name="grip"),
                MoveTo(ramp(up, self.lift_step, "ref_lift"), gripper=1.0, tol=1e-9, until=lifted, timeout=400,
                       name="lift"),
                SwingUpright(self.obj, step=0.002 / (1 + self.retry), floor=0.40 + 0.03 * min(self.retry, 2)),
                Do(standing, name="check_vertical"),
                MoveTo(ramp(over_stage, self.carry_step, "ref_carry", self.carry_accel), gripper=self.carry_grip, tol=1e-9, until=over,
                       check=in_hand, timeout=400,
                       name="over_stage"),
                MoveTo(ramp(on_stage), gripper=self.carry_grip, tol=1e-9, until=resting, check=in_hand, timeout=200,
                       name="set_down"),
                Hold(0.0, steps=self.settle_steps, on_done=let_go, name="release"),
                MoveTo(risen, gripper=0.0, tol=0.01, timeout=150, name="rise"),
            ],
            name="upright",
        )


class StackWithRecovery:
    name = "stack_with_recovery"
    version = "0.1.0"

    max_attempts = 6  # actions per object (its first placement included) before giving up on it
    # Checking a spot is filled: positions are averaged over the last `avg_window` observations (noise
    # std up to 2 mm -> ~0.6 mm), then allowed `spot_margin` of slack. With single readings and 6 mm
    # of slack, cylinders 0.5-2.7 mm outside their spots passed as filled and the episode failed.
    avg_window = 10
    spot_margin = 0.002
    # metres inside its spot a cylinder must sit while nothing stands on it yet; else it is re-placed
    # (one 0.5 mm outside was built on, and then could not be fixed)
    strict_margin = 0.0015
    clear_radius = 0.06  # metres: a standing object this close to an empty spot is in the way
    pick_margin = 0.005  # metres the open hand must clear other objects by to stand a cylinder up

    def reset(self, task: TaskSpec) -> None:
        conds = success_conditions(task.success)
        if not isinstance(task.success, AllOf) or not all(isinstance(c, ObjectInRegion) for c in conds):
            raise ValueError("StackWithRecovery needs an all_of task of object_in_region conditions")
        self.task, self.conds = task, conds
        self.builder = PlaceInOrder()
        self.builder.reset(task)  # for the closing axis and the per-shape placer settings
        self.across = self.builder.skills[0].closing_axis
        self.objs = {o.name: o for o in task.scene.objects if not o.fixed}
        self.walls = [o for o in task.scene.objects if o.fixed and o.shape == "box"]
        self.cur = None
        self.cur_cond: ObjectInRegion | None = None
        self.attempts: dict[str, int] = {}
        self.swing_drops: dict[str, int] = {}  # per cylinder: stand-ups that dropped it
        self.seen: dict[str, list[np.ndarray]] = {}  # per object: its last observed positions
        self.log: list[str] = []
        self.steps = 0

    # -- what the scene looks like -----------------------------------------------

    def _in_spot(self, obs: Observation, c: ObjectInRegion, margin: float | None = None) -> bool:
        s = obs.objects.get(c.object)
        if s is None or not upright(s.quat):
            return False
        r = self.task.scene.region(c.region)
        h = self.seen.get(c.object)
        pos = np.mean(h, axis=0) if h else np.asarray(s.pos)
        d = np.abs(pos - np.asarray(r.center))
        return bool(np.all(d <= np.asarray(r.half_extents) + (self.spot_margin if margin is None else margin)))

    def _settled(self, obs: Observation, c: ObjectInRegion) -> bool:
        """Filled for building on: well inside its spot while it is still free to be re-placed (once
        something stands on it, it can no longer be fixed, so then the normal slack applies)."""
        if self._carrying(obs, c.object):
            return self._in_spot(obs, c)
        return self._in_spot(obs, c, -self.strict_margin)

    def _free_xy(self, obs: Observation, want: np.ndarray, me: str) -> np.ndarray:
        """`want` if no other object stands within reach of it, else the nearest free patch, else None."""
        rad = max(float(max(o.size[0], o.size[1])) for o in self.objs.values())
        spots = [np.asarray(self.task.scene.region(c.region).center[:2]) for c in self.conds]
        taken = [np.asarray(s.pos[:2]) for n, s in obs.objects.items() if n != me and n in self.objs] + spots
        # a lying cylinder still has to be gripped: keep its hand patch (+-10 cm across it) clear
        lying_xy = [np.asarray(s.pos[:2]) for n, s in obs.objects.items()
                    if n != me and n in self.objs and self.objs[n].shape == "cylinder" and lying(s.quat)]

        def ok(xy):
            r = float(np.linalg.norm(xy))
            return (
                0.33 <= r <= 0.66
                and all(float(np.linalg.norm(xy - p)) > 2 * rad + 0.02 for p in taken)
                and all(float(np.linalg.norm(xy - p)) > HAND_HALF_WIDTH + 2 * rad for p in lying_xy)
                and all(footprint_gap(xy, w) > rad + 0.03 for w in self.walls)  # clear of the border
            )

        if ok(want):
            return want
        for ring in (0.06, 0.12, 0.18, 0.24, 0.30, 0.36):
            for k in range(16):
                a = 2 * np.pi * k / 16
                xy = want + ring * np.array([np.cos(a), np.sin(a)])
                if ok(xy):
                    return xy
        return None

    def _stage_xy(self, obs: Observation, name: str) -> np.ndarray | None:
        """A free spot to stand `name` on (its start spot if free), or None if there is none."""
        return self._free_xy(obs, np.asarray(self.objs[name].pose.pos[:2], dtype=float), name)

    # -- choosing the next action ------------------------------------------------

    def _next(self, obs: Observation):
        lying_now = [n for n, o in self.objs.items() if o.shape == "cylinder" and n in obs.objects
                     and lying(obs.objects[n].quat) and self._budget(n)]
        blocked: dict[str, list[str]] = {}
        for name in lying_now:  # 1. stand up anything lying down, if the hand fits beside it
            gaps = self._end_gaps(obs, name)
            in_way = [n for n, gp in gaps.items() if gp < self.pick_margin]
            if not in_way:
                return self._upright(obs, name)
            blocked[name] = in_way
        for name, in_way in blocked.items():  # 1b. no room: move a standing neighbour out of the way
            for n in in_way:
                if n in lying_now or not upright(obs.objects[n].quat) or self._carrying(obs, n) or not self._budget(n):
                    continue
                xy = self._stage_xy(obs, n)
                if xy is None or float(np.linalg.norm(xy - np.asarray(obs.objects[n].pos[:2]))) < 0.03:
                    continue  # nowhere better to put it
                self.log.append(f"{self.steps}:clear_for:{name}:{n}")
                return self._mover(n, xy)
        if lying_now:  # nothing to move: try anyway
            return self._upright(obs, lying_now[0])
        for c in self.conds:  # 2. the first spot that is not filled and can be filled now
            if self._settled(obs, c):
                continue
            me = obs.objects.get(c.object)
            if me is None or not upright(me.quat):
                continue  # lying (step 1 stands it up, or it is out of attempts): not placeable as is
            if not all(self._settled(obs, cc) for cc in self._supports(c)):
                continue  # what it stands on is missing or off: build elsewhere first
            if not self._budget(c.object):
                continue  # out of attempts for this one: keep building the rest
            if self._carrying(obs, c.object):
                continue  # slightly off but holding others up: picking it would bring them down
            centre = np.asarray(self.task.scene.region(c.region).center)
            for n, s in obs.objects.items():  # something else standing in the way?
                if n == c.object or n not in self.objs or not upright(s.quat):
                    continue
                if any(cc.object == n and self._in_spot(obs, cc) for cc in self.conds):
                    continue  # it is in its own spot: leave it
                near = float(np.linalg.norm(np.asarray(s.pos[:2]) - centre[:2])) < self.clear_radius
                xy = self._stage_xy(obs, n) if near and abs(float(s.pos[2]) - float(centre[2])) < 0.03 else None
                if xy is not None and self._budget(n):
                    self.log.append(f"{self.steps}:clear:{n}")
                    return self._mover(n, xy)
            self.log.append(f"{self.steps}:place:{c.object}")
            sk = self.builder._placer(self.objs[c.object].shape)
            sk.closing_axis = self.across
            return sk, self.task.model_copy(update={"success": c})
        return None  # 3. done

    def _upright(self, obs: Observation, name: str):
        o = self.objs[name]
        others = {n: oo for n, oo in self.objs.items() if n != name}
        # stood up where it lies (the hand fitted there for the pick); a later step moves it if needed
        sk = UprightCylinder(name, float(o.size[1]), float(o.size[0]), None, self.walls, others,
                             retry=self.swing_drops.get(name, 0))
        self.log.append(f"{self.steps}:upright:{name}")
        return sk, self.task

    def _end_gaps(self, obs: Observation, name: str) -> dict[str, float]:
        """Gaps between the open hand and every other object, at the better end of lying `name`."""
        o, s = self.objs[name], obs.objects[name]
        half, r = float(o.size[1]), float(o.size[0])
        axis = quat_to_mat(s.quat)[:, 2]
        axis = axis - axis[2] * WORLD_Z
        axis /= np.linalg.norm(axis)
        closing = np.cross(WORLD_Z, axis)
        others = {n: oo for n, oo in self.objs.items() if n != name}
        best: dict[str, float] = {}
        for sgn in (1.0, -1.0):
            e = np.asarray(s.pos, dtype=float) + sgn * (half - UprightCylinder.end_margin) * axis
            gaps = pick_gaps(np.array([e[0], e[1], r]), closing, axis, r, obs, others)
            if not best or min(gaps.values(), default=np.inf) > min(best.values(), default=np.inf):
                best = gaps
        return best

    def _carrying(self, obs: Observation, name: str) -> bool:
        """Whether something rests on `name` (it cannot be moved without bringing that down)."""
        p, w = np.asarray(obs.objects[name].pos), 2 * float(self.objs[name].size[0])
        return any(float(np.asarray(s.pos)[2]) > float(p[2]) + 0.01 and float(np.linalg.norm(np.asarray(s.pos)[:2] - p[:2])) < w
                   for n, s in obs.objects.items() if n != name and n in self.objs)

    def _mover(self, name: str, xy: np.ndarray):
        """A placer that stands `name` on the table at `xy` (out of the way)."""
        o = self.objs[name]
        h = float(o.size[1] if o.shape == "cylinder" else o.size[2])
        reg = Region(name=f"_stage_{name}", center=(float(xy[0]), float(xy[1]), h), half_extents=(0.01, 0.01, 0.01))
        scene = self.task.scene.model_copy(update={"regions": (*self.task.scene.regions, reg)})
        sub = self.task.model_copy(update={"scene": scene, "success": ObjectInRegion(object=name, region=reg.name)})
        sk = self.builder._placer(o.shape) if o.shape == "cylinder" else SpotPlace()
        sk.closing_axis = self.across
        return sk, sub

    def _budget(self, name: str) -> bool:
        return self.attempts.get(name, 0) < self.max_attempts

    def _supports(self, c: ObjectInRegion) -> list[ObjectInRegion]:
        """The spots below `c` that its object stands on."""
        me = np.asarray(self.task.scene.region(c.region).center)
        w = 2 * max(float(o.size[0]) for o in self.objs.values())
        out = []
        for cc in self.conds:
            p = np.asarray(self.task.scene.region(cc.region).center)
            if me[2] - p[2] > 0.01 and float(np.linalg.norm(me[:2] - p[:2])) < w:
                out.append(cc)
        return out

    def _guard(self, obs: Observation) -> None:
        """If the current placement's supports were knocked away, set the object down on the table
        instead of on the gap (where it would fall and roll)."""
        sk, c = self.cur, self.cur_cond
        if c is None or getattr(sk, "redirected", False) or not hasattr(sk, "redirect"):
            return
        if all(self._in_spot(obs, cc) for cc in self._supports(c)):
            return
        o = self.objs[c.object]
        xy = self._stage_xy(obs, c.object)
        if xy is None:
            xy = np.asarray(o.pose.pos[:2], dtype=float)  # its start spot
        sk.redirect(Region(name=f"_stage_{c.object}", center=(float(xy[0]), float(xy[1]), float(o.size[1])),
                           half_extents=(0.01, 0.01, 0.01)))
        sk.redirected = True
        self.log.append(f"{self.steps}:redirect:{c.object}")

    def act(self, obs: Observation) -> Action:
        self.steps += 1
        for n, st in obs.objects.items():
            h = self.seen.setdefault(n, [])
            h.append(np.asarray(st.pos, dtype=float))
            del h[: -self.avg_window]
        if self.cur is not None and not self.cur.runner.status.running:
            st = self.cur.runner.status
            self.log.append(f"{self.steps}:{st.state}:{st.reason}" if st.reason else f"{self.steps}:{st.state}")
            if isinstance(self.cur, UprightCylinder) and st.failed and st.reason.startswith("grasp_slip"):
                self.swing_drops[self.cur.obj] = self.swing_drops.get(self.cur.obj, 0) + 1
            self.cur = None
        if self.cur is not None:
            self._guard(obs)
        if self.cur is None:
            nxt = self._next(obs)
            if nxt is None:
                return Action(ee_pos=obs.ee_pos, ee_quat=obs.ee_quat, gripper=0.0)
            sk, sub = nxt
            sk.reset(sub)
            sc = sub.success
            self.cur_cond = sc if isinstance(sc, ObjectInRegion) and sc in self.conds else None
            obj = sk.obj if isinstance(sk, UprightCylinder) else sub.success.object
            self.attempts[obj] = self.attempts.get(obj, 0) + 1
            self.cur = sk
        return self.cur.act(obs)
