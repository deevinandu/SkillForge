"""PickPlace v0.2.1: scripted closed-loop pick-and-place built from `primitives`, with recovery.

Over v0.2.0:
- the hold monitor tests a low-pass filtered object-to-gripper error, so observation noise no
  longer triggers false "slips" whose retry opened the gripper over the table;
- grip and release wait until the measured finger width has actually changed and settled;
- the object position used to plan a grasp is the median of several stationary observations
  (not a lightly filtered, often snapped, noisy reading), and the grasp point is frozen once
  planned instead of chasing the noisy estimate during the descent;
- when the object is resting on the table its centre height comes from the known resting height,
  so observation noise in z can no longer drive the fingers into the table;
- the object-in-gripper offset used for placing is the median over a stationary hold above the
  bin (the v0.2.0 offset was a single noisy sample taken at grasp time), so the object is placed
  near the bin centre instead of near its edge;
- the post-release placement check uses the median of several observations, so noise no longer
  triggers needless re-picks out of the bin (which is where the finger-table collisions happened).
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import Action, ObjectInRegion, ObjectSpec, Observation, TaskSpec
from skillforge.skills.grasp import (
    DOWN,
    plan_grasps,
    quat_from_axis_angle,
    quat_mul,
    quat_to_mat,
    select_grasp,
    yaw_of,
)
from skillforge.skills.primitives import (
    DONE,
    RUNNING,
    Context,
    Do,
    HoldMonitor,
    Node,
    Retry,
    Runner,
    Sequence,
    approach,
    descend_until,
    fail,
    grip,
    hold_check,
    lift,
    place,
    release,
    retreat,
    transport,
)


def _resting_half_height(o: ObjectSpec) -> float:
    if o.shape == "box":
        return float(o.size[2])
    if o.shape == "cylinder":
        return float(o.size[1])
    if o.shape == "sphere":
        return float(o.size[0])
    return 0.0


class Observe(Node):
    """Hold the EE still for `steps` steps and store the median of `sample(obs, ctx)` in bb[key]."""

    def __init__(self, key: str, sample, steps: int = 10, gripper: float = 0.0, check=None, name: str = "observe"):
        self.key, self.sample, self.steps, self.gripper, self.check, self.name = key, sample, steps, gripper, check, name
        self.reset()

    def reset(self) -> None:
        self.at = None
        self.samples: list[np.ndarray] = []

    def tick(self, obs, ctx):
        if self.at is None:
            self.at = obs.ee_pos.copy()
        if self.check is not None and (why := self.check(obs, ctx)):
            return None, fail(why)
        s = self.sample(obs, ctx)
        if s is not None:
            self.samples.append(np.asarray(s, dtype=float))
        if len(self.samples) >= self.steps:
            ctx.bb[self.key] = np.median(np.stack(self.samples), axis=0)
            return None, DONE
        if ctx.steps > 10_000:  # never observed: give up rather than hang
            return None, fail("other:not_observed")
        return Action(ee_pos=self.at, ee_quat=ctx.quat, gripper=self.gripper), RUNNING


class FilteredHoldMonitor(HoldMonitor):
    """HoldMonitor whose drift test uses a low-pass filtered offset error instead of single raw
    readings: with obs noise ~1.5 cm per axis a raw 3D error exceeds 3 cm about a quarter of the
    time, so the stock monitor declares false slips (and the retry then opens the gripper)."""

    def __init__(self, dev_hard: float = 0.08, beta: float = 0.3):
        super().__init__(dev_hard=dev_hard)
        self.beta = beta
        self._ema = None

    def arm(self, obs: Observation, obj_pos: np.ndarray) -> None:
        super().arm(obs, obj_pos)
        self._ema = None

    def check(self, obs: Observation, obj: str, obj_pos: np.ndarray | None = None) -> str | None:
        if not self.armed:
            return None
        z = float((obs.objects[obj].pos if obj_pos is None else obj_pos)[2])
        self.rise = max(self.rise, z - self.z0)
        if self.w0 > self.width_eps and obs.gripper_width < self.width_eps:
            return "width_collapsed"
        err = obs.objects[obj].pos - obs.ee_pos - self.offset
        raw = float(np.linalg.norm(err))
        if raw > self.dev_hard:
            return f"object_left_gripper({raw:.3f}m)"
        self._ema = err.copy() if self._ema is None else (1 - self.beta) * self._ema + self.beta * err
        dev = float(np.linalg.norm(self._ema))
        self._bad = self._bad + 1 if dev > self.dev_tol else 0
        if self._bad >= self.sustain:
            return f"object_drifting({dev:.3f}m)"
        return None


class SettledGrip(Node):
    """Close the gripper and hold until the measured width stops changing (at least `min_steps`,
    at most `max_steps`), then run `grip()`'s miss check / monitor arming. A fixed short settle
    lets the arm lift before the fingers have closed when actions are delayed."""

    def __init__(
        self, min_steps: int = 4, max_steps: int = 20, stable: int = 3, eps: float = 0.0005, gripper: float = 1.0
    ):
        self.gripper = gripper
        self.inner = grip(settle=0) if gripper > 0.5 else release(settle=0)
        self.min_steps, self.max_steps, self.stable, self.eps = min_steps, max_steps, stable, eps
        self.name = "grip" if gripper > 0.5 else "release"
        self.reset()

    def reset(self) -> None:
        self.n, self.calm, self.w, self.w0, self.at = 0, 0, None, None, None

    def tick(self, obs, ctx):
        if self.at is None:
            self.at = obs.ee_pos.copy()
        w = float(obs.gripper_width)
        self.calm = self.calm + 1 if self.w is not None and abs(w - self.w) < self.eps else 0
        self.w = w
        if self.w0 is None:
            self.w0 = w
        self.n += 1
        # the fingers have actually started moving (actions may lag several steps)
        moved = w < self.w0 - 0.005 if self.gripper > 0.5 else w > self.w0 + 0.005
        if self.n > self.max_steps or (self.n > self.min_steps and moved and self.calm >= self.stable):
            why = self.inner.on_done(obs, ctx)
            return None, fail(why) if why else DONE
        return Action(ee_pos=self.at, ee_quat=ctx.quat, gripper=self.gripper), RUNNING


class PickPlace:
    name = "pick_place"
    version = "0.2.1"

    hover = 0.10  # metres above grasp/place targets for pre-grasp and transport
    tol = 0.006  # metres: position error that counts as "arrived"
    settle_steps = 4  # steps to hold a gripper command before moving on
    observe_steps = 10  # stationary observations averaged before planning / verifying
    offset_steps = 8  # stationary observations averaged for the in-hand offset
    rest_band = 0.03  # metres: observed z this close to the resting height counts as resting
    miss_retries = 1
    slip_regrasps = 2
    place_retries = 1
    verify_margin = 0.01  # metres: slack on the region check, which uses noisy observations

    def reset(self, task: TaskSpec) -> None:
        if not isinstance(task.success, ObjectInRegion):
            raise ValueError("PickPlace needs an object_in_region task")
        self.task = task
        self.spec = task.scene.object(task.success.object)
        self.region = task.scene.region(task.success.region)
        self.max_width = task.scene.robot.max_gripper_width
        self.rest_z = float(self.spec.pose.pos[2])  # object centre height when resting on the table
        center = np.asarray(self.region.center, dtype=float)
        bottom = center[2] - self.region.half_extents[2]
        self.place_obj = np.array([center[0], center[1], max(center[2], bottom + _resting_half_height(self.spec))])
        self.ctx = Context(obj=self.spec.name, monitor=FilteredHoldMonitor())
        self.ctx.bb["rank"] = 0
        self.runner = Runner(self._build(), self.ctx)

    def act(self, obs: Observation) -> Action:
        return self.runner.act(obs)

    # -- tree ----------------------------------------------------------------

    def _build(self):
        hv, tol, st = self.hover, self.tol, self.settle_steps

        def raw_obj(obs, c):
            o = obs.objects.get(c.obj)
            return None if o is None else o.pos

        def raw_offset(obs, c):
            o = obs.objects.get(c.obj)
            return None if o is None else o.pos - obs.ee_pos

        def safe_z(obs, c):
            z_obj = c.bb.get("rest_z", c.obj_pos[2] if c.obj_pos is not None else obs.ee_pos[2])
            return max(z_obj, self.place_obj[2]) + hv

        def grasp_target(obs, c):
            return c.bb["grasp_world"]

        def offset(c):
            off = np.asarray(c.bb.get("offset", c.monitor.offset), dtype=float).copy()
            off[:2] = np.clip(off[:2], -0.02, 0.02)
            off[2] = float(np.clip(off[2], -0.03, 0.03))
            return off

        def place_target(obs, c):
            return self.place_obj - offset(c)

        def above_place(obs, c):
            return place_target(obs, c) + [0.0, 0.0, hv]

        attempt = Sequence(
            [
                Do(self._disarm, name="disarm"),
                retreat(safe_z, tol=tol, name="clear"),
                Observe("obj_est", raw_obj, steps=self.observe_steps, name="observe_obj"),
                Do(self._plan, name="plan_grasp"),
                approach(grasp_target, hover=hv, tol=tol),
                descend_until(grasp_target, tol=tol),
                SettledGrip(min_steps=st),
                lift(safe_z, tol=tol),
                transport(above_place, tol=0.02, name="transport_rough"),
                Observe(
                    "offset",
                    raw_offset,
                    steps=self.offset_steps,
                    gripper=1.0,
                    check=hold_check("grasp_slip"),
                    name="measure_offset",
                ),
                Do(self._refine_offset, name="refine_offset"),
                transport(above_place, tol=tol),
                place(place_target, tol=tol),
                SettledGrip(min_steps=st, gripper=0.0),
                retreat(safe_z, tol=tol),
                Observe("final_est", raw_obj, steps=self.observe_steps, name="observe_final"),
                Do(self._verify, name="verify"),
            ],
            name="pick_place",
        )
        budgets = {"grasp_miss": self.miss_retries, "grasp_slip": self.slip_regrasps, "wrong_placement": self.place_retries}
        return Retry(attempt, budgets, on_retry=self._on_retry)

    # -- steps ---------------------------------------------------------------

    def _disarm(self, obs: Observation, c: Context) -> str | None:
        c.monitor.disarm()
        c.bb.pop("rest_z", None)  # re-read from wherever the object is now
        c.bb.pop("offset", None)
        return None

    def _refine_offset(self, obs: Observation, c: Context) -> str | None:
        if "offset" in c.bb:
            c.monitor.offset = np.asarray(c.bb["offset"], dtype=float).copy()
            if isinstance(c.monitor, FilteredHoldMonitor):
                c.monitor._ema = None
        return None

    def _plan(self, obs: Observation, c: Context) -> str | None:
        est = c.bb.get("obj_est")
        if est is None:
            return f"other:object '{c.obj}' not observed"
        est = np.asarray(est, dtype=float).copy()
        if abs(est[2] - self.rest_z) < self.rest_band:
            est[2] = self.rest_z  # resting on the table: trust geometry over noisy z
        c.bb["rest_z"] = float(est[2])
        q_obj = c.obj_quat if c.obj_quat is not None else np.array([1.0, 0, 0, 0])
        grasps = plan_grasps(self.spec, self.max_width, obj_quat=q_obj)
        if grasps:
            _, quat, g = select_grasp(grasps, est, q_obj, ee_quat=obs.ee_quat, rank=c.bb["rank"])
            gw = est + quat_to_mat(q_obj) @ g.pos
            gw[2] = max(gw[2], self.rest_z - 0.002)  # never aim the TCP into the table
            c.bb["grasp_world"] = gw
            c.quat = quat
            return None
        if self.spec.shape != "mesh":
            return f"other:no_feasible_grasp(max_width={self.max_width})"
        c.bb["grasp_world"] = est
        c.quat = quat_mul(quat_from_axis_angle([0, 0, 1], yaw_of(q_obj)), DOWN)
        return None

    def _verify(self, obs: Observation, c: Context) -> str | None:
        est = c.bb.get("final_est")
        if est is None:
            return None
        d = np.abs(np.asarray(est) - np.asarray(self.region.center))
        if np.all(d <= np.asarray(self.region.half_extents) + self.verify_margin):
            return None
        return f"wrong_placement:observed {np.round(est, 3).tolist()} outside {self.region.name}"

    def _on_retry(self, c: Context, reason: str) -> None:
        if reason.startswith("grasp_miss"):
            c.bb["rank"] += 1  # try the next-ranked grasp
