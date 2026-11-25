"""PickPlace v0.3: scripted closed-loop pick-and-place built from `primitives`, with recovery.

Over v0.1 (still available as `skillforge.skills.pick_place_v01.PickPlaceV01`):
- grasp orientation from the grasp planner (`grasp.py`) and the observed object orientation;
- filtered object observations instead of chasing raw noisy positions;
- a missed grasp (object not following the gripper on lift, or fingers closed on nothing) is
  retried with the next-ranked grasp (`miss_retries`, default once);
- a slip while carrying (object leaves the gripper, or the measured width collapses) triggers a
  re-grasp from wherever the object ended up (`slip_regrasps`, default twice);
- after release, if the object is observed outside the goal region it is picked again once.

v0.3 (2026-11-22) is the agent-written hard-cube skill 0.2.2
(`results/2026-11-04-claude-code/hard/pick_place_0.2.2.py`, Claude Code, 2026-11-04) promoted to
the library, plus v0.2's optional gripper pre-shape (`preshape_margin`, used by PickFromClutter).
Over v0.2 it handles observation noise up to 1.5 cm: a windowed hold monitor, grasp height
snapped to the nominal resting height, the held offset averaged over the carry, placement
judged on the mean of several observations, and a settle time longer than the action latency.

Loadable as a `SkillSource`: one file, imports only `skillforge.*` and numpy (absolute imports).
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
    Context,
    Do,
    Hold,
    HoldMonitor,
    Retry,
    Runner,
    Sequence,
    approach,
    descend_until,
    grip,
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


class SmoothedHoldMonitor(HoldMonitor):
    """HoldMonitor that judges the object-to-EE drift on a short window mean of raw observations.

    The stock monitor tests a single raw sample against 3 cm; with observation noise up to 1.5 cm
    per axis (about 2.6 cm in norm) that fires on a perfectly good hold, and the slip recovery then
    opens the gripper mid-carry. A real drop moves the object 10 cm or more, which the window
    mean still catches within a few steps.
    """

    def __init__(self, window: int = 8, dev_tol: float = 0.04, dev_hard: float = 0.07):
        super().__init__(dev_tol=dev_tol, dev_hard=dev_hard)
        self.window = window
        self._devs: list[np.ndarray] = []

    def arm(self, obs: Observation, obj_pos: np.ndarray) -> None:
        super().arm(obs, obj_pos)
        self._devs = []

    def check(self, obs: Observation, obj: str, obj_pos: np.ndarray | None = None) -> str | None:
        if not self.armed:
            return None
        z = float((obs.objects[obj].pos if obj_pos is None else obj_pos)[2])
        self.rise = max(self.rise, z - self.z0)
        if self.w0 > self.width_eps and obs.gripper_width < self.width_eps:
            return "width_collapsed"
        devs = self._devs
        devs.append(np.asarray(obs.objects[obj].pos, dtype=float) - obs.ee_pos - self.offset)
        del devs[: -self.window]
        if len(devs) < 3:
            return None
        dev = float(np.linalg.norm(np.mean(devs, axis=0)))
        if dev > self.dev_hard:
            return f"object_left_gripper({dev:.3f}m)"
        self._bad = self._bad + 1 if dev > self.dev_tol else 0
        if self._bad >= self.sustain:
            return f"object_drifting({dev:.3f}m)"
        return None


class PickPlace:
    name = "pick_place"
    version = "0.3.0"

    hover = 0.10  # metres above grasp/place targets for pre-grasp and transport
    tol = 0.006  # metres: position error that counts as "arrived"
    # Steps to hold a gripper command before moving on. Must exceed the action latency (up to 6
    # steps under randomization) or the arm starts moving before the fingers have closed/opened.
    settle_steps = 10
    miss_retries = 1
    slip_regrasps = 2
    place_retries = 1
    verify_margin = 0.01  # metres: slack on the region check, which uses noisy observations
    preshape_margin: float | None = None  # metres: open only to the widest grasp + this (None = fully open)
    # World axis the fingers should close along (None = any). Placing next to neighbours: close
    # across the row so the fingers stay clear of them. Falls back to any grasp if none fits.
    closing_axis: tuple[float, float, float] | None = None
    offset_window = 40  # most recent carry samples averaged for the object-to-EE offset
    verify_samples = 12 # raw observations averaged before the region check (noise up to 1.5 cm)

    def reset(self, task: TaskSpec) -> None:
        if not isinstance(task.success, ObjectInRegion):
            raise ValueError("PickPlace needs an object_in_region task")
        self.task = task
        self.spec = task.scene.object(task.success.object)
        self.region = task.scene.region(task.success.region)
        self.max_width = task.scene.robot.max_gripper_width
        self.rest_height = float(self.spec.pose.pos[2])  # nominal object centre height at rest
        center = np.asarray(self.region.center, dtype=float)
        bottom = center[2] - self.region.half_extents[2]
        self.place_obj = np.array([center[0], center[1], max(center[2], bottom + _resting_half_height(self.spec))])
        self.ctx = Context(obj=self.spec.name, monitor=SmoothedHoldMonitor())
        self.ctx.bb["rank"] = 0
        self.runner = Runner(self._build(), self.ctx)

    def act(self, obs: Observation) -> Action:
        return self.runner.act(obs)

    # -- tree ----------------------------------------------------------------

    def _build(self):
        hv, tol, st = self.hover, self.tol, self.settle_steps

        def safe_z(obs, c):
            # Object's resting height as seen at planning time (it rises with the gripper later).
            z_obj = c.bb.get("rest_z", c.obj_pos[2] if c.obj_pos is not None else obs.ee_pos[2])
            return max(z_obj, self.place_obj[2]) + hv

        def grasp_target(obs, c):
            g, q_obj = c.bb["grasp"], c.bb["grasp_obj_quat"]
            p = c.obj_pos.copy()
            # Observed z is noisy (~1.5 cm). An object resting on the support surface sits at the
            # task's nominal resting height; use it, and never aim below it (fingers hit the table).
            if abs(p[2] - self.rest_height) < 0.025:
                p[2] = self.rest_height
            p[2] = max(p[2], self.rest_height)
            return p + quat_to_mat(q_obj) @ g

        def held_offset(obs, c):
            # Object-to-EE offset averaged over the carry, instead of one noisy sample at grip.
            if c.obj in obs.objects and c.monitor.armed:
                c.bb.setdefault("offsets", []).append(np.asarray(obs.objects[c.obj].pos) - obs.ee_pos)
            offs = c.bb.get("offsets") or []
            if len(offs) < 3:
                return c.monitor.offset
            return np.mean(offs[-self.offset_window :], axis=0)

        def place_target(obs, c):
            return self.place_obj - held_offset(obs, c)

        def above_place(obs, c):
            return place_target(obs, c) + [0.0, 0.0, hv]

        attempt = Sequence(
            [
                Do(self._disarm, name="disarm"),
                retreat(safe_z, tol=tol, name="clear"),
                Do(self._plan, name="plan_grasp"),
                approach(grasp_target, hover=hv, tol=tol, gripper=self._open_cmd()),
                descend_until(grasp_target, tol=tol, gripper=self._open_cmd()),
                grip(settle=st),
                lift(safe_z, tol=tol),
                transport(above_place, tol=tol),
                place(place_target, tol=tol),
                release(settle=st),
                retreat(safe_z, tol=tol),
                Do(self._clear_samples, name="clear_samples"),
                Hold(0.0, steps=self.verify_samples, check=self._sample, name="observe"),
                Do(self._verify, name="verify"),
            ],
            name="pick_place",
        )
        budgets = {"grasp_miss": self.miss_retries, "grasp_slip": self.slip_regrasps, "wrong_placement": self.place_retries}
        return Retry(attempt, budgets, on_retry=self._on_retry)

    # -- steps ---------------------------------------------------------------

    def _open_cmd(self) -> float:
        """Gripper command for the approach: fully open, or (with `preshape_margin`) just wide
        enough for the widest planned grasp, so the fingers do not land on neighbours."""
        if self.preshape_margin is None:
            return 0.0
        grasps = plan_grasps(self.spec, self.max_width)
        if not grasps:
            return 0.0
        width = max(g.width for g in grasps) + self.preshape_margin
        return float(np.clip(1.0 - width / self.max_width, 0.0, 0.45))  # > 0.5 reads as "closed"

    def _disarm(self, obs: Observation, c: Context) -> str | None:
        c.monitor.disarm()
        c.bb.pop("rest_z", None)  # re-read from wherever the object is now
        c.bb["offsets"] = []
        return None

    def _plan(self, obs: Observation, c: Context) -> str | None:
        if c.obj_pos is None:
            return f"other:object '{c.obj}' not observed"
        c.bb["rest_z"] = float(c.obj_pos[2])
        q_obj = c.obj_quat if c.obj_quat is not None else np.array([1.0, 0, 0, 0])
        grasps = plan_grasps(self.spec, self.max_width, obj_quat=q_obj)
        if grasps and self.closing_axis is not None:
            r, ax = quat_to_mat(q_obj), np.asarray(self.closing_axis, dtype=float)
            grasps = [g for g in grasps if abs(float(np.dot(r @ quat_to_mat(g.quat)[:, 1], ax))) > 0.9] or grasps
        if grasps:
            _, quat, g = select_grasp(grasps, c.obj_pos, q_obj, ee_quat=obs.ee_quat, rank=c.bb["rank"])
            c.bb["grasp"], c.bb["grasp_obj_quat"] = g.pos, q_obj
            c.quat = quat
            return None
        if self.spec.shape != "mesh":
            return f"other:no_feasible_grasp(max_width={self.max_width})"
        # Mesh: no planner yet. Top-down at the observed centre, yawed with the object.
        c.bb["grasp"], c.bb["grasp_obj_quat"] = np.zeros(3), np.array([1.0, 0, 0, 0])
        c.quat = quat_mul(quat_from_axis_angle([0, 0, 1], yaw_of(q_obj)), DOWN)
        return None

    def _clear_samples(self, obs: Observation, c: Context) -> str | None:
        c.bb["samples"] = []
        return None

    def _sample(self, obs: Observation, c: Context) -> str | None:
        if c.obj in obs.objects:
            c.bb.setdefault("samples", []).append(np.asarray(obs.objects[c.obj].pos, dtype=float).copy())
        return None

    def _verify(self, obs: Observation, c: Context) -> str | None:
        # Judge placement on the mean of several raw observations, not one noisy sample:
        # a false "outside" sends the open gripper back down for a needless re-pick.
        samples = c.bb.get("samples") or []
        pos = np.mean(samples, axis=0) if samples else c.obj_pos
        if pos is None:
            return None
        d = np.abs(pos - np.asarray(self.region.center))
        if np.all(d <= np.asarray(self.region.half_extents) + self.verify_margin):
            return None
        return f"wrong_placement:observed {np.round(pos, 3).tolist()} outside {self.region.name}"

    def _on_retry(self, c: Context, reason: str) -> None:
        if reason.startswith("grasp_miss"):
            c.bb["rank"] += 1  # try the next-ranked grasp
