"""PickPlace v0.2: scripted closed-loop pick-and-place built from `primitives`, with recovery.

Over v0.1 (still available as `skillforge.skills.pick_place_v01.PickPlaceV01`):
- grasp orientation from the grasp planner (`grasp.py`) and the observed object orientation;
- filtered object observations instead of chasing raw noisy positions;
- a missed grasp (object not following the gripper on lift, or fingers closed on nothing) is
  retried with the next-ranked grasp (`miss_retries`, default once);
- a slip while carrying (object leaves the gripper, or the measured width collapses) triggers a
  re-grasp from wherever the object ended up (`slip_regrasps`, default twice);
- after release, if the object is observed outside the goal region it is picked again once.

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
    Node,
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


class _SettledGrip(Node):
    """Wrap a `grip` Hold: close for at least `min_steps`, then finish once the measured width has
    stopped changing for `stable` steps (or the Hold's own step budget runs out). The fixed
    4-step settle of v0.2.0 is shorter than the action latency (up to 5 steps), so the lift
    started before the fingers had squeezed the ball."""

    def __init__(self, hold, min_steps: int, stable: int, eps: float = 5e-4):
        self.hold, self.min_steps, self.stable, self.eps = hold, min_steps, stable, eps
        self.name = "grip"
        self.reset()

    def reset(self) -> None:
        self.hold.reset()
        self.w, self.k = None, 0

    def tick(self, obs, ctx):
        w = float(obs.gripper_width)
        self.k = self.k + 1 if self.w is not None and abs(w - self.w) < self.eps else 0
        self.w = w
        if self.hold.n >= self.min_steps and self.k >= self.stable:
            self.hold.n = self.hold.steps  # settled: let the Hold finish (grasp check + arm monitor)
        return self.hold.tick(obs, ctx)


class PickPlace:
    name = "pick_place"
    version = "0.2.1"

    hover = 0.10  # metres above grasp targets for pre-grasp, and for clearing/retreating
    tol = 0.006  # metres: position error that counts as "arrived"
    settle_steps = 4  # steps to hold a gripper command before moving on
    # Metres above the place height while carrying. v0.2.0 carried at `hover` (10 cm); the held
    # object creeps out of the soft finger contacts over time (worst at low friction), so a low
    # carry (shorter lift, shorter path, short drop if it does slip) loses it far less often.
    # The goal region has no walls in the scene, so nothing needs clearing.
    carry_hover = 0.03
    grip_min_steps = 8  # keep closing at least this long (covers action latency and slow actuators)
    grip_max_steps = 30  # give up waiting for the fingers to settle after this many steps
    grip_stable_steps = 3  # gripper width unchanged this many steps = fingers have closed on the object
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
        center = np.asarray(self.region.center, dtype=float)
        bottom = center[2] - self.region.half_extents[2]
        self.place_obj = np.array([center[0], center[1], max(center[2], bottom + _resting_half_height(self.spec))])
        self.ctx = Context(obj=self.spec.name)
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

        def carry_z(obs, c):
            z_obj = c.bb.get("rest_z", c.obj_pos[2] if c.obj_pos is not None else obs.ee_pos[2])
            return max(z_obj, self.place_obj[2]) + self.carry_hover

        def grasp_target(obs, c):
            g, q_obj = c.bb["grasp"], c.bb["grasp_obj_quat"]
            return c.obj_pos + quat_to_mat(q_obj) @ g

        def place_target(obs, c):
            return self.place_obj - c.monitor.offset

        def above_place(obs, c):
            return place_target(obs, c) + [0.0, 0.0, self.carry_hover]

        attempt = Sequence(
            [
                Do(self._disarm, name="disarm"),
                retreat(safe_z, tol=tol, name="clear"),
                Do(self._plan, name="plan_grasp"),
                approach(grasp_target, hover=hv, tol=tol),
                descend_until(grasp_target, tol=tol),
                _SettledGrip(grip(settle=self.grip_max_steps), self.grip_min_steps, self.grip_stable_steps),
                lift(carry_z, tol=tol),
                transport(above_place, tol=tol),
                place(place_target, tol=tol),
                release(settle=st),
                retreat(safe_z, tol=tol),
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
        return None

    def _plan(self, obs: Observation, c: Context) -> str | None:
        if c.obj_pos is None:
            return f"other:object '{c.obj}' not observed"
        c.bb["rest_z"] = float(c.obj_pos[2])
        q_obj = c.obj_quat if c.obj_quat is not None else np.array([1.0, 0, 0, 0])
        grasps = plan_grasps(self.spec, self.max_width, obj_quat=q_obj)
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

    def _verify(self, obs: Observation, c: Context) -> str | None:
        if c.obj_pos is None:
            return None
        d = np.abs(c.obj_pos - np.asarray(self.region.center))
        if np.all(d <= np.asarray(self.region.half_extents) + self.verify_margin):
            return None
        return f"wrong_placement:observed {np.round(c.obj_pos, 3).tolist()} outside {self.region.name}"

    def _on_retry(self, c: Context, reason: str) -> None:
        if reason.startswith("grasp_miss"):
            c.bb["rank"] += 1  # try the next-ranked grasp
