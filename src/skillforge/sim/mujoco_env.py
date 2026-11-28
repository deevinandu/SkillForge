"""MuJoCo backend: any SceneSpec with a Franka-Panda-structured robot -> a physics SimEnv.

Scene: robot from `RobotSpec.source` (menagerie / mjcf / urdf), a table whose top is the
z = 0 plane, free-jointed objects from `ObjectSpec`, regions as visual-only sites, a `tcp`
site between the fingertip pads, a `wrist` camera on the hand.

Control: `Action(mode="ee")` goes through a damped-least-squares differential IK on the
`tcp` site (6-DoF when `ee_quat` is given, else holding the last commanded orientation),
with a speed-limited reference, joint-limit clamping and a null-space pull to home. The
result is a joint-position target for the arm's position servos. `mode="joint"` passes
targets through. Arm bodies are gravity-compensated, as the real Panda's controller is.

Randomness: all of it is drawn from `np.random.default_rng(seed)` at reset (scene draws),
`np.random.default_rng([seed, 1])` (observation noise stream) and `np.random.default_rng([seed, 2])`
(mid-episode disturbances, `sim/disturbances.py`) and `np.random.default_rng([seed, 3])` (random
initial resting poses, `SceneSpec.scatter`, `sim/scatter.py`), nothing else.

Disturbances (`SceneSpec.disturbances`) are applied after the physics step they are due on:
'pose' teleports the object at rest (shrinking the offset if it would interpenetrate a neighbour,
skipping it if even a quarter would), 'knock' sets an angular velocity about a horizontal axis,
'relocate' puts it down at rest on a drawn face and yaw at the first drawn free spot (or offset).
An object the fingers touch is not disturbed until it is let go. Objects named in an
`object_toppled` leaf emit `toppled:<obj>` the first time they tilt past `TOPPLE_EVENT_TILT`.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from collections import deque
from dataclasses import dataclass
from typing import get_args

import numpy as np

from . import default_mujoco_gl

default_mujoco_gl()  # must happen before mujoco is first imported; see sim/__init__.py

import mujoco  # noqa: E402

from skillforge.contracts import (  # noqa: E402
    Action,
    ColorName,
    ObjectState,
    ObjectToppled,
    Observation,
    StepInfo,
    TaskSpec,
    success_conditions,
)

from .assets import resolve_robot_source  # noqa: E402
from .disturbances import (  # noqa: E402
    RELOCATE_CLEARANCE,
    DisturbanceSchedule,
    Planned,
    knock_axis,
    quat_mul,
    yaw_quat,
)
from .scatter import draw_scatter, free_spot, rest_height, rest_quat  # noqa: E402
from .success import RestandTracker, is_success, resting_half_height, tilt_of  # noqa: E402

# Process-wide: once creating a GL context has failed, never try again. MuJoCo can abort the
# whole process (not raise) on a second attempt without a working context.
_GL_BROKEN: str | None = None


def _gl_unavailable_reason() -> str | None:
    """Reasons we know up front that a Renderer would fail (or abort)."""
    ctx = getattr(mujoco, "GLContext", None)
    if ctx is None:
        return "MuJoCo GL is disabled"
    if ctx.__module__.endswith("glfw") and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        return "GLFW backend without a DISPLAY (mujoco was imported before MUJOCO_GL was set?)"
    return None

TABLE_TOP_Z = 0.0
SHELF_CLEAR = 0.11  # metres from a shelf's inner face that arm body origins keep (link radius ~0.06-0.08)
SHELF_GAIN = 2.0  # null-space push per metre inside that band
SHELF_WATCH = ("link3", "link4", "link5", "link6", "link7", "hand")
TABLE_THICKNESS = 0.04
DROP_Z = TABLE_TOP_Z - 0.05  # an object below this has fallen off the table: episode terminated
EJECT_SPEED = 3.0  # m/s; a tabletop object moving faster than this was flung by the solver
# Foam fingertips (RobotSpec.fingertips = "foam"): sliding, torsional, rolling friction; a slower,
# softer contact (time constant 15 ms instead of MuJoCo's 20 ms default but with a lower impedance
# floor, so the pad gives a few mm under load).
FOAM_FRICTION = (1.6, 0.02, 0.001)
FOAM_SOLREF = (0.015, 1.0)
FOAM_SOLIMP = (0.6, 0.95, 0.004, 0.5, 2.0)
TOPPLE_EVENT_TILT = 0.5  # radians: an object_toppled object past this tilt emits 'toppled:<obj>' once
PENETRATION_TOL = 0.002  # metres: a teleported object may overlap a neighbour by at most this
TCP_OFFSET = 0.1034  # metres from the hand frame to the fingertip pad centre, along hand z
MAX_TIMESTEP = 0.0035  # s; physics step is the largest dt / n <= this (20 Hz control: 15 x 3.33 ms)
# Gripper servo on the finger tendon (menagerie: kp 100, kv 10, i.e. ~1 N per finger on a 4 cm
# cube, which let a 50-75 g cube slip out under transport accelerations). kp 1000 gives ~10 N
# per finger on a 4 cm object, still well under the real Franka Hand's 70 N continuous force.
GRIP_KP = 1000.0
GRIP_KV = 20.0
OBJECT_COLORS = [(0.85, 0.15, 0.15, 1), (0.15, 0.35, 0.85, 1), (0.95, 0.75, 0.1, 1), (0.2, 0.7, 0.3, 1)]
COLOR_NAMES: tuple[str, ...] = get_args(ColorName)
COLOR_RGBA: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.15, 0.15, 1), "blue": (0.15, 0.35, 0.85, 1), "green": (0.2, 0.7, 0.3, 1),
    "yellow": (0.95, 0.75, 0.1, 1), "orange": (0.95, 0.5, 0.1, 1), "purple": (0.55, 0.25, 0.75, 1),
    "white": (0.92, 0.92, 0.92, 1), "grey": (0.55, 0.55, 0.55, 1), "black": (0.08, 0.08, 0.08, 1),
}
assert set(COLOR_RGBA) == set(COLOR_NAMES)


@dataclass(frozen=True)
class PandaProfile:
    """Names in the menagerie Panda that the controller and event detection rely on."""

    arm_joints: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
    arm_actuators: tuple[str, ...] = tuple(f"actuator{i}" for i in range(1, 8))
    gripper_actuator: str = "actuator8"
    finger_joints: tuple[str, str] = ("finger_joint1", "finger_joint2")
    finger_bodies: tuple[str, str] = ("left_finger", "right_finger")
    hand_body: str = "hand"
    base_body: str = "link0"
    # menagerie home with joint7 turned +90 deg so the tcp starts at quat (0, 1, 0, 0): pointing
    # down, fingers closing along world y
    home: tuple[float, ...] = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853)


class MujocoEnv:
    """contracts.SimEnv on MuJoCo. One instance per TaskSpec; `reset(seed)` reuses the model."""

    def __init__(
        self,
        task: TaskSpec,
        *,
        render_size: tuple[int, int] = (240, 320),  # (height, width)
        obs_cameras: tuple[str, ...] = (),  # cameras rendered into Observation.images each step
        max_ee_speed: float = 0.5,  # m/s, reference speed limit for mode="ee"
        max_ee_ang_speed: float = 2.0,  # rad/s
        ik_tol: float = 0.01,  # metres; residual above this after the IK solve is an ik_failure
        ik_iters: int = 8,
        damping: float = 0.05,
        arm_armature: float = 0.5,  # kg m^2 floor on arm joint armature; see _index
    ):
        self.task = task
        self.profile = PandaProfile()
        self.render_size = render_size
        # the ctor's cameras plus the ones the task asks for, in order, no repeats
        self.obs_cameras = tuple(dict.fromkeys((*obs_cameras, *task.scene.obs_cameras)))
        self.max_ee_speed = max_ee_speed
        self.max_ee_ang_speed = max_ee_ang_speed
        self.ik_tol = ik_tol
        self.ik_iters = ik_iters
        self.damping = damping
        self.arm_armature = arm_armature
        self.model = self._build()
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)  # IK works here so physics state is untouched
        self.dt = 1.0 / task.control_hz
        self.n_substeps = max(1, round(self.dt / self.model.opt.timestep))
        self._index()
        self._nominal = {
            "geom_friction": self.model.geom_friction.copy(),
            "body_mass": self.model.body_mass.copy(),
            "body_inertia": self.model.body_inertia.copy(),
            "actuator_gainprm": self.model.actuator_gainprm.copy(),
            "actuator_biasprm": self.model.actuator_biasprm.copy(),
        }
        self._renderer = None
        self._render_warned = False
        self._sample: dict[str, float] = {}
        self.reset(0)

    # ------------------------------------------------------------------ build

    def _build(self) -> mujoco.MjModel:
        sc = self.task.scene
        rs = sc.robot
        path = resolve_robot_source(rs.source)
        spec = mujoco.MjSpec.from_file(str(path))
        p = self.profile
        if spec.body(p.base_body) is None or spec.actuator(p.gripper_actuator) is None:
            raise NotImplementedError(
                f"{rs.source}: only Franka-Panda-structured models (menagerie panda.xml naming) are supported"
            )
        # Physics: the step divides the control period exactly (see MAX_TIMESTEP). Elliptic friction
        # cones + impratio 10 make grasps behave.
        dt = 1.0 / self.task.control_hz
        spec.option.timestep = dt / int(np.ceil(dt / MAX_TIMESTEP - 1e-9))
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10.0
        spec.option.tolerance = 1e-6
        base = spec.body(p.base_body)
        base.pos = list(rs.base_pose.pos)
        base.quat = list(rs.base_pose.quat)
        for b in spec.bodies:  # gravity compensation on every robot body (the real arm has it)
            b.gravcomp = 1.0
        spec.add_exclude(bodyname1=p.base_body, bodyname2="world")  # base bolted to the table
        # Fingertips: menagerie has the finger mesh, one 17 x 17 mm pad box and four 6 mm knob boxes
        # per finger. The knobs doubled the contact count on a grasp (~16 -> ~8 contacts) for no
        # behavioural gain we could measure, so only the mesh and the pad collide.
        for fb in p.finger_bodies:
            for g in spec.body(fb).geoms:
                if g.type == mujoco.mjtGeom.mjGEOM_BOX and g.size[0] < 0.008 and g.contype:
                    g.contype = g.conaffinity = 0
        if rs.fingertips == "foam":  # foam on the pads: grippy, soft, resists twisting in the grasp
            for fb in p.finger_bodies:
                for g in spec.body(fb).geoms:
                    if g.type == mujoco.mjtGeom.mjGEOM_BOX and g.contype:  # the pad (knobs are off above)
                        g.friction = list(FOAM_FRICTION)
                        g.condim = 6
                        g.priority = 2  # above objects (1): the pad's friction and softness govern
                        g.solref = list(FOAM_SOLREF)
                        g.solimp = list(FOAM_SOLIMP)
        grip = spec.actuator(p.gripper_actuator)
        lo, hi = grip.ctrlrange  # menagerie: 0..255 -> tendon length 0..0.04 m
        grip.gainprm[0] = 0.04 * GRIP_KP / (hi - lo)
        grip.biasprm[1], grip.biasprm[2] = -GRIP_KP, -GRIP_KV
        hand = spec.body(p.hand_body)
        if spec.site(rs.ee_site) is None:
            hand.add_site(name=rs.ee_site, pos=[0, 0, TCP_OFFSET], size=[0.004, 0, 0], rgba=[1, 0, 1, 0.6], group=4)
        # wrist camera: on the hand, offset along hand x (out of the fingers' path), looking along hand +z
        hand.add_camera(name="wrist", pos=[0.07, 0, 0.04], quat=[0, 1, 0, 0], fovy=70)

        wb = spec.worldbody
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
        wb.add_light(name="top", pos=[0.4, 0, 2.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        bx, by = rs.base_pose.pos[0], rs.base_pose.pos[1]
        wb.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05],
            pos=[0, 0, TABLE_TOP_Z - 0.75], rgba=[0.3, 0.33, 0.36, 1],
        )
        tb = sc.table
        tx = bx - tb.back + tb.depth / 2  # table centre (x)
        wb.add_geom(
            name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[tb.depth / 2, tb.width / 2, TABLE_THICKNESS / 2],
            pos=[tx, by, TABLE_TOP_Z - TABLE_THICKNESS / 2], rgba=[0.62, 0.5, 0.38, 1],
        )
        if tb.shelves:  # side panels of the shelf units on both sides of the table
            for side, sgn in (("shelf_l", 1.0), ("shelf_r", -1.0)):
                wb.add_geom(
                    name=side, type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[tb.depth / 2, tb.shelf_thickness / 2, tb.shelf_height / 2],
                    pos=[tx, by + sgn * (tb.width / 2 + tb.shelf_thickness / 2), TABLE_TOP_Z + tb.shelf_height / 2],
                    rgba=[0.55, 0.56, 0.6, 1],
                )
        front = wb.add_camera(name="front", pos=[bx + 1.9, by, 0.95], fovy=50)
        front.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        front.alt.xyaxes = [0, 1, 0, -0.42, 0, 0.91]  # looking back at the robot, tilted down
        for i, o in enumerate(sc.objects):
            body = wb.add_body(name=o.name, pos=list(o.pose.pos), quat=list(o.pose.quat))
            if not o.fixed:
                body.add_freejoint(name=f"{o.name}_free")
            g = dict(
                name=o.name, mass=o.mass, friction=[o.friction, 0.005, 0.0001], condim=4,
                priority=1,  # the object's friction governs its contacts, so friction_scale is real
                rgba=list(COLOR_RGBA[o.color] if o.color else OBJECT_COLORS[i % len(OBJECT_COLORS)]),
            )
            if o.shape == "box":
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(o.size), **g)
            elif o.shape == "cylinder":
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[o.size[0], o.size[1], 0], **g)
            elif o.shape == "sphere":
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[o.size[0], 0, 0], **g)
            else:  # mesh
                mesh_name = f"{o.name}_mesh"
                spec.add_mesh(name=mesh_name, file=str(os.path.abspath(o.mesh)), scale=list(o.size))
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name, **g)
        for r in sc.regions:  # invisible ones keep their site (code finds regions by name), not drawn
            wb.add_site(
                name=r.name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(r.center),
                size=list(r.half_extents), rgba=[0.1, 0.8, 0.2, 0.25 if r.visible else 0.0],
                group=0 if r.visible else 5,
            )
        return spec.compile()

    def _index(self) -> None:
        m, p = self.model, self.profile
        self._arm_q = np.array([m.joint(j).qposadr[0] for j in p.arm_joints])
        self._arm_v = np.array([m.joint(j).dofadr[0] for j in p.arm_joints])
        self._arm_act = np.array([m.actuator(a).id for a in p.arm_actuators])
        self._grip_act = m.actuator(p.gripper_actuator).id
        self._grip_ctrl = m.actuator_ctrlrange[self._grip_act].copy()  # [closed, open]
        self._finger_q = np.array([m.joint(j).qposadr[0] for j in p.finger_joints])
        self._jnt_lo = np.array([m.jnt_range[m.joint(j).id][0] for j in p.arm_joints]) + 1e-3
        self._jnt_hi = np.array([m.jnt_range[m.joint(j).id][1] for j in p.arm_joints]) - 1e-3
        self._site = m.site(self.task.scene.robot.ee_site).id
        self._home = np.array(p.home)
        # The menagerie Panda has armature 0.1 on every arm joint against position servos with
        # kp ~5000 and kv ~500. At the 3.33 ms step that is numerically stiff: the servos
        # bang-bang between their force limits on consecutive substeps (hundreds to thousands of
        # substeps per pick-and-place episode), worst near the shoulder singularity (joint 2 ~ 0),
        # where joints 1 and 3 counter-oscillate at +-87 Nm and the TCP creeps off its target.
        # It shook grasps loose (most "grasp_slip" failures on the ball task) and stalled carries.
        # A 0.5 floor (plausible for the real arm's reflected rotor inertia) removes it.
        self.model.dof_armature[self._arm_v] = np.maximum(self.model.dof_armature[self._arm_v], self.arm_armature)
        # robot = the base body's subtree
        base = m.body(p.base_body).id
        self._robot_bodies = {b for b in range(m.nbody) if self._is_descendant(b, base)}
        self._is_robot = np.zeros(m.nbody, dtype=bool)
        self._is_robot[list(self._robot_bodies)] = True
        self._rec: list[tuple[int, int, int, int, bool]] = []
        self._finger_bodies = {m.body(n).id: n for n in p.finger_bodies}
        self._ee_bodies = self._finger_bodies.keys() | {m.body(p.hand_body).id}
        self._obj_bodies = {m.body(o.name).id: o.name for o in self.task.scene.objects}
        self._obj_geoms = np.array([m.geom(o.name).id for o in self.task.scene.objects], dtype=int)
        self._obj_free = {
            o.name: m.joint(f"{o.name}_free").qposadr[0] for o in self.task.scene.objects if not o.fixed
        }
        self._obj_dof = {
            o.name: m.joint(f"{o.name}_free").dofadr[0] for o in self.task.scene.objects if not o.fixed
        }
        self._table_geom = m.geom("table").id
        self._shelf_geoms = {m.geom(n).id: n for n in ("shelf_l", "shelf_r") if self.task.scene.table.shelves}
        # With shelves the IK keeps the arm away from them in its null space (the elbow swivel):
        # arm bodies whose origin comes within SHELF_CLEAR of a shelf's inner face are pushed back.
        tb = self.task.scene.table
        self._shelf_y = (self.task.scene.robot.base_pose.pos[1], tb.width / 2) if tb.shelves else None
        self._shelf_bodies = [b for b in self._robot_bodies if self._body_name(b) in SHELF_WATCH]
        self._jacb = np.zeros((3, m.nv))
        self._obj_body = {name: b for b, name in self._obj_bodies.items()}
        self._topple_watch = sorted(
            {c.object for c in success_conditions(self.task.success) if isinstance(c, ObjectToppled)}
        )
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))

    def _is_descendant(self, b: int, root: int) -> bool:
        while b != 0:
            if b == root:
                return True
            b = self.model.body_parentid[b]
        return False

    # ------------------------------------------------------------------ contract

    def reset(self, seed: int) -> Observation:
        m, d = self.model, self.data
        rng = np.random.default_rng(seed)
        self._noise_rng = np.random.default_rng([seed, 1])
        rnd = self.task.scene.randomization
        u = lambda r: float(rng.uniform(r.lo, r.hi))
        s = {
            "object_xy_jitter": u(rnd.object_xy_jitter),
            "object_yaw_jitter": u(rnd.object_yaw_jitter),
            "friction_scale": u(rnd.friction_scale),
            "mass_scale": u(rnd.mass_scale),
            "actuator_gain_scale": u(rnd.actuator_gain_scale),
            "obs_noise_std": u(rnd.obs_noise_std),
            "action_latency_steps": float(round(u(rnd.action_latency_steps))),
        }
        # per-object pose draws: xy uniform in [-j, j]^2, yaw uniform in [-y, y]
        offsets = {}
        for o in self.task.scene.objects:
            if o.fixed:
                continue
            dx, dy = rng.uniform(-s["object_xy_jitter"], s["object_xy_jitter"], size=2)
            dyaw = rng.uniform(-s["object_yaw_jitter"], s["object_yaw_jitter"])
            offsets[o.name] = (float(dx), float(dy), float(dyaw))
            s[f"{o.name}.dx"], s[f"{o.name}.dy"], s[f"{o.name}.dyaw"] = offsets[o.name]
        # colours: the spec's, or (shuffle_colors) a permutation of them among the movable objects
        # that have one. Drawn last, so tasks without it keep every earlier draw unchanged.
        coloured = [o for o in self.task.scene.objects if o.color and not o.fixed]
        names = [o.color for o in coloured]
        if rnd.shuffle_colors and len(coloured) > 1:
            names = [names[j] for j in rng.permutation(len(coloured))]
        self._colors = {o.name: c for o, c in zip(coloured, names, strict=True)}
        for o, c in self._colors.items():
            s[f"{o}.color"] = float(COLOR_NAMES.index(c))
        # mid-episode disturbances: their own stream, so no draw above changes
        self._disturb = DisturbanceSchedule(self.task, seed)
        s.update(self._disturb.sample)
        self._scatter, scat = draw_scatter(self.task, seed)  # random initial resting poses
        s.update(scat)
        self._sample = s

        # physical parameters, re-applied from nominal every reset
        n = self._nominal
        m.geom_friction[:] = n["geom_friction"]
        m.geom_friction[self._obj_geoms, 0] *= s["friction_scale"]
        for name, c in self._colors.items():  # this episode's colours (shuffled or as specified)
            m.geom_rgba[m.geom(name).id] = COLOR_RGBA[c]
        m.body_mass[:] = n["body_mass"]
        m.body_inertia[:] = n["body_inertia"]
        ob = list(self._obj_bodies)
        m.body_mass[ob] *= s["mass_scale"]
        m.body_inertia[ob] *= s["mass_scale"]
        k = s["actuator_gain_scale"]  # scales kp (gain and position bias together) and kv
        m.actuator_gainprm[:] = n["actuator_gainprm"]
        m.actuator_biasprm[:] = n["actuator_biasprm"]
        m.actuator_gainprm[:, 0] *= k
        m.actuator_biasprm[:, 1:3] *= k
        # Derived constants (body/dof invweight0, used to scale constraint regularisation) are
        # computed from mass and inertia at compile time. Without recomputing them, a cube whose
        # mass was scaled gets a mismatched contact impedance and the solver can fling it at
        # many m/s on a corner/edge landing: this was the 'object_ejected' artifact.
        mujoco.mj_setConst(m, d)

        mujoco.mj_resetData(m, d)
        d.qpos[self._arm_q] = self._home
        d.qpos[self._finger_q] = self.task.scene.robot.max_gripper_width / 2
        for o in self.task.scene.objects:
            if o.fixed:
                continue
            dx, dy, dyaw = offsets[o.name]
            a = self._obj_free[o.name]
            d.qpos[a : a + 3] = np.asarray(o.pose.pos) + [dx, dy, 0.0]
            qz = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, qz, np.asarray(o.pose.quat, dtype=float))
            d.qpos[a + 3 : a + 7] = q
            if o.name in self._scatter:
                d.qpos[a : a + 3], d.qpos[a + 3 : a + 7] = self._scatter[o.name]
        self._q_des = self._home.copy()
        self._grip_cmd = 0.0
        self._apply_ctrl()
        mujoco.mj_forward(m, d)

        self._ref_pos = d.site_xpos[self._site].copy()
        self._ref_quat = self._site_quat(d)
        self._t = 0.0
        self._queue: deque[Action] = deque()
        self._contacts: set[tuple[str, str]] = set()
        self._held: str | None = None
        self._miss_reported = False
        self._ik_failing = False
        self._unstable_seen = False
        self._ejected: set[str] = set()
        self._step_i = 0
        self._toppled_seen: set[str] = set()
        self._restand = RestandTracker(self.task)
        self._n_warn = self._warnings()
        self._rec = self._robot_env_contacts()
        self._start = self._true_obj_pos()
        return self._observe()

    def step(self, action: Action) -> tuple[Observation, StepInfo]:
        self._queue.append(action)
        lag = int(self._sample["action_latency_steps"])
        act = self._queue.popleft() if len(self._queue) > lag else None
        events: list[str] = []
        if act is not None:
            events += self._control(act)
        self._apply_ctrl()
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self._t += self.dt
        unstable = self._unstable()
        if unstable:
            # MuJoCo has reset the state behind our back: nothing after this is meaningful
            self._unstable_seen = True
            self._rec = []
            events.append("sim_unstable")
        else:
            events += self._contact_events()
        self._step_i += 1
        if self._disturb and not self._unstable_seen:
            events += self._apply_disturbances(events)
        events += self._topple_events()
        pos = self._true_obj_pos()
        quats = self._true_obj_quat()
        events += self._restand.update(quats)
        info = StepInfo(
            success=not self._unstable_seen
            and is_success(self.task, pos, self._start, colors=self._colors, quats=quats,
                           restood=self._restand.restood),
            terminated=self._unstable_seen or any(p[2] < DROP_Z for p in pos.values()),
            events=events,
        )
        return self._observe(), info

    def randomization_sample(self) -> dict[str, float]:
        return dict(self._sample)

    def render(self, camera: str | None = None) -> np.ndarray | None:
        global _GL_BROKEN
        if self._renderer is None:
            if _GL_BROKEN is None:
                _GL_BROKEN = _gl_unavailable_reason()
            if _GL_BROKEN is None:
                try:
                    h, w = self.render_size
                    self._renderer = mujoco.Renderer(self.model, h, w)
                except Exception as e:  # no GL on this machine: degrade, don't crash
                    _GL_BROKEN = f"{type(e).__name__}: {e}"
            if self._renderer is None:
                if not self._render_warned:
                    self._render_warned = True
                    warnings.warn(
                        f"MuJoCo offscreen rendering unavailable (MUJOCO_GL={os.environ.get('MUJOCO_GL')}): "
                        f"{_GL_BROKEN}; render() returns None",
                        stacklevel=2,
                    )
                return None
        if camera is None:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [self.task.scene.robot.base_pose.pos[0] + 0.35, 0, 0.25]
            cam.distance, cam.azimuth, cam.elevation = 2.1, 150, -25
        else:
            cam = camera
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    # ------------------------------------------------------------------ control

    def _apply_ctrl(self) -> None:
        d = self.data
        d.ctrl[self._arm_act] = self._q_des
        lo, hi = self._grip_ctrl  # menagerie: lo = closed, hi = open
        d.ctrl[self._grip_act] = hi + (lo - hi) * self._grip_cmd

    def _control(self, act: Action) -> list[str]:
        self._grip_cmd = float(act.gripper)
        if act.mode == "joint":
            q = np.asarray(act.joint_targets, dtype=float)
            self._q_des = np.clip(q[: len(self._arm_q)], self._jnt_lo, self._jnt_hi)
            self._fk(self._q_des)
            self._ref_pos = self._scratch.site_xpos[self._site].copy()
            self._ref_quat = self._site_quat(self._scratch)
            return []
        # speed-limited reference toward the commanded pose
        delta = np.asarray(act.ee_pos, dtype=float) - self._ref_pos
        n = math.sqrt(delta @ delta)
        step = self.max_ee_speed * self.dt
        self._ref_pos = self._ref_pos + (delta if n <= step else delta / n * step)
        if act.ee_quat is not None:
            target = np.asarray(act.ee_quat, dtype=float)
            target = target / np.linalg.norm(target)
            if np.dot(target, self._ref_quat) < 0:
                target = -target
            err = np.zeros(3)
            mujoco.mju_subQuat(err, target, self._ref_quat)  # local-frame rotation ref -> target
            ang = math.sqrt(err @ err)
            lim = self.max_ee_ang_speed * self.dt
            if ang > lim:
                err *= lim / ang
            q = self._ref_quat.copy()
            mujoco.mju_quatIntegrate(q, err, 1.0)
            self._ref_quat = q / np.linalg.norm(q)
        q_des, pos_err, rot_err = self._ik(self._ref_pos, self._ref_quat, self._q_des)
        failed = pos_err > self.ik_tol or rot_err > 10 * self.ik_tol
        self._q_des = q_des
        events = []
        if failed:
            # anti-windup: pull the reference back to what the arm can actually reach
            self._ref_pos = self._scratch.site_xpos[self._site].copy()
            self._ref_quat = self._site_quat(self._scratch)
            if not self._ik_failing:
                events.append("ik_failure")
        self._ik_failing = failed
        return events

    def _fk(self, q: np.ndarray) -> None:
        """Kinematics only (site poses) on the scratch data; mj_jacSite also needs mj_comPos."""
        s = self._scratch
        s.qpos[self._arm_q] = q
        mujoco.mj_kinematics(self.model, s)

    def _ik(self, pos: np.ndarray, quat: np.ndarray, q0: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Damped least squares on the tcp site, from seed q0. Returns (q, pos_err, rot_err)."""
        m, s = self.model, self._scratch
        q = q0.copy()
        lam2 = self.damping**2 * np.eye(6)
        e = np.zeros(6)
        rhs = np.zeros((6, 2))
        qerr, cur = np.zeros(4), np.zeros(4)
        site, dofs = self._site, self._arm_v
        for it in range(self.ik_iters + 1):
            self._fk(q)
            e[:3] = pos - s.site_xpos[site]
            mujoco.mju_mat2Quat(cur, s.site_xmat[site])
            cur[1:] *= -1  # conjugate
            mujoco.mju_mulQuat(qerr, quat, cur)  # world-frame rotation cur -> target
            if qerr[0] < 0:
                qerr *= -1  # shortest way round
            mujoco.mju_quat2Vel(e[3:], qerr, 1.0)
            pe, re = math.sqrt(e[:3] @ e[:3]), math.sqrt(e[3:] @ e[3:])
            if (pe < 2e-4 and re < 2e-3) or it == self.ik_iters:
                break
            mujoco.mj_comPos(m, s)  # only when a Jacobian is needed (most solves stop at it 0-1)
            mujoco.mj_jacSite(m, s, self._jacp, self._jacr, site)
            J = np.concatenate([self._jacp[:, dofs], self._jacr[:, dofs]])
            # null-space pull toward home posture, first iteration only: the damped projector leaks
            # a little into the task space, which later iterations then clean up
            v = 0.1 * (self._home - q) if it == 0 else np.zeros_like(q)
            if it == 0 and self._shelf_y is not None:
                y0, hw = self._shelf_y
                for b in self._shelf_bodies:
                    dy = float(s.xpos[b][1]) - y0
                    over = abs(dy) - (hw - SHELF_CLEAR)
                    if over > 0:
                        mujoco.mj_jacBody(m, s, self._jacb, None, b)
                        v -= SHELF_GAIN * over * math.copysign(1.0, dy) * self._jacb[1, dofs]
            rhs[:, 0], rhs[:, 1] = e, J @ v
            y = np.linalg.solve(J @ J.T + lam2, rhs)
            # DLS step J^T (JJ^T + l^2 I)^-1 e, plus v projected onto the (damped) null space
            dq = J.T @ (y[:, 0] - y[:, 1]) + v
            np.clip(q + dq, self._jnt_lo, self._jnt_hi, out=q)
        return q, pe, re

    # ------------------------------------------------------------------ sensing

    _BAD = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL)

    def _warnings(self) -> int:
        return sum(self.data.warning[w].number for w in self._BAD)

    def _unstable(self) -> bool:
        n = self._warnings()
        bad, self._n_warn = n > self._n_warn, n
        return bad

    def _site_quat(self, d: mujoco.MjData) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, d.site_xmat[self._site])
        return q

    def _true_obj_pos(self) -> dict[str, np.ndarray]:
        return {name: self.data.xpos[b].copy() for b, name in self._obj_bodies.items()}

    def _true_obj_quat(self) -> dict[str, np.ndarray]:
        return {name: self.data.xquat[b].copy() for b, name in self._obj_bodies.items()}

    # ------------------------------------------------------------------ disturbances

    def _fingers_touch(self, name: str) -> bool:
        b = self._obj_body[name]
        return name == self._held or any(rb in self._finger_bodies and ob == b for _, rb, ob, _, _ in self._rec)

    def _penetration(self, name: str) -> float:
        """Deepest overlap (metres, >= 0) between `name` and anything but the table."""
        d = self.data
        if d.ncon == 0:
            return 0.0
        g = d.contact.geom[: d.ncon]
        dist = d.contact.dist[: d.ncon]
        mine = self.model.geom(name).id
        worst = 0.0
        for (g1, g2), dd in zip(g, dist, strict=True):
            other = g2 if g1 == mine else g1 if g2 == mine else None
            if other is None or other == self._table_geom:
                continue
            worst = max(worst, -float(dd))
        return worst

    def _apply_disturbances(self, events: list[str]) -> list[str]:
        out: list[str] = []
        for p in self._disturb.update(self._step_i, events):
            name = p.obj
            if self._fingers_touch(name):
                if e := self._disturb.blocked(p):
                    out.append(e)
                continue
            if p.kind == "pose":
                ok = self._teleport(name, p)
            elif p.kind == "relocate":
                ok = self._relocate(name, p)
            else:
                self._knock(name, p)
                ok = True
            out.append(self._disturb.applied(p) if ok else self._disturb.skipped(p))
        if out:
            self._rec = self._robot_env_contacts()
        return out

    def _teleport(self, name: str, p: Planned) -> bool:
        """Move `name` by (dx, dy, dyaw), at rest; shrink the move if it would overlap a neighbour."""
        m, d = self.model, self.data
        a, v = self._obj_free[name], self._obj_dof[name]
        saved = d.qpos[a : a + 7].copy()
        for scale in (1.0, 0.5, 0.25):
            d.qpos[a : a + 2] = saved[:2] + scale * np.array([p.dx, p.dy])
            q = quat_mul(yaw_quat(scale * p.dyaw), saved[3:7])
            d.qpos[a + 3 : a + 7] = q / np.linalg.norm(q)
            d.qvel[v : v + 6] = 0.0
            mujoco.mj_forward(m, d)
            if self._penetration(name) <= PENETRATION_TOL:
                return True
        d.qpos[a : a + 7] = saved
        mujoco.mj_forward(m, d)
        return False

    def _relocate(self, name: str, p: Planned) -> bool:
        """Put `name` down at rest on `p.face` at `p.yaw_abs`: at the first drawn spot that is free,
        or (no regions) where it is plus (dx, dy)."""
        m, d = self.model, self.data
        o = self.task.scene.object(name)
        a, v = self._obj_free[name], self._obj_dof[name]
        saved = d.qpos[a : a + 7].copy()
        if p.cands:
            others = [self.data.xpos[b][:2].copy() for b, n in self._obj_bodies.items() if n != name]
            spots = free_spot(p.cands, others, RELOCATE_CLEARANCE)
        else:
            spots = [saved[:2] + np.array([p.dx, p.dy])]
        q = rest_quat(o, p.face, p.yaw_abs)
        for xy in spots:
            d.qpos[a : a + 3] = [xy[0], xy[1], rest_height(o, p.face) + 0.001]
            d.qpos[a + 3 : a + 7] = q
            d.qvel[v : v + 6] = 0.0
            mujoco.mj_forward(m, d)
            if self._penetration(name) <= PENETRATION_TOL:
                return True
        d.qpos[a : a + 7] = saved
        mujoco.mj_forward(m, d)
        return False

    def _knock(self, name: str, p: Planned) -> None:
        """Spin `name` about a horizontal axis so its top moves toward `p.heading`."""
        m, d = self.model, self.data
        b, v = self._obj_body[name], self._obj_dof[name]
        w = p.ang_vel * knock_axis(p.heading)  # world frame
        R = d.xmat[b].reshape(3, 3)
        hh = resting_half_height(self.task.scene.object(name))
        d.qvel[v : v + 3] = np.cross(w, R[:, 2] * hh)  # centre turning about its bottom face centre
        d.qvel[v + 3 : v + 6] = R.T @ w  # free-joint angular velocity is in the body frame
        mujoco.mj_forward(m, d)

    def _topple_events(self) -> list[str]:
        out = []
        for name in self._topple_watch:
            if name not in self._toppled_seen and tilt_of(self.data.xquat[self._obj_body[name]]) > TOPPLE_EVENT_TILT:
                self._toppled_seen.add(name)
                out.append(f"toppled:{name}")
        return out

    def _body_name(self, b: int) -> str:
        if b == 0:
            return "world"
        return self.model.body(b).name

    def _robot_env_contacts(self) -> list[tuple[int, int, int, int, bool]]:
        """(contact index, robot body, other body, other geom, robot is geom2) for robot-env contacts."""
        d = self.data
        if d.ncon == 0:
            return []
        g = d.contact.geom[: d.ncon]
        b = self.model.geom_bodyid[g]
        r = self._is_robot[b]
        out = []
        for i in np.flatnonzero(r[:, 0] != r[:, 1]):
            r2 = bool(r[i, 1])
            out.append((int(i), int(b[i, 1 if r2 else 0]), int(b[i, 0 if r2 else 1]), int(g[i, 0 if r2 else 1]), r2))
        return out

    def _contact_events(self) -> list[str]:
        events: list[str] = []
        touching: dict[str, set[int]] = {}  # object name -> finger body ids touching it
        pairs: set[tuple[str, str]] = set()
        self._rec = self._robot_env_contacts()
        for _, rb, ob, og, _ in self._rec:
            if ob in self._obj_bodies:
                if rb in self._finger_bodies:
                    touching.setdefault(self._obj_bodies[ob], set()).add(rb)
                    continue  # fingertip contact with an object is a grasp, not a collision
                pairs.add((self._body_name(rb), self._obj_bodies[ob]))
            elif og == self._table_geom:
                pairs.add((self._body_name(rb), "table"))
            elif og in self._shelf_geoms:
                pairs.add((self._body_name(rb), self._shelf_geoms[og]))
        for a, b in sorted(pairs - self._contacts):
            events.append(f"collision:{a}-{b}")
        self._contacts = pairs

        both = sorted(o for o, f in touching.items() if len(f) == 2)
        closed = self._grip_cmd > 0.5
        if self._held is not None and self._held not in both:
            events.append("grasp_slip" if closed else f"release:{self._held}")
            self._held = None
        if self._held is None and both and closed:
            self._held = both[0]
            events.append(f"grasp:{self._held}")
        if not closed:
            self._miss_reported = False
        elif self._held is None and not self._miss_reported and self._gripper_width() < 0.004:
            events.append("grasp_miss")  # closed all the way on nothing
            self._miss_reported = True
        # Safety net for solver artifacts: a tabletop object faster than EJECT_SPEED was flung by
        # the solver, not the skill. The known cause (stale invweight0 after mass randomization,
        # fixed in reset) is gone; keep flagging so a new one is tagged sim_artifact, not a drop.
        for name, adr in self._obj_dof.items():
            if name not in self._ejected and np.linalg.norm(self.data.qvel[adr : adr + 3]) > EJECT_SPEED:
                self._ejected.add(name)
                events.append(f"object_ejected:{name}")
        return events

    def _gripper_width(self) -> float:
        return float(np.sum(self.data.qpos[self._finger_q]))

    def _ee_force(self) -> np.ndarray:
        """Net contact force on hand + fingers from the environment, world frame."""
        m, d = self.model, self.data
        f = np.zeros(3)
        c6 = np.zeros(6)
        for i, rb, _, _, r2 in self._rec:
            if rb not in self._ee_bodies:
                continue
            mujoco.mj_contactForce(m, d, i, c6)
            fw = d.contact.frame[i].reshape(3, 3).T @ c6[:3]  # force on geom2, world frame
            f += fw if r2 else -fw
        return f

    def _observe(self) -> Observation:
        d = self.data
        sd = self._sample["obs_noise_std"]
        objs = {}
        for b, name in self._obj_bodies.items():
            p = d.xpos[b].copy()
            if sd > 0:
                p = p + self._noise_rng.normal(0.0, sd, 3)
            objs[name] = ObjectState(pos=p, quat=d.xquat[b].copy())
        images = {}
        for cam in self.obs_cameras:
            img = self.render(cam)
            if img is not None:
                images[cam] = img
        return Observation(
            t=self._t,
            qpos=d.qpos[self._arm_q].copy(),
            qvel=d.qvel[self._arm_v].copy(),
            ee_pos=d.site_xpos[self._site].copy(),
            ee_quat=self._site_quat(d),
            gripper_width=self._gripper_width(),
            objects=objs,
            images=images,
            contact_force=self._ee_force(),
        )
