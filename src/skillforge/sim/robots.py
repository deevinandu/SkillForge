"""Robot profiles: everything `MujocoEnv` needs to know about one arm + gripper model.

A profile names the bodies, joints and actuators the controller and event detection use, the
home posture, the TCP, and how the gripper is commanded and measured. It also makes the
build-time edits that model needs (gripper gains, pads). `MujocoEnv` reads only the profile,
so a second arm is a new profile, not a change to the env.

Profiles are looked up by `RobotSpec.source` (`profile_for`). A source with no registered
profile gets the Panda profile, as before profiles existed; the env then checks the model really
has the Panda's names and refuses it otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

# Gripper servo on the Panda's finger tendon (menagerie: kp 100, kv 10, i.e. ~1 N per finger on a
# 4 cm cube, which let a 50-75 g cube slip out under transport accelerations). kp 1000 gives
# ~10 N per finger on a 4 cm object, still well under the real Franka Hand's 70 N continuous force.
GRIP_KP = 1000.0
GRIP_KV = 20.0
PANDA_TCP_OFFSET = 0.1034  # metres from the Panda hand frame to the fingertip pad centre, along hand z


@dataclass(frozen=True)
class RobotProfile:
    """Names and conventions of one arm + gripper model. Subclass and override the hooks below
    for a gripper that is not the Panda's (two prismatic fingers on one tendon)."""

    name: str
    arm_joints: tuple[str, ...]
    arm_actuators: tuple[str, ...]
    gripper_actuator: str
    finger_bodies: tuple[str, str]  # bodies whose contacts with an object count as a grasp
    hand_body: str  # carries the TCP site and the wrist camera
    base_body: str  # bolted to the table; its subtree is "the robot"
    home: tuple[float, ...]  # arm joints at reset; the IK's null-space pull aims here
    tcp_offset: float  # metres along hand z from the hand frame to the TCP (between the pads)
    finger_joints: tuple[str, ...] = ()

    @property
    def dof(self) -> int:
        return len(self.arm_joints)

    def check(self, spec: mujoco.MjSpec, source: str) -> None:
        """Refuse a model that does not have this profile's names."""
        missing = [n for n in (self.base_body, self.hand_body, *self.finger_bodies) if spec.body(n) is None]
        missing += [n for n in (*self.arm_actuators, self.gripper_actuator) if spec.actuator(n) is None]
        if missing:
            raise NotImplementedError(f"{source}: model lacks {missing} that robot profile '{self.name}' needs")

    def prepare(self, spec: mujoco.MjSpec, *, closed_pad_gap: float) -> None:
        """Build-time edits to the robot model (gripper gains, pad geometry). Default: none."""

    def open_gripper(self, model: mujoco.MjModel, data: mujoco.MjData, width: float) -> None:
        """Set the gripper's joints to `width` open at reset (before mj_forward)."""

    def gripper_width(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """The MEASURED opening between the fingertips, metres."""
        raise NotImplementedError

    def gripper_ctrl(self, command: float, ctrlrange: np.ndarray) -> float:
        """Actuator control for a gripper command (0 open, 1 closed). Default: ctrlrange[0] closed."""
        lo, hi = ctrlrange
        return hi + (lo - hi) * command


@dataclass(frozen=True)
class PandaProfile(RobotProfile):
    """The menagerie Franka Panda with the Franka Hand (two prismatic fingers on one tendon)."""

    name: str = "franka_panda"
    arm_joints: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
    arm_actuators: tuple[str, ...] = tuple(f"actuator{i}" for i in range(1, 8))
    gripper_actuator: str = "actuator8"
    finger_joints: tuple[str, ...] = ("finger_joint1", "finger_joint2")
    finger_bodies: tuple[str, str] = ("left_finger", "right_finger")
    hand_body: str = "hand"
    base_body: str = "link0"
    # menagerie home with joint7 turned +90 deg so the tcp starts at quat (0, 1, 0, 0): pointing
    # down, fingers closing along world y
    home: tuple[float, ...] = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853)
    tcp_offset: float = PANDA_TCP_OFFSET

    def prepare(self, spec: mujoco.MjSpec, *, closed_pad_gap: float) -> None:
        # Fingertips: menagerie has the finger mesh, one 17 x 17 mm pad box and four 6 mm knob boxes
        # per finger. The knobs doubled the contact count on a grasp (~16 -> ~8 contacts) for no
        # behavioural gain we could measure, so only the mesh and the pad collide.
        # The menagerie pads (17 x 8 x 17 mm boxes at finger-local y = 5.5 mm) stop 3.0 mm apart when
        # the fingers are fully closed, so nothing thinner than 3 mm (a mug wall) can be pinched. The
        # real Franka Hand closes its fingertips fully. Each pad is moved toward the other finger so
        # the closed gap is `closed_pad_gap` (0.003 reproduces the menagerie model).
        for fb in self.finger_bodies:
            for g in spec.body(fb).geoms:
                if g.type == mujoco.mjtGeom.mjGEOM_BOX and g.size[0] < 0.008 and g.contype:
                    g.contype = g.conaffinity = 0
                elif g.type == mujoco.mjtGeom.mjGEOM_BOX and g.contype:  # the pad
                    g.pos[1] = g.size[1] + closed_pad_gap / 2
        grip = spec.actuator(self.gripper_actuator)
        lo, hi = grip.ctrlrange  # menagerie: 0..255 -> tendon length 0..0.04 m
        grip.gainprm[0] = 0.04 * GRIP_KP / (hi - lo)
        grip.biasprm[1], grip.biasprm[2] = -GRIP_KP, -GRIP_KV

    def open_gripper(self, model: mujoco.MjModel, data: mujoco.MjData, width: float) -> None:
        for j in self.finger_joints:
            data.qpos[model.joint(j).qposadr[0]] = width / 2

    def gripper_width(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        return float(sum(data.qpos[model.joint(j).qposadr[0]] for j in self.finger_joints))


# ------------------------------------------------------------------ registry

_REGISTRY: dict[str, Callable[[], RobotProfile]] = {
    "menagerie:franka_emika_panda": PandaProfile,
}


def register_profile(source: str, factory: Callable[[], RobotProfile]) -> None:
    """Map a `RobotSpec.source` to the profile for its model."""
    _REGISTRY[source] = factory


def registered_sources() -> list[str]:
    return sorted(_REGISTRY)


def profile_for(source: str) -> RobotProfile:
    """The profile for a `RobotSpec.source`; the Panda's for a source with none registered."""
    return _REGISTRY.get(source, PandaProfile)()
