"""Success predicates, shared by every env so 'success' means the same thing everywhere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skillforge.contracts import (
    ColorInRegion,
    ObjectHung,
    ObjectInRegion,
    ObjectLifted,
    ObjectOnObject,
    ObjectSpec,
    ObjectToppled,
    TaskSpec,
    success_conditions,
)

from .mug import handle_threaded


def resting_half_height(o: ObjectSpec) -> float:
    """Centre height above a flat surface the object rests on upright (mesh: 0, unknown)."""
    if o.shape == "box":
        return float(o.size[2])
    if o.shape == "cylinder":
        return float(o.size[1])
    if o.shape == "sphere":
        return float(o.size[0])
    if o.shape == "mug":
        return float(o.size[1])
    return 0.0


def footprint_radius(o: ObjectSpec) -> float:
    """Radius of a disc that fits inside the object's top face at ANY yaw (box: the inscribed
    circle, which is conservative at the corners; positions alone do not give the yaw)."""
    if o.shape == "box":
        return float(min(o.size[0], o.size[1]))
    if o.shape == "cylinder":
        return float(o.size[0])
    return 0.0  # a sphere or mesh is not a support


def up_axis(quat: np.ndarray) -> np.ndarray:
    """The body's own z axis in world frame, from a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def tilt_of(quat: np.ndarray) -> float:
    """Angle in [0, pi] between the body's z axis and world up (0 = upright)."""
    return float(np.arccos(np.clip(up_axis(quat)[2], -1.0, 1.0)))


def toppled(s: ObjectToppled, task: TaskSpec, pos: np.ndarray, quat: np.ndarray) -> bool:
    """The `object_toppled` leaf on one object's TRUE pose."""
    z = up_axis(quat)
    if float(np.arccos(np.clip(z[2], -1.0, 1.0))) < s.min_tilt:
        return False
    if s.direction is not None:
        h = z[:2]
        if float(np.linalg.norm(h)) < 1e-9:  # upside down: no direction (cannot pass min_tilt < pi anyway)
            return False
        d = np.asarray(s.direction, dtype=float)
        cos = float(np.dot(h, d) / (np.linalg.norm(h) * np.linalg.norm(d)))
        if float(np.arccos(np.clip(cos, -1.0, 1.0))) > s.dir_tol:
            return False
    return s.region is None or _in_region(task, s.region, pos)


def _in_region(task: TaskSpec, region: str, p: np.ndarray) -> bool:
    r = task.scene.region(region)
    return bool(np.all(np.abs(p - np.asarray(r.center)) <= np.asarray(r.half_extents)))


@dataclass
class ContactState:
    """What only a physics env knows, for success kinds that need more than positions.

    quats: object name -> (w, x, y, z); touching: object name -> names of what it touches (other
    object names, "robot", "table"); speeds: object name -> linear speed of its centre, m/s."""

    quats: dict[str, np.ndarray]
    touching: dict[str, set[str]]
    speeds: dict[str, float]


def _hung(task: TaskSpec, s: ObjectHung, pos: dict[str, np.ndarray], cs: ContactState | None) -> bool:
    if cs is None:
        raise ValueError("object_hung needs contact state (quats, contacts, speeds): only a physics env can judge it")
    fixed = {o.name for o in task.scene.objects if o.fixed}
    if not cs.touching.get(s.object) or not cs.touching[s.object] <= fixed:
        return False  # resting on nothing (falling), or touching the robot, the table or a loose object
    if cs.speeds[s.object] > s.max_speed:
        return False
    mug, peg = task.scene.object(s.object), task.scene.object(s.support)
    return handle_threaded(mug, pos[s.object], cs.quats[s.object], peg, pos[s.support], cs.quats[s.support])


def hold_steps(task: TaskSpec) -> int:
    """Consecutive control steps the instantaneous condition must hold before it counts as success:
    1 for most kinds, `hold_s` worth of steps for object_hung."""
    n = [round(c.hold_s * task.control_hz) for c in success_conditions(task.success) if isinstance(c, ObjectHung)]
    return max([1, *n])


def _leaf(
    task: TaskSpec,
    s,
    pos: dict[str, np.ndarray],
    start: dict[str, np.ndarray],
    colors: dict[str, str],
    quats: dict[str, np.ndarray] | None,
    cs: ContactState | None = None,
) -> bool:
    if isinstance(s, ObjectHung):
        return _hung(task, s, pos, cs)
    if isinstance(s, ColorInRegion):
        return all(_in_region(task, s.region, pos[n]) for n, c in colors.items() if c == s.color)
    p = pos[s.object]
    if isinstance(s, ObjectToppled):
        if quats is None:
            raise ValueError("object_toppled needs true object orientations: pass quats= to is_success")
        return toppled(s, task, p, quats[s.object])
    if isinstance(s, ObjectInRegion):
        r = task.scene.region(s.region)
        return bool(np.all(np.abs(p - np.asarray(r.center)) <= np.asarray(r.half_extents)))
    if isinstance(s, ObjectLifted):
        return bool(p[2] - start[s.object][2] >= s.min_height)
    if isinstance(s, ObjectOnObject):
        top, sup = task.scene.object(s.object), task.scene.object(s.support)
        q = pos[s.support]
        dxy = float(np.linalg.norm(p[:2] - q[:2]))
        dz = float(p[2] - q[2])
        want = resting_half_height(top) + resting_half_height(sup)
        return dxy <= footprint_radius(sup) - s.xy_margin and abs(dz - want) <= s.z_tol
    raise TypeError(f"unknown success spec {type(s).__name__}")


def is_success(
    task: TaskSpec,
    true_obj_pos: dict[str, np.ndarray],
    start_obj_pos: dict[str, np.ndarray],
    colors: dict[str, str] | None = None,
    quats: dict[str, np.ndarray] | None = None,
    contacts: ContactState | None = None,
) -> bool:
    """Evaluate on TRUE object positions (not noisy observations). AllOf: every leaf at once.

    `colors`: each movable object's colour this episode (the env's draw when `shuffle_colors` is
    on). None means the spec's colours. `quats`: TRUE (w, x, y, z) per object; only
    `object_toppled` reads them, and raises without them (taken from `contacts` if not given).
    `contacts`: orientations, contacts and speeds, needed by object_hung only. This is the
    condition at one instant; for object_hung the env also requires it to hold for
    `hold_steps(task)` steps in a row."""
    if colors is None:
        colors = {o.name: o.color for o in task.scene.objects if o.color and not o.fixed}
    if quats is None and contacts is not None:
        quats = contacts.quats
    return all(
        _leaf(task, s, true_obj_pos, start_obj_pos, colors, quats, contacts)
        for s in success_conditions(task.success)
    )
