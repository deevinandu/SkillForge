"""Success predicates, shared by every env so 'success' means the same thing everywhere."""

from __future__ import annotations

import numpy as np

from skillforge.contracts import (
    ColorInRegion,
    ObjectInRegion,
    ObjectLifted,
    ObjectOnObject,
    ObjectSpec,
    TaskSpec,
    success_conditions,
)


def resting_half_height(o: ObjectSpec) -> float:
    """Centre height above a flat surface the object rests on upright (mesh: 0, unknown)."""
    if o.shape == "box":
        return float(o.size[2])
    if o.shape == "cylinder":
        return float(o.size[1])
    if o.shape == "sphere":
        return float(o.size[0])
    return 0.0


def footprint_radius(o: ObjectSpec) -> float:
    """Radius of a disc that fits inside the object's top face at ANY yaw (box: the inscribed
    circle, which is conservative at the corners; positions alone do not give the yaw)."""
    if o.shape == "box":
        return float(min(o.size[0], o.size[1]))
    if o.shape == "cylinder":
        return float(o.size[0])
    return 0.0  # a sphere or mesh is not a support


def _in_region(task: TaskSpec, region: str, p: np.ndarray) -> bool:
    r = task.scene.region(region)
    return bool(np.all(np.abs(p - np.asarray(r.center)) <= np.asarray(r.half_extents)))


def _leaf(task: TaskSpec, s, pos: dict[str, np.ndarray], start: dict[str, np.ndarray], colors: dict[str, str]) -> bool:
    if isinstance(s, ColorInRegion):
        return all(_in_region(task, s.region, pos[n]) for n, c in colors.items() if c == s.color)
    p = pos[s.object]
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
) -> bool:
    """Evaluate on TRUE object positions (not noisy observations). AllOf: every leaf at once.

    `colors`: each movable object's colour this episode (the env's draw when `shuffle_colors` is
    on). None means the spec's colours."""
    if colors is None:
        colors = {o.name: o.color for o in task.scene.objects if o.color and not o.fixed}
    return all(_leaf(task, s, true_obj_pos, start_obj_pos, colors) for s in success_conditions(task.success))
