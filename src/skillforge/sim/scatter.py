"""Resting poses of boxes, and random initial poses (`SceneSpec.scatter`).

A box rests on the table on one of three faces (`RestFace`): "upright" on its own z axis (the
spawn convention, e.g. a standing domino), "flat" on its largest face (its thinnest axis vertical:
a domino lying face down) or "side" on the remaining face (its middle axis vertical: a domino
lying on its long edge). Non-box objects only have "upright".

Randomness: `draw_scatter` uses `np.random.default_rng([seed, 3])`, a stream of its own, so a task
without scatter keeps every other draw (scene `seed`, noise `[seed, 1]`, disturbances `[seed, 2]`).
"""

from __future__ import annotations

import math

import numpy as np

from skillforge.contracts import ObjectSpec, TaskSpec

FACES = ("upright", "flat", "side")
_SQ = math.sqrt(0.5)
# rotation taking local axis k to world +z (w, x, y, z)
_BASE = {2: np.array([1.0, 0.0, 0.0, 0.0]), 0: np.array([_SQ, 0.0, -_SQ, 0.0]), 1: np.array([_SQ, _SQ, 0.0, 0.0])}


def rest_axis(obj: ObjectSpec, face: str) -> int:
    """Index of the object's local axis that points up when it rests on `face`."""
    if face == "upright" or obj.shape != "box":
        return 2
    order = list(np.argsort(np.asarray(obj.size[:3], dtype=float), kind="stable"))
    return int(order[0] if face == "flat" else order[1])


def rest_height(obj: ObjectSpec, face: str) -> float:
    """Height of the object's centre above the table when it rests on `face`."""
    if obj.shape == "box":
        return float(obj.size[rest_axis(obj, face)])
    if obj.shape == "sphere":
        return float(obj.size[0])
    if obj.shape == "cylinder":
        return float(obj.size[1])
    return float(obj.pose.pos[2])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def rest_quat(obj: ObjectSpec, face: str, yaw: float) -> np.ndarray:
    """Orientation (w, x, y, z) resting on `face`, then turned by `yaw` about world z."""
    base = _BASE[rest_axis(obj, face)] if obj.shape == "box" else np.asarray(obj.pose.quat, dtype=float)
    qz = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    q = _qmul(qz, base)
    return q / np.linalg.norm(q)


def region_bounds(task: TaskSpec, name: str) -> tuple[np.ndarray, np.ndarray]:
    r = task.scene.region(name)
    c, h = np.asarray(r.center, dtype=float), np.asarray(r.half_extents, dtype=float)
    return c[:2] - h[:2], c[:2] + h[:2]


def draw_scatter(task: TaskSpec, seed: int) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    """Initial poses for the scattered objects: {name: (pos, quat)}, plus the values drawn (for
    the randomization sample). Empty when the task has no scatter."""
    sc = task.scene.scatter
    if sc is None:
        return {}, {}
    rng = np.random.default_rng([seed, 3])
    movable = [o.name for o in task.scene.objects if not o.fixed]
    names = list(sc.objects) or movable
    placed = [np.asarray(o.pose.pos[:2], dtype=float) for o in task.scene.objects if o.name not in names]
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sample: dict[str, float] = {}
    for name in names:
        o = task.scene.object(name)
        fi = int(rng.integers(len(sc.faces)))
        face = sc.faces[fi]
        yaw = float(rng.uniform(-math.pi, math.pi))
        xy, ok = None, False
        for _ in range(200):
            lo, hi = region_bounds(task, sc.regions[int(rng.integers(len(sc.regions)))])
            xy = rng.uniform(lo, hi)
            if all(float(np.linalg.norm(xy - p)) >= sc.min_dist for p in placed):
                ok = True
                break
        placed.append(xy)
        pos = np.array([xy[0], xy[1], rest_height(o, face)])
        poses[name] = (pos, rest_quat(o, face, yaw))
        k = f"scatter.{name}"
        sample[f"{k}.face"] = float(FACES.index(face))
        sample[f"{k}.x"], sample[f"{k}.y"], sample[f"{k}.yaw"] = float(xy[0]), float(xy[1]), yaw
        sample[f"{k}.crowded"] = 0.0 if ok else 1.0  # no spot met min_dist in 200 tries
    return poses, sample


def free_spot(cands: list[np.ndarray], others: list[np.ndarray], min_dist: float) -> list[np.ndarray]:
    """The candidate xy points at least `min_dist` from every other object, in order."""
    return [c for c in cands if all(float(np.linalg.norm(c - o)) >= min_dist for o in others)]
