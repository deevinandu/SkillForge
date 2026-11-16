"""T7: the picture an operator places an object by. Pure numpy drawing on a camera frame.

`placement_overlay` draws where the object should be (green outline of its footprint on the
table, at the target pose) and where the cameras currently estimate it to be (red cross and
outline). The operator moves the object until the two coincide. Uses T6's `project`, so it is
only as right as the camera calibration.

Tested on rendered sim frames. Not yet looked at by a person at a real table.
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import ObjectSpec
from skillforge.perception import CameraFrame, Estimate, project
from skillforge.perception.estimators import quat_yaw

from .core import Placement

GREEN = (40, 220, 60)
RED = (240, 40, 40)


def footprint(spec: ObjectSpec, pos, yaw: float, z: float | None = None, n: int = 24) -> np.ndarray:
    """The object's outline seen from above, as a closed loop of world points (N, 3).
    Box: its four corners. Cylinder, sphere, mesh: a circle of its radius."""
    x, y = float(pos[0]), float(pos[1])
    z = float(pos[2]) if z is None else z
    if spec.shape == "box":
        hx, hy = spec.size[0], spec.size[1]
        local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy], [hx, hy]])
    else:
        r = float(spec.size[0])
        a = np.linspace(0.0, 2 * np.pi, n + 1)
        local = r * np.stack([np.cos(a), np.sin(a)], axis=1)
    c, s = np.cos(yaw), np.sin(yaw)
    xy = local @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.column_stack([xy, np.full(len(xy), z)])


def _line(img: np.ndarray, p0, p1, color, width: int = 1) -> None:
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]), 1)) + 1
    us = np.round(np.linspace(p0[0], p1[0], n)).astype(int)
    vs = np.round(np.linspace(p0[1], p1[1], n)).astype(int)
    h, w = img.shape[:2]
    r = width // 2
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            u, v = us + du, vs + dv
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            img[v[ok], u[ok]] = color


def _polyline(img: np.ndarray, frame: CameraFrame, pts: np.ndarray, color, width: int) -> bool:
    uv, z = project(frame, pts)
    if np.any(z <= 0.01) or not np.all(np.isfinite(uv)):
        return False  # behind the camera
    for a, b in zip(uv[:-1], uv[1:], strict=True):
        _line(img, a, b, color, width)
    return True


def placement_overlay(frame: CameraFrame, spec: ObjectSpec, target: Placement, estimate: Estimate | None = None,
                      width: int = 2) -> np.ndarray:
    """A copy of `frame.rgb` with the target footprint in green and the estimate in red."""
    img = np.array(frame.rgb[..., :3], dtype=np.uint8, copy=True)
    # draw on the object's top face (box: half-height size[2]; cylinder: size[1]); a sphere at its equator
    top = target.pos[2] + {"box": spec.size[2], "cylinder": spec.size[1]}.get(spec.shape, 0.0)
    _polyline(img, frame, footprint(spec, target.pos, quat_yaw(np.asarray(target.quat)), z=top), GREEN, width)
    if estimate is not None and estimate.found:
        _polyline(img, frame, footprint(spec, estimate.pos, quat_yaw(np.asarray(estimate.quat)), z=top), RED, width)
        arm = 0.6 * float(max(spec.size[0], 0.01))
        c = np.array([estimate.pos[0], estimate.pos[1], top])
        for d in ([arm, 0, 0], [0, arm, 0]):
            _polyline(img, frame, np.array([c - d, c + d]), RED, width)
    return img
