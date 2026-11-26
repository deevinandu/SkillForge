"""Geometry of a `shape="mug"` object, shared by the MuJoCo builder and the success check.

MuJoCo collides a mesh as its convex hull, which would fill both the cup and the handle loop, so a
mug is built from primitives instead: `N_WALL` thin boxes around the wall, a disc for the bottom
and a chain of capsules for the D-shaped handle. The handle sits on the mug's +x side (object
frame) in the x-z plane; the hole through it runs along the mug's y axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from skillforge.contracts import ObjectSpec

N_WALL = 20
BOTTOM_T = 0.004  # metres, thickness of the bottom disc
WALL_LIFT = 0.001  # metres between the wall's lower edge and the mug's bottom face
# mass split of a mug: wall, bottom, handle
MASS_SPLIT = (0.6, 0.25, 0.15)


@dataclass(frozen=True)
class MugPart:
    kind: str  # "box" | "cylinder" | "capsule"
    size: tuple[float, float, float]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0  # box only: rotation about the mug's z
    fromto: tuple[float, ...] | None = None  # capsule only
    mass: float = 0.0


def handle_points(o: ObjectSpec) -> list[tuple[float, float]]:
    """(x, z) centreline of the handle in the mug frame, wall to wall, a D with chamfered corners."""
    r_out, _, _ = o.size
    h = o.mug_handle()
    xo = r_out + h.reach - h.radius  # outer bar centreline
    x0 = r_out - 0.002  # start slightly inside the wall so the handle is welded on
    c = min(0.006, (h.reach - 2 * h.radius) / 3, (h.top - h.bottom - 2 * h.radius) / 4)
    return [(x0, h.top), (xo - c, h.top), (xo, h.top - c), (xo, h.bottom + c), (xo - c, h.bottom), (x0, h.bottom)]


def handle_opening(o: ObjectSpec) -> tuple[float, float, float, float]:
    """Clear opening of the handle loop in the mug frame: (x_min, x_max, z_min, z_max), y = 0."""
    r_out, _, _ = o.size
    h = o.mug_handle()
    return r_out, r_out + h.reach - 2 * h.radius, h.bottom + h.radius, h.top - h.radius


def mug_parts(o: ObjectSpec) -> list[MugPart]:
    r_out, half_h, wall = o.size
    h = o.mug_handle()
    m_wall, m_bot, m_handle = (o.mass * f for f in MASS_SPLIT)
    parts = []
    rm = r_out - wall / 2
    half_t = r_out * math.tan(math.pi / N_WALL) * 1.05  # slight overlap so the wall has no slits
    for i in range(N_WALL):
        a = 2 * math.pi * i / N_WALL
        # the wall stops WALL_LIFT above the bottom: only the bottom disc touches a table (4 contacts, not 80)
        parts.append(MugPart("box", (wall / 2, half_t, half_h - WALL_LIFT / 2),
                             (rm * math.cos(a), rm * math.sin(a), WALL_LIFT / 2), a,
                             mass=m_wall / N_WALL))
    parts.append(MugPart("cylinder", (r_out, BOTTOM_T / 2, 0.0), (0.0, 0.0, -half_h + BOTTOM_T / 2), mass=m_bot))
    pts = handle_points(o)
    for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:], strict=True):
        parts.append(MugPart("capsule", (h.radius, 0.0, 0.0), fromto=(x0, 0.0, z0, x1, 0.0, z1),
                             mass=m_handle / (len(pts) - 1)))
    return parts


def quat_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def peg_segment(peg: ObjectSpec, pos=None, quat=None) -> tuple[np.ndarray, np.ndarray]:
    """End points (root, tip) of a cylinder's axis in world frame; its axis is the local z."""
    p = np.asarray(peg.pose.pos if pos is None else pos, dtype=float)
    ax = quat_mat(peg.pose.quat if quat is None else quat)[:, 2] * float(peg.size[1])
    return p - ax, p + ax


def handle_threaded(mug: ObjectSpec, mug_pos, mug_quat, peg: ObjectSpec, peg_pos=None, peg_quat=None) -> bool:
    """Does the peg's axis segment pass through the inside of the mug's handle loop?"""
    rot = quat_mat(mug_quat)
    a, b = (rot.T @ (e - np.asarray(mug_pos, dtype=float)) for e in peg_segment(peg, peg_pos, peg_quat))
    if (a[1] > 0) == (b[1] > 0):  # both ends on one side of the loop's plane: it does not cross
        return False
    t = a[1] / (a[1] - b[1])
    x, z = a[0] + t * (b[0] - a[0]), a[2] + t * (b[2] - a[2])
    x0, x1, z0, z1 = handle_opening(mug)
    return x0 < x < x1 and z0 < z < z1
