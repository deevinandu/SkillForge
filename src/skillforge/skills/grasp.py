"""Geometric grasp planning for a parallel-jaw gripper, from `ObjectSpec` geometry. No learning.

Conventions (shared with `primitives.py`):
- Quaternions are (w, x, y, z), MuJoCo order.
- Gripper frame: local +z is the approach direction (fingers point along it), local +y is the
  finger closing axis. With this convention the world "pointing down" orientation is
  q = (0, 1, 0, 0) (180 degrees about x), whose closing axis is world -y.
- A `Grasp` is expressed in the OBJECT frame: `pos` is the TCP target (the point midway between
  the fingertips), `quat` the gripper orientation. `grasp_to_world` maps it to the world frame
  using the object's (observed) pose.

Shapes: box, cylinder and sphere are planned analytically. Mesh grasps (antipodal sampling on the
mesh surface) are not implemented: `plan_grasps` returns [] for meshes and callers fall back to
a top-down grasp at the observed centre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skillforge.contracts import ObjectSpec

DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # gripper pointing straight down, closing along world y
WORLD_DOWN = np.array([0.0, 0.0, -1.0])

# ---------------------------------------------------------------- quaternion helpers


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return np.array([1.0, 0.0, 0.0, 0.0]) if n < 1e-12 else q / n


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix to (w, x, y, z), w >= 0."""
    m = np.asarray(m, dtype=float)
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = quat_normalize(np.array(q))
    return -q if q[0] < 0 else q


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation angle between two orientations, in [0, pi]."""
    d = abs(float(np.dot(quat_normalize(a), quat_normalize(b))))
    return 2.0 * float(np.arccos(min(1.0, d)))


def yaw_of(q: np.ndarray) -> float:
    """Heading of the body x axis projected on the world xy plane."""
    x = quat_to_mat(q)[:, 0]
    return float(np.arctan2(x[1], x[0]))


def frame_quat(approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """Gripper orientation whose local z is `approach` and local y is `closing` (orthonormalised)."""
    z = np.asarray(approach, dtype=float)
    z = z / np.linalg.norm(z)
    y = np.asarray(closing, dtype=float)
    y = y - np.dot(y, z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return mat_to_quat(np.column_stack([x, y, z]))


# ---------------------------------------------------------------- grasps


@dataclass
class Grasp:
    pos: np.ndarray  # (3,) TCP target, object frame
    quat: np.ndarray  # (4,) gripper orientation, object frame
    width: float  # metres: object extent across the closing axis
    score: float  # higher is better, in [0, 1]

    @property
    def approach(self) -> np.ndarray:
        """Approach direction (gripper local z) in the object frame."""
        return quat_to_mat(self.quat)[:, 2]


def grasp_to_world(g: Grasp, obj_pos: np.ndarray, obj_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(TCP position, gripper quat) in world frame for a grasp on an object at (obj_pos, obj_quat)."""
    r = quat_to_mat(obj_quat)
    return np.asarray(obj_pos, dtype=float) + r @ g.pos, quat_normalize(quat_mul(obj_quat, g.quat))


def _score(width: float, max_width: float, clearance: float, approach_world: np.ndarray) -> float:
    margin = (max_width - width - clearance) / max_width  # 0 = just fits
    margin_term = float(np.clip(margin / 0.5, 0.0, 1.0))
    align_term = float(np.clip(np.dot(approach_world, WORLD_DOWN), 0.0, 1.0))
    return 0.4 * margin_term + 0.6 * align_term


def plan_grasps(
    obj: ObjectSpec,
    max_width: float,
    obj_quat: np.ndarray | None = None,
    clearance: float = 0.01,
    depth: float = 0.02,
    n_yaw: int = 8,
    max_tilt: float = np.pi / 4,
) -> list[Grasp]:
    """Ranked grasps (best first) in the object frame.

    obj_quat: the object's world orientation, used to score/filter by how vertical the approach
      is. None means identity (the object rests as specified).
    clearance: the gripper must open at least `width + clearance`.
    depth: how far past the top surface the TCP goes along the approach (clipped at the centre).
    max_tilt: drop grasps whose approach is further than this from world-down.
    """
    q_obj = np.array([1.0, 0.0, 0.0, 0.0]) if obj_quat is None else quat_normalize(obj_quat)
    r_obj = quat_to_mat(q_obj)
    raw: list[tuple[np.ndarray, np.ndarray, float, float]] = []  # approach, closing, width, extent

    if obj.shape == "box":
        h = np.asarray(obj.size, dtype=float)
        axes = np.eye(3)
        for a in range(3):
            for sa in (1.0, -1.0):
                for c in range(3):
                    if c == a:
                        continue
                    for sc in (1.0, -1.0):  # both signs: same physical grasp, 180 deg about approach
                        raw.append((sa * axes[a], sc * axes[c], 2 * h[c], h[a]))
    elif obj.shape == "cylinder":
        r, hh = float(obj.size[0]), float(obj.size[1])
        for k in range(n_yaw):
            yaw = 2 * np.pi * k / n_yaw
            radial = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            tangent = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
            for sa in (1.0, -1.0):  # end caps, across the diameter
                raw.append((np.array([0, 0, -sa]), radial, 2 * r, hh))
            raw.append((-radial, tangent, 2 * r, r))  # side, across the diameter
            raw.append((-radial, np.array([0.0, 0.0, 1.0]), 2 * hh, r))  # side, along the axis
    elif obj.shape == "sphere":
        r = float(obj.size[0])
        # A sphere looks the same from everywhere: approach straight down in the WORLD frame.
        down_obj = r_obj.T @ WORLD_DOWN
        ref = np.array([1.0, 0, 0]) if abs(down_obj[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(down_obj, ref)
        u /= np.linalg.norm(u)
        v = np.cross(down_obj, u)
        for k in range(n_yaw):
            yaw = np.pi * k / n_yaw
            raw.append((down_obj, np.cos(yaw) * u + np.sin(yaw) * v, 2 * r, r))
    else:  # mesh: antipodal sampling not implemented yet
        return []

    out: list[Grasp] = []
    for approach, closing, width, extent in raw:
        if width + clearance > max_width:
            continue
        approach_world = r_obj @ approach
        if np.dot(approach_world, WORLD_DOWN) < np.cos(max_tilt):
            continue
        pos = -approach * max(0.0, extent - depth)
        out.append(
            Grasp(
                pos=pos,
                quat=frame_quat(approach, closing),
                width=float(width),
                score=_score(width, max_width, clearance, approach_world),
            )
        )
    out.sort(key=lambda g: -g.score)
    return out


def select_grasp(
    grasps: list[Grasp], obj_pos: np.ndarray, obj_quat: np.ndarray, ee_quat: np.ndarray | None = None, rank: int = 0
) -> tuple[np.ndarray, np.ndarray, Grasp] | None:
    """World-frame (pos, quat, grasp) for the `rank`-th physically distinct grasp.

    Grasps that differ only by a 180 degree flip about the approach axis are the same grasp for a
    parallel jaw; of each such pair the one needing less wrist rotation from `ee_quat` is kept.
    Among grasps within 0.05 of the best score, the least wrist rotation wins. `rank` cycles
    through the distinct candidates (used to try a different grasp after a miss).
    """
    if not grasps:
        return None
    rot = lambda g: 0.0 if ee_quat is None else quat_angle(grasp_to_world(g, obj_pos, obj_quat)[1], ee_quat)
    groups: dict[tuple, Grasp] = {}
    for g in grasps:
        m = quat_to_mat(g.quat)
        c = m[:, 1] if m[np.argmax(np.abs(m[:, 1])), 1] > 0 else -m[:, 1]
        key = (tuple(np.round(m[:, 2], 3)), tuple(np.round(c, 3)), tuple(np.round(g.pos, 4)))
        if key not in groups or rot(g) < rot(groups[key]):
            groups[key] = g
    distinct = sorted(groups.values(), key=lambda g: -g.score)
    best = distinct[0].score
    top = sorted([g for g in distinct if g.score >= best - 0.05], key=rot)
    ordered = top + [g for g in distinct if g.score < best - 0.05]
    g = ordered[rank % len(ordered)]
    pos, quat = grasp_to_world(g, obj_pos, obj_quat)
    return pos, quat, g
