"""State and action vectors for learning: Observation/Action <-> flat float arrays.

Two layers:

1. **Raw vectors** (what the dataset stores, `observation.state` and `action`). Lossless enough
   to rebuild any feature later: arm qpos and qvel, ee pose, measured gripper width, every
   movable object's pose (task order), and the goal. Layout is described by `state_names()`
   and saved in the dataset's meta, so a reader never has to guess column order.
2. **Policy features** (`policy_features`), computed from raw state: positions relative to the
   end effector, rotations as 6D (first two columns of the rotation matrix, continuous and free
   of the q / -q ambiguity). Absolute joint angles are left out on purpose: they tie the policy
   to the training workspace, and the relative features carry the same information for the
   approach/transport phases.

Quaternions are (w, x, y, z), MuJoCo order. Units: metres, radians.
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import Action, ObjectInRegion, Observation, TaskSpec
from skillforge.skills.grasp import mat_to_quat, quat_to_mat

ACTION_NAMES: tuple[str, ...] = ("ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz", "gripper")
ACTION_DIM = len(ACTION_NAMES)


def movable_objects(task: TaskSpec) -> list[str]:
    return [o.name for o in task.scene.objects if not o.fixed]


def goal_of(task: TaskSpec) -> tuple[np.ndarray, np.ndarray]:
    """(goal centre, goal half-extents). For a lift task: the object start lifted by min_height."""
    s = task.success
    if isinstance(s, ObjectInRegion):
        r = task.scene.region(s.region)
        return np.asarray(r.center, float), np.asarray(r.half_extents, float)
    o = task.scene.object(s.object)
    return np.asarray(o.pose.pos, float) + [0.0, 0.0, s.min_height], np.zeros(3)


def state_names(task: TaskSpec, n_q: int) -> list[str]:
    names = [f"qpos{i}" for i in range(n_q)] + [f"qvel{i}" for i in range(n_q)]
    names += ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz", "gripper_width"]
    for o in movable_objects(task):
        names += [f"{o}.{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    names += ["goal_x", "goal_y", "goal_z", "goal_hx", "goal_hy", "goal_hz"]
    return names


def state_vector(obs: Observation, task: TaskSpec) -> np.ndarray:
    parts = [obs.qpos, obs.qvel, obs.ee_pos, obs.ee_quat, [obs.gripper_width]]
    for o in movable_objects(task):
        st = obs.objects.get(o)
        parts += [np.full(7, np.nan)] if st is None else [st.pos, st.quat]
    c, h = goal_of(task)
    parts += [c, h]
    return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])


def action_vector(a: Action) -> np.ndarray:
    if a.mode != "ee":
        raise ValueError("learning records ee-mode actions only")
    q = np.array([1.0, 0, 0, 0]) if a.ee_quat is None else np.asarray(a.ee_quat, float)
    return np.concatenate([np.asarray(a.ee_pos, float), q, [float(a.gripper)]])


def vector_action(v: np.ndarray) -> Action:
    v = np.asarray(v, float)
    q = v[3:7] / max(np.linalg.norm(v[3:7]), 1e-9)
    return Action(mode="ee", ee_pos=v[:3].copy(), ee_quat=q, gripper=float(np.clip(v[7], 0.0, 1.0)))


# ---------------------------------------------------------------- rotations


def quat_to_rot6(q: np.ndarray) -> np.ndarray:
    m = quat_to_mat(np.asarray(q, float))
    return np.concatenate([m[:, 0], m[:, 1]])


def rot6_to_quat(r: np.ndarray) -> np.ndarray:
    """Gram-Schmidt the two columns back into a rotation, then to (w, x, y, z)."""
    a, b = np.asarray(r[:3], float), np.asarray(r[3:6], float)
    x = a / max(np.linalg.norm(a), 1e-9)
    b = b - x * np.dot(x, b)
    y = b / max(np.linalg.norm(b), 1e-9)
    return mat_to_quat(np.stack([x, y, np.cross(x, y)], axis=1))


# ---------------------------------------------------------------- policy features


class Layout:
    """Column indices into a raw state vector, from its names."""

    def __init__(self, names: list[str], obj: str):
        ix = {n: i for i, n in enumerate(names)}
        self.n_q = sum(n.startswith("qpos") for n in names)
        self.ee_pos = [ix["ee_x"], ix["ee_y"], ix["ee_z"]]
        self.ee_quat = [ix[k] for k in ("ee_qw", "ee_qx", "ee_qy", "ee_qz")]
        self.width = ix["gripper_width"]
        self.obj_pos = [ix[f"{obj}.{k}"] for k in ("x", "y", "z")]
        self.obj_quat = [ix[f"{obj}.{k}"] for k in ("qw", "qx", "qy", "qz")]
        self.goal = [ix["goal_x"], ix["goal_y"], ix["goal_z"]]
        self.qvel = [ix[f"qvel{i}"] for i in range(self.n_q)]


FEATURE_NAMES: tuple[str, ...] = (
    "ee_x", "ee_y", "ee_z",
    *(f"ee_r6_{i}" for i in range(6)),
    "gripper_width",
    "obj_rel_x", "obj_rel_y", "obj_rel_z",
    *(f"obj_r6_{i}" for i in range(6)),
    "goal_rel_x", "goal_rel_y", "goal_rel_z",
    "obj_goal_x", "obj_goal_y", "obj_goal_z",
)
FEATURE_DIM = len(FEATURE_NAMES)


def policy_features(states: np.ndarray, lay: Layout) -> np.ndarray:
    """(T, D_state) raw states -> (T, FEATURE_DIM) policy features."""
    s = np.atleast_2d(states)
    ee = s[:, lay.ee_pos]
    obj = s[:, lay.obj_pos]
    goal = s[:, lay.goal]
    ee_r6 = np.stack([quat_to_rot6(q) for q in s[:, lay.ee_quat]])
    obj_r6 = np.stack([quat_to_rot6(q) for q in s[:, lay.obj_quat]])
    return np.concatenate(
        [ee, ee_r6, s[:, [lay.width]], obj - ee, obj_r6, goal - ee, goal - obj], axis=1
    ).astype(np.float32)


# ---------------------------------------------------------------- policy action targets
#
# A policy action is (dpos 3, rot6 6, gripper 1) = 10 numbers. dpos is the commanded ee target
# minus the current ee position, clipped to `pos_clip` metres. The env's controller moves the
# reference at most max_ee_speed / control_hz (2.5 cm at 0.5 m/s, 20 Hz) per step toward the
# target, so a target farther than a few cm only sets a direction: clipping keeps the command's
# effect and removes a large, uninformative spread from the regression targets.

POLICY_ACTION_DIM = 10


def encode_actions(actions: np.ndarray, states: np.ndarray, lay: Layout, pos_clip: float) -> np.ndarray:
    """(T, 8) raw actions + (T, D) states -> (T, 10): target relative to that step's ee, clipped."""
    ee = states[:, lay.ee_pos]
    d = actions[:, :3] - ee
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.where(n > pos_clip, d / np.maximum(n, 1e-12) * pos_clip, d)
    r6 = np.stack([quat_to_rot6(q) for q in actions[:, 3:7]])
    return np.concatenate([d, r6, actions[:, 7:8]], axis=1).astype(np.float32)


def chunk_targets(enc: np.ndarray, states: np.ndarray, lay: Layout, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Action chunks for every step t: steps t..t+H-1, positions re-expressed relative to ee_t.

    Returns (T, H, 10) targets and a (T, H) bool mask (False past the episode end, where the
    last action is repeated as padding, as ACT does).
    """
    T = len(enc)
    ee = states[:, lay.ee_pos].astype(np.float32)
    idx = np.arange(T)[:, None] + np.arange(horizon)[None, :]
    mask = idx < T
    idx = np.minimum(idx, T - 1)
    out = enc[idx].copy()  # (T, H, 10)
    out[:, :, :3] += ee[idx] - ee[:, None, :]  # target_k = ee_{t+k} + d_k, relative to ee_t
    return out, mask


def history_index(T: int, n_hist: int) -> np.ndarray:
    """(T, n_hist) indices of the frames each step sees, oldest first, clamped at episode start."""
    return np.maximum(np.arange(T)[:, None] - np.arange(n_hist)[::-1][None, :], 0)
