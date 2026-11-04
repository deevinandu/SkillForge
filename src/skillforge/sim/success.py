"""Success predicates, shared by every env so 'success' means the same thing everywhere."""

from __future__ import annotations

import numpy as np

from skillforge.contracts import ObjectInRegion, ObjectLifted, TaskSpec


def is_success(task: TaskSpec, true_obj_pos: dict[str, np.ndarray], start_obj_pos: dict[str, np.ndarray]) -> bool:
    """Evaluate on TRUE object positions (not noisy observations)."""
    s = task.success
    p = true_obj_pos[s.object]
    if isinstance(s, ObjectInRegion):
        r = task.scene.region(s.region)
        return bool(np.all(np.abs(p - np.asarray(r.center)) <= np.asarray(r.half_extents)))
    if isinstance(s, ObjectLifted):
        return bool(p[2] - start_obj_pos[s.object][2] >= s.min_height)
    raise TypeError(f"unknown success spec {type(s).__name__}")
