"""PickFromClutter v0.1: PickPlace that chooses the grasp whose hand clears the other objects.

In clutter the parallel-jaw hand is the problem, not the fingers: the Panda hand is about 0.2 m
long along the closing axis and its underside is under 4 cm above the TCP, so gripping a small cube with
the closing axis pointing at a tall neighbour drives the hand into it. Before each grasp this skill
scores every distinct grasp by the clearance between a box model of the hand (at the height it
will reach) and every other object's observed footprint. A grasp that is blocked only by short
neighbours is raised up the object (keeping at least `min_finger_depth` of it between the
fingers) until the hand passes over them. The skill takes the grasp that is clear with the least
raise, else the one with the most clearance. A grasp miss moves on to the next one.

It also carries the object high enough to pass over the tallest neighbour (the default 10 cm\nhover drags a small cube across the lid of a 10 cm can), and the fingers open only to the grasp width plus `preshape_margin` instead of fully, so they\ndo not come down on a neighbour. Everything else (approach, grip, lift, transport, place, retries, slip recovery) is PickPlace v0.2.
"""

from __future__ import annotations

import numpy as np

from skillforge.contracts import ObjectSpec, Observation
from skillforge.skills.grasp import plan_grasps, quat_to_mat, select_grasp
from skillforge.skills.pick_place import PickPlace
from skillforge.skills.primitives import Context

# Box model of the Panda hand in the grasp frame (metres): half-length along the closing axis,
# half-width across it, and how far above the TCP its underside sits. Measured from the menagerie
# model's collision geometry (hand spans y -0.104..0.100, x +-0.032, 0.037..0.129 above the TCP).
HAND_HALF_LEN = 0.104
HAND_HALF_WID = 0.032
HAND_BOTTOM = 0.037


def footprint_radius(o: ObjectSpec) -> float:
    """Radius of a circle that covers the object's footprint in any yaw."""
    if o.shape == "box":
        return float(np.hypot(o.size[0], o.size[1]))
    return float(o.size[0])  # cylinder and sphere: radius


def top_height(o: ObjectSpec, z_center: float) -> float:
    """Highest point above the table for an object resting upright at `z_center`."""
    half = {"box": o.size[2], "cylinder": o.size[1], "sphere": o.size[0]}.get(o.shape, o.size[2])
    return float(z_center + half)


def hand_clearance(
    grasp_pos: np.ndarray, grasp_quat: np.ndarray, others: list[tuple[ObjectSpec, np.ndarray]]
) -> float:
    """Smallest gap in metres between the hand box and any other object (negative = overlap).

    Only objects tall enough to reach the hand's underside count; the fingers may brush the rest.
    """
    m = quat_to_mat(grasp_quat)
    close = m[:2, 1]
    n = np.linalg.norm(close)
    close = close / n if n > 1e-6 else np.array([1.0, 0.0])
    across = np.array([-close[1], close[0]])
    gap = np.inf
    for spec, pos in others:
        if top_height(spec, pos[2]) < grasp_pos[2] + HAND_BOTTOM:
            continue
        d = pos[:2] - grasp_pos[:2]
        along, side = abs(float(d @ close)), abs(float(d @ across))
        # distance from the object's centre to the hand rectangle, minus its footprint radius
        dx, dy = max(along - HAND_HALF_LEN, 0.0), max(side - HAND_HALF_WID, 0.0)
        gap = min(gap, float(np.hypot(dx, dy)) - footprint_radius(spec))
    return gap


def object_half_height(o: ObjectSpec) -> float:
    return float({"box": o.size[2], "cylinder": o.size[1], "sphere": o.size[0]}.get(o.shape, o.size[2]))


class PickFromClutter(PickPlace):
    name = "pick_from_clutter"
    version = "0.1.0"

    clear_margin = 0.005  # metres of gap wanted between the hand and any neighbour
    min_finger_depth = 0.012  # metres of the object that must stay between the fingers
    raise_step = 0.002
    preshape_margin = 0.02  # open to the grasp width + 2 cm, not the full 8 cm
    carry_clearance = 0.04  # metres between the carried object's underside and the tallest neighbour

    def reset(self, task) -> None:
        # Carry high enough that the held object passes over every neighbour: the hover above the
        # object's resting height must cover the tallest neighbour plus the object's own height.
        spec = task.scene.object(task.success.object)
        rest = spec.pose.pos[2]
        tallest = max(
            (top_height(o, o.pose.pos[2]) for o in task.scene.objects if o.name != spec.name), default=0.0
        )
        need = tallest + 2 * object_half_height(spec) + self.carry_clearance - rest
        self.hover = max(PickPlace.hover, need)
        super().reset(task)

    def _plan(self, obs: Observation, c: Context) -> str | None:
        if c.obj_pos is None:
            return f"other:object '{c.obj}' not observed"
        c.bb["rest_z"] = float(c.obj_pos[2])
        q_obj = c.obj_quat if c.obj_quat is not None else np.array([1.0, 0, 0, 0])
        grasps = plan_grasps(self.spec, self.max_width, obj_quat=q_obj)
        if not grasps:
            return f"other:no_feasible_grasp(max_width={self.max_width})"
        others = [
            (o, np.asarray(obs.objects[o.name].pos, float))
            for o in self.task.scene.objects
            if o.name != self.spec.name and o.name in obs.objects
        ]
        # distinct grasps in PickPlace's own order, then the clearest first (stable on ties)
        seen, cands = set(), []
        for rank in range(len(grasps)):
            pos, quat, g = select_grasp(grasps, c.obj_pos, q_obj, ee_quat=obs.ee_quat, rank=rank)
            key = (tuple(np.round(g.pos, 4)), tuple(np.round(g.quat, 3)))
            if key in seen:
                break  # select_grasp cycles once it runs out of distinct grasps
            seen.add(key)
            cands.append((rank, pos, quat, g))
        top = c.obj_pos[2] + object_half_height(self.spec)
        scored = []
        for rank, pos, quat, g in cands:
            max_raise = max(0.0, top - self.min_finger_depth - pos[2])
            best = None
            for dz in np.arange(0.0, max_raise + 1e-9, self.raise_step):
                gap = hand_clearance(pos + [0.0, 0.0, dz], quat, others)
                if best is None or gap > best[0] + 1e-9:
                    best = (gap, float(dz))
                if gap >= self.clear_margin:
                    best = (gap, float(dz))
                    break
            gap, dz = best
            clear = gap >= self.clear_margin
            scored.append((not clear, dz if clear else -gap, rank, quat, g, dz, gap))
        scored.sort(key=lambda t: t[:3])
        _, _, _, quat, g, dz, gap = scored[c.bb["rank"] % len(scored)]
        r_obj = quat_to_mat(q_obj)
        c.bb["grasp"] = g.pos + r_obj.T @ np.array([0.0, 0.0, dz])
        c.bb["grasp_obj_quat"] = q_obj
        c.bb["clearance"], c.bb["grasp_raise"] = gap, dz
        c.quat = quat
        return None
