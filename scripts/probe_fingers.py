"""How low can the fingertip pads get before the fingers touch the table? (MuJoCo, no objects)

    python scripts/probe_fingers.py

For each grasp tilt (radians about the closing axis) and gripper state, lowers the hand at
(0.40, -0.21) 1 mm at a time and prints the TCP height (= pad centre) at the first finger/hand
contact with the table, and the lowest TCP height reached. A domino lying flat is 10 mm tall: the
pads can pinch it only if the TCP can get to about 5-8 mm.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from skillforge.contracts import Action, TaskSpec
from skillforge.sim import make_env
from skillforge.skills.dominoes import _axis_rot, hand_quat
from skillforge.skills.grasp import mat_to_quat, quat_to_mat

TASK = Path(__file__).resolve().parents[1] / "tasks" / "franka_topple_domino_chain.json"


def main() -> None:
    d = json.loads(TASK.read_text())
    d["scene"]["objects"] = d["scene"]["objects"][:1]
    d["scene"]["objects"][0]["pose"]["pos"] = [0.62, 0.25, 0.04]  # out of the way
    d["scene"]["disturbances"] = []
    d["success"]["conditions"] = d["success"]["conditions"][:1]
    task = TaskSpec.model_validate(d)
    env = make_env(task)
    m = env.model
    hand_bodies = set(env._ee_bodies)  # fingers + hand, as the env's robot profile names them
    table = m.geom("table").id
    print(f"{'tilt':>5} {'gripper':>8} | first table contact: TCP z | lowest TCP z | force at lowest")
    for tilt in (0.0, 0.35, 0.5, 0.65):
        for grip in (0.0, 1.0):
            env.reset(0)
            R0 = quat_to_mat(hand_quat(0.0))
            q = mat_to_quat(_axis_rot(R0[:, 1], tilt) @ R0)
            xy = np.array([0.40, -0.21])
            obs = None
            for _ in range(60):  # get above the spot
                obs, _ = env.step(Action(ee_pos=np.array([*xy, 0.08]), ee_quat=q, gripper=grip))
            first, lowest, force = None, 1.0, 0.0
            for k in range(90):
                z = 0.06 - 0.001 * k
                for _ in range(3):
                    obs, _ = env.step(Action(ee_pos=np.array([*xy, z]), ee_quat=q, gripper=grip))
                d_ = env.data
                touch = False
                for i in range(d_.ncon):
                    g1, g2 = d_.contact.geom[i]
                    other = g2 if g1 == table else g1 if g2 == table else None
                    if other is not None and m.geom_bodyid[other] in hand_bodies:
                        touch = True
                if touch and first is None:
                    first = float(obs.ee_pos[2])
                if obs.ee_pos[2] < lowest:
                    lowest = float(obs.ee_pos[2])
                    f = obs.contact_force
                    force = float(np.linalg.norm(f)) if f is not None else float("nan")
            fs = "never" if first is None else f"{first * 1000:6.1f} mm"
            print(f"{tilt:5.2f} {('open' if grip == 0 else 'closed'):>8} | {fs:>12} | {lowest * 1000:6.1f} mm | {force:.1f} N")


if __name__ == "__main__":
    main()
