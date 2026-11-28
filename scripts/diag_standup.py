"""Can the robot stand a lying domino up? One domino, no disturbances, StandUp only.

    python scripts/diag_standup.py              # faces x yaws x sides, 3 seeds each
    python scripts/diag_standup.py --seeds 5 --yaws 0 90
    python scripts/diag_standup.py --view --face flat --yaws 45 --sides -1   # watch one

For a lying domino it also prints where the fingers touched down ("touch") on the first failure.

A run counts as stood up when StandUp finishes and the domino ends standing (tilt < 0.15 rad)
within 3 cm of its target spot. Prints, per cell, the count and the first failure: the reason
StandUp gave (the node that failed is in it, e.g. 'grasp_slip:...' while carrying, 'timeout:stand'
while turning) and the grasp tilt it planned.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from skillforge.contracts import TaskSpec
from skillforge.sim import make_env
from skillforge.sim.scatter import rest_height, rest_quat
from skillforge.skills.dominoes import StandUp, tilt

TASK = Path(__file__).resolve().parents[1] / "tasks" / "franka_build_domino_chain_scattered.json"


def one_lying(face: str, yaw: float, side: float) -> tuple[TaskSpec, np.ndarray]:
    d = json.loads(TASK.read_text())
    sc = d["scene"]
    sc["objects"] = sc["objects"][:1]
    d["success"]["conditions"] = d["success"]["conditions"][:1]
    sc["disturbances"] = []
    o = TaskSpec.model_validate(d).scene.object("domino_1")
    pos = [0.40, 0.21 * side, rest_height(o, face)]
    sc["objects"][0]["pose"] = {"pos": pos, "quat": [float(v) for v in rest_quat(o, face, yaw)]}
    sc["scatter"] = None
    sc["regions"] = [r for r in sc["regions"] if r["name"] in ("lane", "spot_domino_1", "side_r", "side_l")]
    d["id"] = "diag_standup"
    d["max_steps"] = 700
    t = TaskSpec.model_validate(d)
    return t, np.asarray(t.scene.region("spot_domino_1").center, dtype=float)


def run_one(env, task: TaskSpec, spot: np.ndarray, seed: int, viewer=None) -> tuple[bool, str]:
    obs = env.reset(seed)
    job = StandUp(task, "domino_1", spot, np.array([1.0, 0.0, 0.0]))
    plan = None
    for _ in range(task.max_steps):
        obs, _ = env.step(job.act(obs))
        if plan is None and job.ctx.bb.get("grasp") is not None:
            plan = job.ctx.bb["grasp"][2]
        if viewer is not None:
            viewer.sync()
            time.sleep(1.0 / task.control_hz)
        if not job.status.running:
            break
    o = obs.objects["domino_1"]
    up = tilt(o.quat) < 0.15 and float(np.linalg.norm(o.pos[:2] - spot[:2])) < 0.03
    why = job.status.reason if job.status.failed else ("" if up else f"ended tilt={tilt(o.quat):.2f}")
    if not job.status.running and not job.status.failed and not up:
        why = f"finished but not standing (tilt={tilt(o.quat):.2f})"
    if job.status.running:
        why = "out of steps"
    tilt_s = "none" if plan is None else f"{plan:+.2f}"
    touch = job.ctx.bb.get("touch", "no touch-down")
    return up, f"{why} [grasp tilt {tilt_s}; {touch}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--faces", nargs="*", default=["flat", "side"])
    ap.add_argument("--face", default=None, help="shorthand for --faces FACE")
    ap.add_argument("--yaws", type=float, nargs="*", default=[0, 45, 90, 135], help="degrees")
    ap.add_argument("--sides", type=float, nargs="*", default=[-1, 1], help="-1: y = -0.21, +1: y = +0.21")
    ap.add_argument("--view", action="store_true")
    a = ap.parse_args()
    faces = [a.face] if a.face else a.faces
    print(f"{'face':>5} {'yaw':>4} {'side':>4} | stood | first failure")
    for face in faces:
        for yd in a.yaws:
            for side in a.sides:
                task, spot = one_lying(face, math.radians(yd), side)
                env = make_env(task)
                viewer = None
                if a.view:
                    import mujoco.viewer

                    viewer = mujoco.viewer.launch_passive(env.model, env.data)
                ok, first = 0, ""
                for s in range(a.seeds):
                    up, why = run_one(env, task, spot, s, viewer)
                    ok += up
                    if not up and not first:
                        first = f"seed {s}: {why}"
                # (the viewer is left open: closing it from the script segfaults on some GL drivers)
                print(f"{face:>5} {yd:4.0f} {side:+4.0f} | {ok:2d}/{a.seeds:<2d} | {first}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # skip interpreter teardown: a viewer's GL context segfaults there on some drivers
