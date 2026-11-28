"""Can the robot stand a lying domino up using the shelf edge? One domino, LedgeStandUp only.

    python scripts/diag_ledge.py                      # flat and side, 4 yaws, 2 seeds each
    python scripts/diag_ledge.py --view --face flat --yaws 0 --seeds 1   # watch one

The domino lies alone on the table at (--x, --y). LedgeStandUp lays it on the ledge board (the
top shelf) with its +z end over the edge, pinches that end with the hand level, pulls it off,
turns the hand straight down and stands the domino at (0.44, 0.09). Prints, per cell, how many
stood up and the first failure (the node that failed is in the reason: e.g. 'timeout:slide_in',
'grasp_slip:...' while pulling, 'not_on_ledge' if it did not land over the edge).
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
from skillforge.skills.dominoes import LedgeStandUp, tilt

TASK = Path(__file__).resolve().parents[1] / "tasks" / "franka_rebuild_fallen_dominoes.json"
SPOT = np.array([0.44, 0.09, 0.0])


def one_lying(face: str, yaw: float, x: float, y: float) -> TaskSpec:
    d = json.loads(TASK.read_text())
    sc = d["scene"]
    sc["objects"] = sc["objects"][:1]
    d["success"]["conditions"] = d["success"]["conditions"][:1]
    sc["disturbances"] = []
    sc["scatter"] = None
    o = TaskSpec.model_validate(d).scene.object("domino_1")
    sc["objects"][0]["pose"] = {"pos": [x, y, rest_height(o, face)], "quat": [float(v) for v in rest_quat(o, face, yaw)]}
    sc["regions"] = [r for r in sc["regions"] if r["name"] in ("lane", "spot_domino_1")]
    d["id"], d["max_steps"] = "diag_ledge", 1200
    return TaskSpec.model_validate(d)


def run_one(task: TaskSpec, env, seed: int, viewer) -> tuple[bool, str]:
    obs = env.reset(seed)
    ledge = next(f for f in task.scene.fixtures if f.ledge)
    job = LedgeStandUp(task, "domino_1", ledge, SPOT, np.array([1.0, 0.0, 0.0]), obs.objects["domino_1"].quat)
    for _ in range(task.max_steps):
        obs, _ = env.step(job.act(obs))
        if viewer is not None:
            viewer.sync()
            time.sleep(1.0 / task.control_hz)
        if not job.status.running:
            break
    o = obs.objects["domino_1"]
    up = tilt(o.quat) < 0.15 and float(np.linalg.norm(o.pos[:2] - SPOT[:2])) < 0.03
    if job.status.running:
        return False, "out of steps"
    if job.status.failed:
        return False, job.status.reason
    return up, "" if up else f"finished but not standing (tilt={tilt(o.quat):.2f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--faces", nargs="*", default=["flat", "side"])
    ap.add_argument("--face", default=None)
    ap.add_argument("--yaws", type=float, nargs="*", default=[0, 45, 90, 135], help="degrees")
    ap.add_argument("--x", type=float, default=0.42)
    ap.add_argument("--y", type=float, default=0.10)
    ap.add_argument("--view", action="store_true")
    a = ap.parse_args()
    faces = [a.face] if a.face else a.faces
    total, n = 0, 0
    print(f"{'face':>5} {'yaw':>4} | stood | first failure")
    for face in faces:
        for yd in a.yaws:
            task = one_lying(face, math.radians(yd), a.x, a.y)
            env = make_env(task)
            viewer = None
            if a.view:
                import mujoco.viewer

                viewer = mujoco.viewer.launch_passive(env.model, env.data)
            ok, first = 0, ""
            for s in range(a.seeds):
                up, why = run_one(task, env, s, viewer)
                ok += up
                first = first or (f"seed {s}: {why}" if not up else "")
            total, n = total + ok, n + a.seeds
            print(f"{face:>5} {yd:4.0f} | {ok:2d}/{a.seeds:<2d} | {first}", flush=True)
    print(f"ledge stand-up: {total}/{n} stood up")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # skip interpreter teardown: a viewer's GL context segfaults there on some drivers
