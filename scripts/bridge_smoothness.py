#!/usr/bin/env python3
"""Is motion through the bridge smooth? Moves the tool sideways and back at real skill speeds
while recording the robot's own joint readings (~100 Hz), and reports the jitter.

Needs the bridge server running (with --publish-hz, the default), the robot blue, a person at
the e-stop, and the table clear under the home pose. Run with SkillForge's venv:

    .venv/bin/python scripts/bridge_smoothness.py

Steps (one Enter each): home; hold 2 s (the baseline); 6 cm sideways and back at 5 cm/s; the same
at 10 cm/s. "Jitter" is the RMS of joint velocity minus its own 0.15 s moving average: the fast
wobble a jerking arm adds on top of the intended motion. Smooth motion stays near the hold's
level; the 2026-11-27 jerk (20 Hz steps) would be several times higher.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from skillforge.contracts import Action, TaskSpec
from skillforge.real import FrankaBridgeRobotIO, MujocoController

ROOT = Path(__file__).resolve().parent.parent
DT = 0.05
TOPIC = "/robot_state_publisher_node_1/robot_state/q"
ROS_SETUP = os.path.expanduser("~/Documents/frankapy/catkin_ws/devel/setup.bash")


class Recorder:
    """`rostopic echo -p` of the robot's joints in a subprocess (ROS lives outside this venv)."""

    def __init__(self):
        self.path = Path(tempfile.mkstemp(suffix=".csv")[1])
        self.proc = None

    def __enter__(self):
        cmd = f"source /opt/ros/noetic/setup.bash; source {ROS_SETUP}; exec rostopic echo -p {TOPIC}"
        self.proc = subprocess.Popen(["bash", "-c", cmd], stdout=self.path.open("w"), stderr=subprocess.DEVNULL)
        time.sleep(1.5)  # rostopic needs a moment to subscribe
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait(5)

    def read(self, t0, t1):
        rows = [ln.strip().split(",") for ln in self.path.read_text().splitlines() if ln[:1].isdigit()]
        t = np.array([float(r[0]) * 1e-9 for r in rows])
        q = np.array([[float(x) for x in r[1:8]] for r in rows])
        sel = (t >= t0) & (t <= t1)
        return t[sel], q[sel]


def jitter(t, q):
    """RMS (rad/s) of joint velocity minus its 0.15 s moving average, worst joint; and the rate."""
    if len(t) < 20:
        return float("nan"), 0.0
    dt = np.diff(t)
    v = np.diff(q, axis=0) / dt[:, None]
    k = max(3, int(round(0.15 / np.median(dt))))
    kernel = np.ones(k) / k
    smooth = np.column_stack([np.convolve(v[:, j], kernel, mode="same") for j in range(7)])
    core = slice(k, len(v) - k)
    return float(np.sqrt(np.mean((v[core] - smooth[core]) ** 2, axis=0)).max()), float(1 / np.median(dt))


def move(robot, ctl, goal, quat):
    t_end = time.monotonic() + 10
    while time.monotonic() < t_end:
        q, _ = ctl.solve(Action(mode="ee", ee_pos=np.asarray(goal, float), ee_quat=quat, gripper=0.0))
        robot.send_joint_targets(q, 0.0)
        time.sleep(DT)
        p, _ = ctl.fk(robot.read_state().q)
        if np.linalg.norm(p - goal) < 0.002:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--distance", type=float, default=0.06)
    a = ap.parse_args()
    task = TaskSpec.model_validate(json.loads((ROOT / "tasks" / "franka_stack_can_pyramid_lab.json").read_text()))
    robot = FrankaBridgeRobotIO(port=a.port)
    results = {}
    try:
        with Recorder() as rec:
            input("\n>>> HOME (slow). Enter to do it, Ctrl-C to stop: ")
            robot.home()
            q0 = robot.read_state().q.copy()
            input(">>> HOLD 2 s (baseline). Enter: ")
            t0 = time.time()
            while time.time() - t0 < 2.0:
                robot.send_joint_targets(q0, 0.0)
                time.sleep(DT)
            results["hold"] = rec.read(t0 + 0.3, time.time())
            for speed in (0.05, 0.10):
                input(f">>> {a.distance * 100:.0f} cm sideways and back at {speed * 100:.0f} cm/s. Enter: ")
                ctl = MujocoController(task, max_ee_speed=speed)
                q = robot.read_state().q
                ctl.reset(q)
                p0, quat = ctl.fk(q)
                t0 = time.time()
                move(robot, ctl, p0 + [0, a.distance, 0], quat)
                move(robot, ctl, p0, quat)
                results[f"{speed * 100:.0f} cm/s"] = rec.read(t0, time.time())
            robot.stop()
    except KeyboardInterrupt:
        print("\ninterrupted: stopping")
        robot.stop()
    finally:
        robot.close()
    print("\njitter (worst joint, rad/s RMS above the 0.15 s trend):")
    base = None
    for name, (t, q) in results.items():
        j, hz = jitter(t, q)
        base = base if base is not None else j
        print(f"  {name:9s} {j:.4f}  ({len(t)} samples at {hz:.0f} Hz){'' if name == 'hold' else f'  = {j / base:.1f}x the hold'}")


if __name__ == "__main__":
    main()
