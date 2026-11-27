#!/usr/bin/env python3
"""First motion through the frankapy bridge: small, slow, and one Enter per step.

Needs the bridge server running (scripts/frankapy_bridge_server.py, real mode), the robot
blue, and a person at the e-stop. Run with SkillForge's venv:

    .venv/bin/python scripts/bridge_first_motion.py

Steps, each after Enter (Ctrl-C at any prompt stops the arm and exits):
  1. read the state only (nothing moves);
  2. home: move to the simulator's home posture at <= 0.3 rad/s per joint, open the gripper;
  3. hold: stream the current joints for 3 s (the arm should not move);
  4. down 3 cm and back up, through the sim controller's IK at 1 cm/s;
  5. stop.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from skillforge.contracts import TaskSpec
from skillforge.real import SIM_HOME, FrankaBridgeRobotIO, MujocoController

TASK = Path(__file__).resolve().parent.parent / "tasks" / "franka_lift_cube.json"
DT = 0.05  # 20 Hz, the tasks' control rate


def ask(msg: str) -> None:
    input(f"\n>>> {msg}\n    Press Enter to do it (Ctrl-C to stop): ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--depth", type=float, default=0.03, help="metres to move down in step 4")
    ap.add_argument("--speed", type=float, default=0.01, help="m/s in step 4")
    a = ap.parse_args()
    if not 0 < a.depth <= 0.05 or not 0 < a.speed <= 0.03:
        raise SystemExit("first motion is limited to <= 5 cm at <= 3 cm/s")

    task = TaskSpec.model_validate(json.loads(TASK.read_text()))
    ctl = MujocoController(task, max_ee_speed=a.speed)
    robot = FrankaBridgeRobotIO(port=a.port)
    print(f"connected: {robot.info}")
    try:
        s = robot.read_state()
        p, _ = ctl.fk(s.q)
        print(f"1. state: joints {np.round(s.q, 3)}\n   tool point {np.round(p, 3)} m, gripper {s.gripper_width * 100:.1f} cm")
        hp, _ = ctl.fk(np.array(SIM_HOME))
        print(f"   home joints {np.round(SIM_HOME, 3)}, tool point there {np.round(hp, 3)} m; "
              f"largest joint move {np.abs(np.array(SIM_HOME) - s.q).max():.2f} rad")

        ask("2. HOME: the arm moves to the sim home posture (slow) and opens the gripper.")
        robot.home()
        s = robot.read_state()
        print(f"   at home: joints {np.round(s.q, 3)}, events {s.events}")

        ask("3. HOLD: stream the current joints for 3 s. The arm should NOT move.")
        q0 = robot.read_state().q.copy()
        t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end:
            robot.send_joint_targets(q0, 0.0)
            time.sleep(DT)
        s = robot.read_state()
        print(f"   drift {np.abs(s.q - q0).max() * 1e3:.2f} mrad, events {s.events}")

        ask(f"4. DOWN {a.depth * 100:.0f} cm and back UP at {a.speed * 100:.0f} cm/s.")
        q = robot.read_state().q.copy()
        ctl.reset(q)
        p0, quat0 = ctl.fk(q)
        from skillforge.contracts import Action

        for goal in (p0 - [0, 0, a.depth], p0):
            n = int(a.depth / a.speed / DT) + 20
            for _ in range(n):
                q, ev = ctl.solve(Action(mode="ee", ee_pos=goal, ee_quat=quat0, gripper=0.0))
                if ev:
                    print("   controller:", ev)
                robot.send_joint_targets(q, 0.0)
                time.sleep(DT)
            s = robot.read_state()
            pm, _ = ctl.fk(s.q)
            print(f"   tool point {np.round(pm, 4)} (goal {np.round(goal, 4)}), "
                  f"error {np.linalg.norm(pm - goal) * 1e3:.1f} mm, events {s.events}")

        print("\n5. stopping the stream; the arm holds where it is.")
        robot.stop()
    except KeyboardInterrupt:
        print("\ninterrupted: stopping")
        robot.stop()
    finally:
        robot.close()


if __name__ == "__main__":
    main()
