#!/usr/bin/env python3
"""Point the gripper at each object's start spot for a trial, so a person knows where to put it.

Needs the bridge server running, the robot blue, and a person at the e-stop. Run with
SkillForge's venv, BEFORE the session (the bridge serves one client at a time):

    .venv/bin/python scripts/show_placements.py franka_stack_can_pyramid_lab --seed 0

The arm homes, then for each object: rises to the travel height, moves over the spot, comes down
to hover `--hover` m above the table (fingertips; default 0.12, clear of a 8.4 cm can), and waits
for Enter. Place the object right under the gripper, then press Enter. Moves at `--speed` m/s.
The spots are `draw_placement(task, seed)`: exactly what `skillforge.real run --seeds <seed>` asks for.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from skillforge.contracts import Action, TaskSpec
from skillforge.real import FrankaBridgeRobotIO, MujocoController, draw_placement

ROOT = Path(__file__).resolve().parent.parent
DT = 0.05


def move_to(robot, ctl, goal, quat, tol=0.002, timeout=30.0):
    """Stream IK targets toward `goal` until the measured tool point is within `tol` (m)."""
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        q, _ = ctl.solve(Action(mode="ee", ee_pos=np.asarray(goal, float), ee_quat=quat, gripper=0.0))
        robot.send_joint_targets(q, 0.0)
        time.sleep(DT)
        p, _ = ctl.fk(robot.read_state().q)
        if np.linalg.norm(p - goal) < tol:
            return p
    raise RuntimeError(f"did not reach {np.round(goal, 3)} within {timeout:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="task id or path")
    ap.add_argument("--seed", type=int, default=0, help="the trial seed you will run")
    ap.add_argument("--hover", type=float, default=0.12, help="fingertip height above the table at each spot (m)")
    ap.add_argument("--travel", type=float, default=0.20, help="fingertip height while moving between spots (m)")
    ap.add_argument("--speed", type=float, default=0.05, help="m/s (at most 0.1)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if not 0 < a.speed <= 0.1 or a.hover < 0.10 or a.travel < a.hover:
        raise SystemExit("keep --speed <= 0.1, --hover >= 0.10 and --travel >= --hover")
    p = Path(a.task)
    p = p if p.is_file() else ROOT / "tasks" / f"{a.task}.json"
    task = TaskSpec.model_validate(json.loads(p.read_text()))
    _, targets = draw_placement(task, a.seed)

    ctl = MujocoController(task, max_ee_speed=a.speed)
    robot = FrankaBridgeRobotIO(port=a.port)
    try:
        print(f"{task.id}, seed {a.seed}: {len(targets)} spots. Homing...")
        robot.home()
        q = robot.read_state().q
        ctl.reset(q)
        _, quat = ctl.fk(q)  # keep the home orientation: pointing straight down
        for i, (name, t) in enumerate(targets.items(), 1):
            x, y = t.pos[0], t.pos[1]
            here, _ = ctl.fk(robot.read_state().q)
            move_to(robot, ctl, np.array([here[0], here[1], max(here[2], a.travel)]), quat)
            move_to(robot, ctl, np.array([x, y, a.travel]), quat)
            move_to(robot, ctl, np.array([x, y, a.hover]), quat)
            input(f"\n[{i}/{len(targets)}] '{name}': put a can right under the gripper "
                  f"(spot x={x:.3f} y={y:.3f} m). Enter when done (Ctrl-C stops): ")
        here, _ = ctl.fk(robot.read_state().q)
        move_to(robot, ctl, np.array([here[0], here[1], a.travel]), quat)
        print("\nall spots shown. Stopping; now start `skillforge.real run` with the same seed.")
        robot.stop()
    except KeyboardInterrupt:
        print("\ninterrupted: stopping")
        robot.stop()
    finally:
        robot.close()


if __name__ == "__main__":
    main()
