"""Where does picking a domino from the staging row fail? One domino, no disturbances.

    python scripts/diag_pick.py                 # grid of staging positions x wrist yaw, 10 seeds each
    python scripts/diag_pick.py --seeds 20

For each (x, y, yaw) it stages a single domino there, lets ToppleChain carry it to the lane spot
and push it, and counts how often the carry fails ("repair failed: ...") and how often the
whole thing succeeds. Prints the step and phase of the first failure per cell.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from skillforge.contracts import TaskSpec
from skillforge.eval import run_episode
from skillforge.sim import make_env
from skillforge.skills import ToppleChain

TASK = Path(__file__).resolve().parents[1] / "tasks" / "franka_setup_domino_chain.json"


def one_domino(x: float, y: float, yaw: float) -> TaskSpec:
    d = json.loads(TASK.read_text())
    sc = d["scene"]
    dom = sc["objects"][0]
    dom["name"] = "domino_1"
    dom["pose"] = {"pos": [x, y, dom["size"][2]], "quat": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]}
    sc["objects"] = [dom]
    sc["regions"] = [r for r in sc["regions"] if r["name"] in ("lane", "spot_domino_1")]
    sc["disturbances"] = []
    d["success"]["conditions"] = d["success"]["conditions"][:1]
    d["id"] = f"diag_pick_{round(x * 100)}_{round(y * 100)}_{round(yaw * 100)}".replace("-", "m")
    d["max_steps"] = 900
    return TaskSpec.model_validate(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--xs", type=float, nargs="*", default=[0.36, 0.46, 0.56, 0.66])
    ap.add_argument("--ys", type=float, nargs="*", default=[-0.18, 0.18])
    ap.add_argument("--yaws", type=float, nargs="*", default=[0.0, math.pi / 2])
    a = ap.parse_args()
    print(f"{'x':>5} {'y':>6} {'yaw':>5} | carry_ok  success | first failure")
    for x in a.xs:
        for y in a.ys:
            for yaw in a.yaws:
                task = one_domino(x, y, yaw)
                env = make_env(task)
                carried = ok = 0
                first = ""
                for s in range(a.seeds):
                    sk = ToppleChain()
                    ep = run_episode(env, sk, task, s)
                    bad = [line for line in sk.log if "repair failed" in line]
                    carried += not bad
                    ok += ep.success
                    if not first and (bad or not ep.success):
                        slip = [e for e in ep.events if e.startswith(("grasp_slip", "collision", "ik_failure"))]
                        first = f"seed {s}: {bad[0] if bad else ep.failure} {slip[:3]}"
                if hasattr(env, "close"):
                    env.close()
                print(f"{x:5.2f} {y:6.2f} {yaw:5.2f} | {carried:4d}/{a.seeds:<3d} {ok:4d}/{a.seeds:<3d} | {first}")


if __name__ == "__main__":
    main()
