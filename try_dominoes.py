"""Run ToppleChain on a domino task in MuJoCo and show what happened.

    python try_dominoes.py                                   # one episode, seed 0, printed log
    python try_dominoes.py --view                            # watch it in the MuJoCo viewer
    python try_dominoes.py --gif runs/dominoes.gif           # save a GIF
    python try_dominoes.py --task franka_topple_domino_chain_knock --seed 3
    python try_dominoes.py --no-disturb                      # same task, disturbances off
    python try_dominoes.py --seeds 20                        # success rate over seeds 0..19

On macOS the viewer needs `mjpython try_dominoes.py --view` (comes with the mujoco package).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from skillforge.contracts import TaskSpec
from skillforge.eval import evaluate, run_episode
from skillforge.sim import make_env
from skillforge.skills import ToppleChain

TASKS = Path(__file__).resolve().parent / "tasks"
SHOW = ("disturbance", "toppled:", "grasp:", "release:", "grasp_slip", "collision", "ik_failure", "sim_")


def load(name: str, disturb: bool) -> TaskSpec:
    t = TaskSpec.model_validate_json((TASKS / f"{name}.json").read_text())
    if not disturb:
        t = t.model_copy(update={"scene": t.scene.model_copy(update={"disturbances": ()})})
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="franka_topple_domino_chain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=0, help="evaluate seeds 0..N-1 instead of one episode")
    ap.add_argument("--view", action="store_true", help="MuJoCo viewer, real time")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--no-disturb", action="store_true")
    a = ap.parse_args()
    task = load(a.task, not a.no_disturb)

    if a.seeds:
        report, eps = evaluate(make_env, ToppleChain(), task, range(a.seeds))
        print(report.summary())
        for e in eps:
            if not e.success:
                print(f"  seed {e.seed}: {e.failure} {e.failure_detail}")
        return

    env, skill = make_env(task), ToppleChain()
    if a.gif:
        from skillforge.eval.video import record_episode

        rec = record_episode(env, skill, task, a.seed, Path(a.gif), every=3, max_frames=400)
        ep = rec.episode
        print(f"gif: {rec.path} ({rec.n_frames} frames) {rec.reason}")
    else:
        viewer, dt = None, 1.0 / task.control_hz
        if a.view:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        def hook(e, i):
            if viewer is not None:
                if not viewer.is_running():
                    raise KeyboardInterrupt
                viewer.sync()
                time.sleep(dt / a.speed)

        try:
            ep = run_episode(env, skill, task, a.seed, on_step=hook)
        except KeyboardInterrupt:
            print("viewer closed")
            return
        finally:
            if viewer is not None:
                time.sleep(1.0)
                viewer.close()

    print(f"\n{task.id} seed {a.seed}: success={ep.success} steps={ep.n_steps} failure={ep.failure} {ep.failure_detail}")
    print("events:", [e for e in ep.events if e.startswith(SHOW)])
    print("skill log:")
    for line in skill.log:
        print("  ", line)


if __name__ == "__main__":
    main()
