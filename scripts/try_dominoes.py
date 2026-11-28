"""Run ToppleChain on a domino task in MuJoCo and show what happened.

    python scripts/try_dominoes.py                                   # one episode, seed 0, printed log
    python scripts/try_dominoes.py --view                            # watch it in the MuJoCo viewer
    python scripts/try_dominoes.py --gif runs/dominoes.gif           # save a GIF
    python scripts/try_dominoes.py --task franka_topple_domino_chain_knock --seed 3
    python scripts/try_dominoes.py --no-disturb                      # same task, disturbances off
    python scripts/try_dominoes.py --seeds 20                        # success rate over seeds 0..19
    python scripts/try_dominoes.py --seeds 10 --task franka_rebuild_fallen_dominoes --stand-up
    python scripts/try_dominoes.py --view --task franka_build_domino_chain_scattered   # random starts
    python scripts/try_dominoes.py --view --task franka_rebuild_domino_chain           # laid-down dominoes

On macOS the viewer needs `mjpython scripts/try_dominoes.py --view` (comes with the mujoco package).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from skillforge.contracts import TaskSpec
from skillforge.eval import evaluate, run_episode
from skillforge.sim import make_env
from skillforge.skills import ToppleChain

TASKS = Path(__file__).resolve().parents[1] / "tasks"
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
    ap.add_argument("--progress", type=int, default=100,
                    help="print step, sim speed and the skill's phase every N steps (0: off)")
    ap.add_argument("--stand-up", action="store_true", help="let ToppleChain stand fallen dominoes up (off by default)")
    a = ap.parse_args()
    task = load(a.task, not a.no_disturb)
    ToppleChain.stand_up = a.stand_up or ToppleChain.stand_up

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

        clock = {"t": time.time(), "i": 0, "idle": 0}

        class Idle(Exception):
            pass

        def hook(e, i):
            # ToppleChain idles once it has nothing left to try; the episode would run on to max_steps
            clock["idle"] = clock["idle"] + 1 if getattr(skill, "phase", "") == "idle" else 0
            if clock["idle"] > 100:
                raise Idle
            if a.progress and i and i % a.progress == 0:
                now = time.time()
                rate = (i - clock["i"]) / max(now - clock["t"], 1e-9)
                clock.update(t=now, i=i)
                last = skill.log[-1] if getattr(skill, "log", None) else ""
                print(f"  step {i:5d}  {rate:6.1f} steps/s  phase={getattr(skill, 'phase', '?')}  last: {last}",
                      flush=True)
            if viewer is not None:
                if not viewer.is_running():
                    raise KeyboardInterrupt
                viewer.sync()
                time.sleep(dt / a.speed)

        try:
            ep = run_episode(env, skill, task, a.seed, on_step=hook)
        except KeyboardInterrupt:
            print("viewer closed", flush=True)
            os._exit(0)  # skip interpreter teardown: the viewer's GL context hangs or segfaults there
        except Idle:
            print(f"\n{task.id} seed {a.seed}: the skill gave up (idle), stopped early. skill log:")
            for line in skill.log:
                print("  ", line)
            sys.stdout.flush()
            os._exit(0)
        finally:
            if viewer is not None:
                time.sleep(1.0)  # not closed from here: that segfaults on some GL drivers

    print(f"\n{task.id} seed {a.seed}: success={ep.success} steps={ep.n_steps} failure={ep.failure} {ep.failure_detail}")
    print("events:", [e for e in ep.events if e.startswith(SHOW)])
    print("skill log:")
    for line in skill.log:
        print("  ", line)
    if a.view:  # skip interpreter teardown: the viewer's GL context segfaults there on some drivers
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
