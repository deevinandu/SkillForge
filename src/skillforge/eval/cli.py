"""`python -m skillforge.eval <command>`. Output goes under runs/ (gitignored) by default.

  bench       serial vs parallel timing on StubEnv and CostlyStubEnv
  sweep       robustness sweep of a library skill on a task -> robustness.json
  regression  `update` rewrites baselines, `check` compares library skills to them
  dashboard   render index.html from history.json / robustness.json / regression.json
  demo        forge (NullImprover) + sweep + regression + GIFs + dashboard on the stub task
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "tasks"
BASELINES = REPO / "tests" / "regression" / "baselines"
RUNS = REPO / "runs"


def _task(path_or_id: str):
    from skillforge.contracts import TaskSpec

    p = Path(path_or_id)
    if not p.is_file():
        p = TASKS / f"{path_or_id}.json"
    return TaskSpec.model_validate(json.loads(p.read_text()))


def _env(name: str):
    from skillforge.sim import make_env

    from . import bench

    return {"default": make_env, "costly": bench.CostlyStubEnv, "render": bench.RenderStubEnv}[name]


def cmd_bench(a: argparse.Namespace) -> None:
    from skillforge.sim import StubEnv

    from .bench import CostlyStubEnv, bench
    from .regression import library_sources

    src = library_sources()[a.skill]
    task = _task(a.task)
    for env, n in ((StubEnv, a.seeds * 4), (CostlyStubEnv, a.seeds)):
        print(bench(env, src, task, list(range(n)), a.workers).model_dump_json())


def cmd_sweep(a: argparse.Namespace) -> None:
    from .regression import library_sources
    from .robustness import sweep

    rep = sweep(_env(a.env), library_sources()[a.skill], _task(a.task), axes=a.axes or None,
                seeds=range(a.seeds), workers=a.workers, others=a.others)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rep.model_dump_json(indent=2))
    print(rep.summary())
    print(f"wrote {out}")


def cmd_regression(a: argparse.Namespace) -> int:
    from .parallel import EvalPool
    from .regression import check, library_sources, load_baselines, load_tasks, update_baselines

    tasks, lib = load_tasks(TASKS), library_sources()
    if a.action == "update":
        for p in update_baselines(_env("default"), tasks, lib, Path(a.baselines), workers=a.workers):
            print(f"wrote {p}")
        return 0
    results, bad = [], 0
    with EvalPool(a.workers) as pool:
        for b in load_baselines(Path(a.baselines)):
            if b.task_id not in tasks or b.skill not in lib:
                print(f"[SKIP] {b.task_id}__{b.skill}: task or skill no longer present")
                continue
            r = check(_env("default"), lib[b.skill], tasks[b.task_id], b, pool=pool)
            results.append(r)
            bad += not r.passed
            print(r.summary())
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
    return 1 if bad else 0


def _load_list(path: str | None, model):
    if not path or not Path(path).is_file():
        return []
    data = json.loads(Path(path).read_text())
    return [model.model_validate(d) for d in (data if isinstance(data, list) else [data])]


def cmd_dashboard(a: argparse.Namespace) -> None:
    from .dashboard import write_dashboard
    from .regression import RegressionResult
    from .robustness import RobustnessReport

    p = write_dashboard(
        Path(a.out),
        history=a.history if a.history and Path(a.history).is_file() else (),
        robustness=_load_list(a.robustness, RobustnessReport),
        regression=_load_list(a.regression, RegressionResult),
        title=a.title,
    )
    print(f"wrote {p}")


def cmd_demo(a: argparse.Namespace) -> None:
    from skillforge.agents import NullImprover
    from skillforge.forge import forge
    from skillforge.skills import load_skill

    from .bench import RenderStubEnv
    from .dashboard import load_history, write_dashboard, write_history
    from .parallel import EvalPool
    from .regression import check, library_sources, load_baselines
    from .robustness import sweep
    from .video import record_selected

    out = Path(a.out)
    task = _task(a.task)
    if a.noise > 0:  # the stub task is too easy to show failures; add observation noise
        from skillforge.contracts import Range

        from .robustness import with_randomization

        task = with_randomization(task, task.scene.randomization.model_copy(update={"obs_noise_std": Range(lo=0.0, hi=a.noise)}))
    src = library_sources()[a.skill]
    hist = forge(task, src, NullImprover(), RenderStubEnv, rounds=2, rollouts=a.seeds, target=1.01)
    rounds = load_history(hist)
    # attach GIFs to the last round's failures (and one success)
    from .harness import evaluate

    skill = load_skill(src)
    rep, eps = evaluate(RenderStubEnv, skill, task, range(a.seeds))
    recs, eps, rep = record_selected(RenderStubEnv, skill, task, eps, out / "videos", report=rep)
    rounds[-1] = rounds[-1].model_copy(update={"report": rep})
    write_history(rounds, out / "history.json")
    with EvalPool(a.workers) as pool:
        rob = sweep(RenderStubEnv, src, task, seeds=range(a.seeds), pool=pool)
        regs = [check(_env("default"), library_sources()[b.skill], _task(a.task), b, pool=pool)
                for b in load_baselines(BASELINES) if b.task_id == task.id and b.skill in library_sources()]
    (out / "robustness.json").write_text(rob.model_dump_json(indent=2))
    succ = [(f"seed {r.seed} · success", r.path) for r in recs if r.path and r.episode and r.episode.success]
    p = write_dashboard(out / "index.html", history=rounds, robustness=[rob], regression=regs,
                        videos=succ, title=f"SkillForge eval · {task.id}")
    print(rob.summary())
    print(f"videos: {[r.path for r in recs]}")
    print(f"wrote {p}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m skillforge.eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", default="stub_pick_cube_to_bin")
    common.add_argument("--skill", default="pick_place")
    common.add_argument("--workers", type=int, default=None)
    common.add_argument("--seeds", type=int, default=50)

    b = sub.add_parser("bench", parents=[common])
    b.set_defaults(fn=cmd_bench)
    s = sub.add_parser("sweep", parents=[common])
    s.add_argument("--env", choices=["default", "costly", "render"], default="default")
    s.add_argument("--axes", nargs="*")
    s.add_argument("--others", choices=["task", "nominal"], default="task")
    s.add_argument("--out", default=str(RUNS / "eval" / "robustness.json"))
    s.set_defaults(fn=cmd_sweep)
    r = sub.add_parser("regression", parents=[common])
    r.add_argument("action", choices=["update", "check"])
    r.add_argument("--baselines", default=str(BASELINES))
    r.add_argument("--out", default=None)
    r.set_defaults(fn=cmd_regression)
    d = sub.add_parser("dashboard")
    d.add_argument("--history")
    d.add_argument("--robustness")
    d.add_argument("--regression")
    d.add_argument("--title", default="SkillForge eval")
    d.add_argument("--out", default=str(RUNS / "eval" / "index.html"))
    d.set_defaults(fn=cmd_dashboard)
    m = sub.add_parser("demo", parents=[common])
    m.add_argument("--out", default=str(RUNS / "demo"))
    m.add_argument("--noise", type=float, default=0.015, help="obs_noise_std upper bound for the demo task (m)")
    m.set_defaults(fn=cmd_demo)

    a = ap.parse_args(argv)
    if a.cmd == "bench" and a.workers is None:
        import os

        a.workers = os.cpu_count() or 1
    return a.fn(a) or 0
