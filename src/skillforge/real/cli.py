"""`python -m skillforge.real <command>`. Output goes under runs/real/ (gitignored).

  approve  record that a person approves a library version for hardware
  run      a session of trials. Only --dry-run works: there is no ROS layer yet.
  sysid    fit latency, tracking gain and gripper timing from a session -> calibration.json
  scoreboard  sim prediction against session results -> scoreboard.json and .html
  close-loop  turn a session's failed trials into pinned sim tasks and regression baselines
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "tasks"
RUNS = REPO / "runs"


def _task(path_or_id: str):
    from skillforge.contracts import TaskSpec

    p = Path(path_or_id)
    if not p.is_file():
        p = TASKS / f"{path_or_id}.json"
    return TaskSpec.model_validate(json.loads(p.read_text()))


def _seeds(spec: str) -> range:
    lo, _, hi = spec.partition("-")
    return range(int(lo), int(hi) + 1)  # inclusive, as the brief writes it: 0-19 is 20 trials


def cmd_approve(a: argparse.Namespace) -> None:
    from skillforge.agents.library import SkillLibrary

    from .runner import Approvals

    task = _task(a.task)
    entry = Approvals(a.approvals).approve(SkillLibrary(a.library), task.id, a.version, a.by)
    print(f"approved {task.id}@{a.version} for hardware by {entry['by']} (code {entry['code_sha'][:12]})")


def cmd_run(a: argparse.Namespace) -> None:
    from skillforge.agents.library import SkillLibrary
    from skillforge.contracts import SkillSource
    from skillforge.skills import load_skill

    from .fakes import make_twin_env
    from .runner import Approvals, TwinLabeller, run_session
    from .safety import SafetyLimits

    task = _task(a.task)
    if a.skill_file:
        if not a.dry_run:
            raise SystemExit("--skill-file is for dry runs only: hardware runs take an approved library version")
        path, _, cls = a.skill_file.partition(":")
        src = SkillSource(name=Path(path).stem, version="file", code=Path(path).read_text(), entrypoint=cls or "PickPlace")
    else:
        if not a.skill:
            raise SystemExit("give --skill <library version> (or --skill-file with --dry-run)")
        src, _ = SkillLibrary(a.library).load(task.id, a.skill)
    if not a.dry_run:
        Approvals(a.approvals).require_approved(task.id, src)  # raises NotApproved
        raise SystemExit("approved, but there is no hardware layer yet (real/ros is not built): use --dry-run")
    env = make_twin_env(task, limits=SafetyLimits(ee_speed=a.speed))
    out = Path(a.out) if a.out else RUNS / "real" / f"{datetime.now():%Y%m%d-%H%M%S}-dry-{task.id}"
    try:
        run_session(env, load_skill(src), task, _seeds(a.seeds), TwinLabeller(), out, mode="dry_run")
    finally:
        env.close()
    print(f"wrote {out}")


def cmd_sysid(a: argparse.Namespace) -> None:
    from .sysid import fit_session

    cal = fit_session(a.session)
    out = Path(a.out) if a.out else Path(a.session) / "calibration.json"
    out.write_text(cal.model_dump_json(indent=2))
    print(cal.summary())
    print(f"wrote {out}")


def cmd_scoreboard(a: argparse.Namespace) -> None:
    from skillforge.contracts import EvalReport

    from .scoreboard import build, write_html, write_json

    board = build(a.sessions, [EvalReport.model_validate_json(Path(p).read_text()) for p in a.sim or ()])
    out = Path(a.out)
    write_json(board, out / "scoreboard.json")
    write_html(board, out / "scoreboard.html")
    for row in board.rows:
        print(row.line())
    print(f"wrote {out}/scoreboard.json and .html")


def cmd_close_loop(a: argparse.Namespace) -> None:
    from skillforge.agents.library import SkillLibrary
    from skillforge.sim import make_env

    from .failures import close_loop
    from .sysid import Calibration

    task = _task(a.task)
    src, _ = SkillLibrary(a.library).load(task.id, a.skill)
    cal = Calibration.model_validate_json(Path(a.calibration).read_text()) if a.calibration else None
    cases = close_loop(a.session, task, src, make_env, a.tasks_dir, a.baselines_dir, seeds=range(a.n_seeds), calibration=cal)
    for c in cases:
        print(f"seed {c.seed} ({c.label_failure}): {c.task_path}, sim {c.baseline.n_success}/{len(c.baseline.seeds)}, "
              f"{'reproduced' if c.reproduced else 'NOT reproduced in sim'}")
    if not cases:
        print("no failed trials in this session")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m skillforge.real", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("approve", cmd_approve), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("task", help="task id or path to a TaskSpec json")
        p.add_argument("--library", default=str(RUNS / "library"))
        p.add_argument("--approvals", default=str(RUNS / "real" / "approved.json"))
        if name == "approve":
            p.add_argument("version")
            p.add_argument("--by", required=True, help="name of the person approving")
        else:
            p.add_argument("--skill", help="a version in the library")
            p.add_argument("--skill-file", help="path/to/skill.py[:Class], dry runs only")
            p.add_argument("--seeds", default="0-19", help="lo-hi, inclusive")
            p.add_argument("--dry-run", action="store_true", help="drive a simulated twin instead of the arm")
            p.add_argument("--speed", type=float, default=0.1, help="tool speed limit, m/s")
            p.add_argument("--out")
        p.set_defaults(fn=fn)
    p = sub.add_parser("sysid")
    p.add_argument("session", help="a session directory written by `run`")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_sysid)
    p = sub.add_parser("scoreboard")
    p.add_argument("--sessions", nargs="+", required=True)
    p.add_argument("--sim", nargs="*", help="EvalReport json files: the sim prediction per task and skill version")
    p.add_argument("--out", default=str(RUNS / "real" / "scoreboard"))
    p.set_defaults(fn=cmd_scoreboard)
    p = sub.add_parser("close-loop")
    p.add_argument("task")
    p.add_argument("session")
    p.add_argument("--skill", required=True, help="a version in the library")
    p.add_argument("--library", default=str(RUNS / "library"))
    p.add_argument("--calibration", help="calibration.json from `sysid`")
    p.add_argument("--tasks-dir", default=str(TASKS))
    p.add_argument("--baselines-dir", default=str(REPO / "tests" / "regression" / "baselines"))
    p.add_argument("--n-seeds", type=int, default=20)
    p.set_defaults(fn=cmd_close_loop)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0
