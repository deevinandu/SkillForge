"""`python -m skillforge.real <command>`. Output goes under runs/real/ (gitignored).

  approve  record that a person approves a library version for hardware
  register evaluate a hand-written skill in sim and add it to the library (accepted if good enough)
  run      a session of trials: --dry-run on the sim twin, or on the lab arm (bridge + camera)
  sysid    fit latency, tracking gain and gripper timing from a session -> calibration.json
  scoreboard  sim prediction against session results -> scoreboard.json and .html
  close-loop  turn a session's failed trials into pinned sim tasks and regression baselines
  camera   capture one RealSense frame -> rgb/depth png, frame.npz; --extrinsics adds a calibration check
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
        entry = Approvals(a.approvals).require_approved(task.id, src)  # raises NotApproved
        if not a.extrinsics:
            raise SystemExit("a hardware run needs --extrinsics (the camera calibration, e.g. "
                             "runs/real/handeye/<date>/T_world_cam.json)")
        from .lab import make_lab_env
        from .runner import TerminalLabeller

        out = Path(a.out) if a.out else RUNS / "real" / f"{datetime.now():%Y%m%d-%H%M%S}-hw-{task.id}"
        print(f"{src.name}@{src.version} approved for hardware by {entry['by']} at {entry['at']}")
        print(f"tool speed limit {a.speed} m/s; a person must be at the e-stop. Placement picture: {out}/placement.png")
        predict = None
        if a.free_placement and a.predict > 0:
            from .free_placement import sim_prediction

            def predict(pinned, src=src, n=a.predict):
                return sim_prediction(pinned, src, n)
        if a.free_placement:
            print("free placement: put the objects anywhere in the start area; the sim predicts each layout "
                  f"({a.predict} runs). Layouts and predictions: {out}/layouts/")
        env = make_lab_env(task, extrinsics=a.extrinsics, port=a.port, speed=a.speed, picture=out / "placement.png",
                           free_placement=a.free_placement, predict=predict, layout_dir=out / "layouts")
        try:
            run_session(env, load_skill(src), task, _seeds(a.seeds), TerminalLabeller(), out, mode="hardware")
        finally:
            env.close()
        print(f"wrote {out}")
        return
    env = make_twin_env(task, limits=SafetyLimits(ee_speed=a.speed))
    out = Path(a.out) if a.out else RUNS / "real" / f"{datetime.now():%Y%m%d-%H%M%S}-dry-{task.id}"
    try:
        run_session(env, load_skill(src), task, _seeds(a.seeds), TwinLabeller(), out, mode="dry_run")
    finally:
        env.close()
    print(f"wrote {out}")


def cmd_register(a: argparse.Namespace) -> None:
    """Evaluate a hand-written skill class in sim and add it to the library, accepted only if
    the Wilson 95% lower bound of its success rate reaches --min-lower."""
    import importlib

    from skillforge.agents.library import SkillLibrary
    from skillforge.eval import evaluate_parallel
    from skillforge.eval.harness import wilson
    from skillforge.eval.regression import source_of
    from skillforge.sim import make_env

    task = _task(a.task)
    mod, _, cls_name = a.skill_class.partition(":")
    cls = getattr(importlib.import_module(mod), cls_name)
    src = source_of(cls)
    lib = SkillLibrary(a.library)
    if lib.has(task.id, src.version):
        raise SystemExit(f"{task.id}@{src.version} is already in the library at {lib.root}; versions are immutable")
    seeds = _seeds(a.seeds)
    print(f"evaluating {src.name}@{src.version} on {task.id}, seeds {a.seeds} ({len(seeds)} episodes, sim)...")
    report, _ = evaluate_parallel(make_env, src, task, seeds)
    lo, hi = wilson(report.n_success, report.n_rollouts)
    accepted = lo >= a.min_lower
    note = (f"registered from {a.skill_class}; sim {report.n_success}/{report.n_rollouts}, "
            f"Wilson 95% [{lo:.3f}, {hi:.3f}], bar {a.min_lower}")
    lib.add(src, report, accepted, note=note)
    print(note)
    if not accepted:
        raise SystemExit(f"NOT accepted: the lower bound {lo:.3f} is below {a.min_lower}. It is stored, but cannot be approved.")
    print(f"accepted into {lib.root}/{task.id}/{src.version}. To approve it for hardware:\n"
          f"  python -m skillforge.real approve {a.task} {src.version} --by \"<your name>\"")


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


def cmd_camera(a: argparse.Namespace) -> None:
    import numpy as np

    from .realsense import (
        RealSenseCamera,
        RealSenseFrameSource,
        calibration_overlay,
        list_devices,
        load_extrinsics,
    )

    devices = list_devices()
    for name, serial in devices:
        print(f"found {name} serial {serial}")
    if not devices:
        raise SystemExit("no RealSense camera found: check the USB 3 cable and `rs-enumerate-devices`")
    T = load_extrinsics(a.extrinsics) if a.extrinsics else np.eye(4)
    cam = RealSenseCamera(a.name, T, serial=a.serial, width=a.width, height=a.height)
    out = Path(a.out) if a.out else RUNS / "real" / "camera" / f"{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    with RealSenseFrameSource([cam]) as src:
        frame = src.capture(a.name)
        if frame is None:
            raise SystemExit("camera started but no frame arrived")
        info = src.info(a.name)
    valid = frame.depth > 0
    info.update(valid_depth_fraction=float(valid.mean()),
                median_depth=float(np.median(frame.depth[valid])) if valid.any() else None,
                extrinsics=str(a.extrinsics) if a.extrinsics else None)
    (out / "camera.json").write_text(json.dumps(info, indent=2))
    np.savez_compressed(out / "frame.npz", rgb=frame.rgb, depth=frame.depth, K=frame.K, T_world_cam=frame.T_world_cam)
    try:
        import imageio.v3 as iio
    except ImportError:
        iio = None
    if iio is not None:
        iio.imwrite(out / "rgb.png", frame.rgb)
        d = np.clip(frame.depth / max(float(np.percentile(frame.depth[valid], 99)), 1e-3), 0, 1) if valid.any() else frame.depth
        iio.imwrite(out / "depth.png", (d * 255).astype(np.uint8))
        if a.extrinsics:
            iio.imwrite(out / "calibration_check.png", calibration_overlay(frame))
    print(f"K fx={info['K'][0][0]:.1f} fy={info['K'][1][1]:.1f} cx={info['K'][0][2]:.1f} cy={info['K'][1][2]:.1f}, "
          f"depth valid {info['valid_depth_fraction']:.0%}, median {info['median_depth'] or 0:.3f} m")
    print(f"wrote {out}")


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
            p.add_argument("--extrinsics", help="camera calibration (hardware runs): T_world_cam .json or .npy")
            p.add_argument("--port", type=int, default=8765, help="frankapy bridge server port (hardware runs)")
            p.add_argument("--free-placement", action="store_true",
                           help="hardware: objects anywhere in the start area, measured and checked, instead of drawn spots")
            p.add_argument("--predict", type=int, default=50,
                           help="with --free-placement: sim runs on each measured layout (0 = skip the prediction)")
            p.add_argument("--out")
        p.set_defaults(fn=fn)
    p = sub.add_parser("register", help="evaluate a hand-written skill in sim and add it to the library")
    p.add_argument("task", help="task id or path to a TaskSpec json")
    p.add_argument("--skill-class", default="skillforge.skills:PlaceInOrder", help="module:Class")
    p.add_argument("--seeds", default="0-199", help="lo-hi, inclusive")
    p.add_argument("--min-lower", type=float, default=0.9, help="accept only if the Wilson 95%% lower bound reaches this")
    p.add_argument("--library", default=str(RUNS / "library"))
    p.set_defaults(fn=cmd_register)
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
    p = sub.add_parser("camera", help="capture one RealSense frame; with --extrinsics, draw the calibration check")
    p.add_argument("--name", default="overview")
    p.add_argument("--serial", help="camera serial; default: the only one connected")
    p.add_argument("--extrinsics", help="world<-camera 4x4 as .npy or .json")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_camera)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0
