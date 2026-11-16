"""T7: the sim2real scoreboard. Per skill version on a task: what sim predicted, what happened.

Rows are built from session directories written by `runner.run_session` and from sim
`EvalReport`s for the same task and skill version. Sessions of the same task, version and mode
are pooled. A dry run is a simulated twin, not a real arm: it gets its own row and is never
counted as a real result.

`write_json` is the record. `write_html` is a minimal static table until T4 renders the JSON in
the dashboard. Every piece of text from a session (notes, tags, names) is escaped.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from skillforge.contracts import EvalReport
from skillforge.eval.harness import wilson


class Agreement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    agree: int
    false_success: int  # the automatic judge said success, the label said no
    missed_success: int
    ci95: tuple[float, float]


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    skill: str
    skill_version: str
    mode: str  # "hardware" or "dry_run"
    sessions: list[str]
    n: int
    n_success: int
    ci95: tuple[float, float]
    failure_counts: dict[str, int]
    sim_n: int | None = None  # the sim prediction for this task and version, if one was given
    sim_n_success: int | None = None
    sim_ci95: tuple[float, float] | None = None
    intervals_overlap: bool | None = None  # do the sim and real 95% intervals overlap
    estimator_vs_label: Agreement | None = None
    vlm_vs_label: Agreement | None = None  # None until T6's judge has verdicts
    safety_event_counts: dict[str, int] = {}
    videos: list[str] = []

    def line(self) -> str:
        real = f"{self.n_success}/{self.n} ({self.ci95[0]:.0%}-{self.ci95[1]:.0%})"
        sim = "no sim report" if self.sim_n is None else (
            f"sim {self.sim_n_success}/{self.sim_n} ({self.sim_ci95[0]:.0%}-{self.sim_ci95[1]:.0%})")
        return f"{self.skill}@{self.skill_version} on {self.task_id} [{self.mode}]: {real}; {sim}"


class Scoreboard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[Row]
    created_at: str


def _agreement(pairs: list[tuple[bool, bool]]) -> Agreement | None:
    """pairs of (judge, label)."""
    if not pairs:
        return None
    same = sum(a == b for a, b in pairs)
    return Agreement(n=len(pairs), agree=same, false_success=sum(a and not b for a, b in pairs),
                     missed_success=sum(b and not a for a, b in pairs), ci95=wilson(same, len(pairs)))


def build(sessions: Iterable[str | Path], sim_reports: Sequence[EvalReport] = ()) -> Scoreboard:
    groups: dict[tuple[str, str, str, str], list[tuple[Path, dict]]] = {}
    for s in sessions:
        s = Path(s)
        side = json.loads((s / "real.json").read_text())
        groups.setdefault((side["task_id"], side["skill"], side["skill_version"], side["mode"]), []).append((s, side))
    sim = {(r.task_id, r.skill, r.skill_version): r for r in sim_reports}
    rows = []
    for (task_id, skill, version, mode), items in sorted(groups.items()):
        trials = [(s, t) for s, side in items for t in side["trials"]]
        n, k = len(trials), sum(t["label_success"] for _, t in trials)
        ci = wilson(k, n)
        ref = sim.get((task_id, skill, version))
        rows.append(Row(
            task_id=task_id, skill=skill, skill_version=version, mode=mode,
            sessions=[str(s) for s, _ in items], n=n, n_success=k, ci95=ci,
            failure_counts=dict(Counter(t["label_failure"] for _, t in trials if not t["label_success"])),
            sim_n=ref.n_rollouts if ref else None,
            sim_n_success=ref.n_success if ref else None,
            sim_ci95=ref.ci95 if ref else None,
            intervals_overlap=(ci[0] <= ref.ci95[1] and ref.ci95[0] <= ci[1]) if ref else None,
            estimator_vs_label=_agreement([(t["auto_success"], t["label_success"]) for _, t in trials]),
            vlm_vs_label=_agreement([(t["vlm_success"], t["label_success"]) for _, t in trials if t["vlm_success"] is not None]),
            safety_event_counts=dict(Counter(e for _, t in trials for e in {ev for _, ev in t["safety_events"]})),
            videos=[str(s / t["video"]) for s, t in trials if t["video"]],
        ))
    return Scoreboard(rows=rows, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def write_json(board: Scoreboard, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(board.model_dump_json(indent=2))
    return path


def _rate(k: int | None, n: int | None, ci) -> str:
    return "none" if n is None else f"{k}/{n} <small>({ci[0]:.0%}&ndash;{ci[1]:.0%})</small>"


def _agree(a: Agreement | None) -> str:
    if a is None:
        return "none"
    return f"{a.agree}/{a.n} <small>({a.false_success} false success, {a.missed_success} missed)</small>"


def write_html(board: Scoreboard, path: str | Path) -> Path:
    e = html.escape
    body = []
    for r in board.rows:
        mode = "real arm" if r.mode == "hardware" else "dry run (simulated twin)"
        fails = ", ".join(f"{e(str(k))} {v}" for k, v in sorted(r.failure_counts.items(), key=lambda kv: -kv[1])) or "none"
        safety = ", ".join(f"{e(k)} {v}" for k, v in sorted(r.safety_event_counts.items())) or "none"
        overlap = {None: "", True: "intervals overlap", False: "<b>intervals do not overlap</b>"}[r.intervals_overlap]
        videos = " ".join(f'<a href="{e(v, quote=True)}">{i + 1}</a>' for i, v in enumerate(r.videos[:20])) or "none"
        body.append(
            f"<tr><td>{e(r.task_id)}</td><td>{e(r.skill)}@{e(r.skill_version)}</td><td>{mode}</td>"
            f"<td>{_rate(r.sim_n_success, r.sim_n, r.sim_ci95)}</td><td>{_rate(r.n_success, r.n, r.ci95)}<br><small>{overlap}</small></td>"
            f"<td>{fails}</td><td>{_agree(r.estimator_vs_label)}</td><td>{_agree(r.vlm_vs_label)}</td>"
            f"<td>{safety}</td><td>{videos}</td></tr>")
    page = f"""<!doctype html><meta charset="utf-8"><title>SkillForge sim2real scoreboard</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;color:#1a1a1a;background:#fff}}table{{border-collapse:collapse}}
th,td{{border:1px solid #ccc;padding:6px 10px;text-align:left;vertical-align:top}}th{{background:#f2f2f2}}small{{color:#555}}</style>
<h1>Sim2real scoreboard</h1>
<p>Success is the operator's label on a real arm and true sim state in a dry run. Intervals are Wilson 95%.
A dry run is a simulated twin and is not a real result. Generated {e(board.created_at)}.</p>
<table><tr><th>Task</th><th>Skill</th><th>Where</th><th>Sim held-out</th><th>Result</th><th>Failure tags</th>
<th>Estimator vs label</th><th>VLM vs label</th><th>Trials with a safety event</th><th>Videos</th></tr>
{''.join(body)}</table>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return path
