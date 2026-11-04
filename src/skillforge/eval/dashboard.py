"""T4: a single static HTML dashboard. No server, no network at view time.

Inputs: forge history (rounds of {round, source, report}, e.g. a `history.json`), robustness
reports, optional regression results and GIFs. Charts are inline SVG, GIFs are embedded as
data URIs, and every piece of agent-written text (rationales, failure details) is escaped.
"""

from __future__ import annotations

import base64
import html
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from skillforge.contracts import EvalReport, SkillSource

from .regression import RegressionResult
from .robustness import RobustnessReport

# Fixed tag -> colour slot, so a tag keeps its colour across rounds and dashboards.
TAG_ORDER = (
    "timeout", "grasp_miss", "grasp_slip", "collision",
    "object_dropped", "wrong_placement", "ik_failure", "exception",
    "terminated", "other",
)
TAG_SLOT = {t: f"var(--series-{i + 1})" for i, t in enumerate(TAG_ORDER[:8])} | {
    "terminated": "var(--muted)", "other": "var(--muted)",
}
MAX_EMBED_BYTES = 8 * 1024 * 1024


class HistoryRound(BaseModel):
    """One forge round as written to history.json (same shape as forge.ForgeRound)."""

    model_config = ConfigDict(extra="ignore")

    round: int
    source: SkillSource
    report: EvalReport


def load_history(path_or_data: str | Path | Sequence[object]) -> list[HistoryRound]:
    if isinstance(path_or_data, (str, Path)):
        data = json.loads(Path(path_or_data).read_text())
    else:
        data = [d.model_dump(mode="json") if isinstance(d, BaseModel) else d for d in path_or_data]
    if isinstance(data, dict) and "rounds" in data:
        data = data["rounds"]
    return [HistoryRound.model_validate(d) for d in data]


def write_history(history: Sequence[BaseModel], path: Path) -> Path:
    """Write forge rounds (any models with round/source/report) as history.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([h.model_dump(mode="json") for h in history], indent=2))
    return path


# ------------------------------------------------------------------ svg helpers

E = lambda s: html.escape(str(s), quote=True)
W, H = 640, 260
ML, MR, MT, MB = 48, 16, 16, 40  # margins
CW, CH = 360, 240  # robustness small multiples


def _pct(v: float) -> str:
    return f"{v:.0%}"


def _y(v: float, h: int = H) -> float:
    return MT + (1 - v) * (h - MT - MB)


def _y_axis(w: int = W, h: int = H) -> str:
    out = []
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        y = _y(v, h)
        out.append(f'<line class="gl" x1="{ML}" x2="{w - MR}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end">{_pct(v)}</text>')
    return "".join(out)


def _svg(body: str, label: str, w: int = W, h: int = H) -> str:
    return (
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{E(label)}" preserveAspectRatio="xMidYMid meet">'
        f"{body}</svg>"
    )


def _x_band(n: int) -> tuple[float, float]:
    """Band scale for n categories: (start, step)."""
    step = (W - ML - MR) / max(n, 1)
    return ML, step


def success_chart(rounds: Sequence[HistoryRound], target: float | None = None) -> str:
    if not rounds:
        return '<p class="empty">No rounds.</p>'
    x0, step = _x_band(len(rounds))
    parts = [_y_axis()]
    if target is not None:
        y = _y(target)
        parts.append(f'<line class="target" x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{W - MR}" y="{y - 6:.1f}" text-anchor="end">target {_pct(target)}</text>')
    pts = []
    for i, r in enumerate(rounds):
        cx = x0 + step * (i + 0.5)
        rep = r.report
        lo, hi = rep.ci95
        tip = f"round {r.round} · {r.source.version}: {rep.n_success}/{rep.n_rollouts} = {_pct(rep.success_rate)} (95% CI {_pct(lo)}–{_pct(hi)})"
        parts.append(f'<line class="ci" x1="{cx:.1f}" x2="{cx:.1f}" y1="{_y(lo):.1f}" y2="{_y(hi):.1f}"/>')
        for yy in (lo, hi):
            parts.append(f'<line class="ci" x1="{cx - 6:.1f}" x2="{cx + 6:.1f}" y1="{_y(yy):.1f}" y2="{_y(yy):.1f}"/>')
        pts.append((cx, _y(rep.success_rate)))
        parts.append(
            f'<circle class="dot" cx="{cx:.1f}" cy="{_y(rep.success_rate):.1f}" r="5"/>'
            f'<rect class="hit" x="{cx - step / 2:.1f}" y="{MT}" width="{step:.1f}" height="{H - MT - MB}" data-tip="{E(tip)}"/>'
            f'<text class="tick" x="{cx:.1f}" y="{H - MB + 16}" text-anchor="middle">r{r.round}</text>'
            f'<text class="tick sub" x="{cx:.1f}" y="{H - MB + 30}" text-anchor="middle">{E(r.source.version)}</text>'
        )
    if len(pts) > 1:
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.insert(1, f'<polyline class="line" points="{d}"/>')
    return _svg("".join(parts), "Success rate per round with 95% Wilson intervals")


def failure_chart(rounds: Sequence[HistoryRound]) -> str:
    """Stacked bars: share of rollouts ending in each failure tag, per round."""
    if not rounds:
        return '<p class="empty">No rounds.</p>'
    tags = [t for t in TAG_ORDER if any(r.report.failure_counts.get(t) for r in rounds)]
    tags += sorted({t for r in rounds for t in r.report.failure_counts} - set(TAG_ORDER))
    x0, step = _x_band(len(rounds))
    bw = min(48.0, step * 0.6)
    parts = [_y_axis()]
    for i, r in enumerate(rounds):
        cx = x0 + step * (i + 0.5)
        n = max(r.report.n_rollouts, 1)
        acc = 0.0
        for t in tags:
            c = r.report.failure_counts.get(t, 0)
            if not c:
                continue
            frac = c / n
            y_top, y_bot = _y(acc + frac), _y(acc)
            hgt = max(y_bot - y_top - 2, 1)  # 2px surface gap between segments
            tip = f"round {r.round} · {t}: {c}/{n} ({_pct(frac)})"
            parts.append(
                f'<rect x="{cx - bw / 2:.1f}" y="{y_top:.1f}" width="{bw:.1f}" height="{hgt:.1f}" rx="2" '
                f'style="fill:{TAG_SLOT.get(t, "var(--muted)")}" data-tip="{E(tip)}"/>'
            )
            acc += frac
        parts.append(f'<text class="tick" x="{cx:.1f}" y="{H - MB + 16}" text-anchor="middle">r{r.round}</text>')
    legend = "".join(
        f'<span class="key"><i style="background:{TAG_SLOT.get(t, "var(--muted)")}"></i>{E(t)}</span>' for t in tags
    ) or '<span class="key">no failures</span>'
    return f'<div class="legend">{legend}</div>' + _svg("".join(parts), "Failure share by tag per round")


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def curve_chart(axis: str, reports: Sequence[RobustnessReport]) -> str:
    """Success vs one axis, one line per skill version, with the Wilson band."""
    series = []
    for rep in reports:
        try:
            series.append((f"{rep.skill}@{rep.skill_version}", rep.curve(axis)))
        except KeyError:
            continue
    vals = sorted({p.value for _, c in series for p in c.points})
    if not vals:
        return ""
    lo_v, hi_v = vals[0], vals[-1]
    span = (hi_v - lo_v) or 1.0
    w, h = CW, CH
    X = lambda v: ML + (v - lo_v) / span * (w - ML - MR)
    Y = lambda v: _y(v, h)
    parts = [_y_axis(w, h)]
    for v in vals:
        parts.append(f'<text class="tick" x="{X(v):.1f}" y="{h - MB + 16}" text-anchor="middle">{_fmt(v)}</text>')
    parts.append(f'<text class="tick sub" x="{(ML + w - MR) / 2:.0f}" y="{h - 6}" text-anchor="middle">{E(axis)}</text>')
    for si, (name, c) in enumerate(series[:8]):
        col = f"var(--series-{si + 1})"
        pts = sorted(c.points, key=lambda p: p.value)
        band = [(X(p.value), Y(p.ci95[1])) for p in pts] + [(X(p.value), Y(p.ci95[0])) for p in reversed(pts)]
        parts.append(
            f'<polygon class="band" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in band)}" style="fill:{col}"/>'
        )
        parts.append(
            f'<polyline class="line" points="{" ".join(f"{X(p.value):.1f},{Y(p.success_rate):.1f}" for p in pts)}" style="stroke:{col}"/>'
        )
        for p in pts:
            fails = ", ".join(f"{k}={v}" for k, v in sorted(p.failure_counts.items(), key=lambda kv: -kv[1])) or "none"
            tip = f"{name} · {axis}={_fmt(p.value)}: {p.n_success}/{p.n} = {_pct(p.success_rate)} (CI {_pct(p.ci95[0])}–{_pct(p.ci95[1])}); failures: {fails}"
            parts.append(
                f'<circle class="dot" cx="{X(p.value):.1f}" cy="{Y(p.success_rate):.1f}" r="4" style="fill:{col}"/>'
                f'<circle class="hit" cx="{X(p.value):.1f}" cy="{Y(p.success_rate):.1f}" r="12" data-tip="{E(tip)}"/>'
            )
    legend = ""
    if len(series) > 1:
        legend = '<div class="legend">' + "".join(
            f'<span class="key"><i style="background:var(--series-{i + 1})"></i>{E(n)}</span>'
            for i, (n, _) in enumerate(series[:8])
        ) + "</div>"
    others = series[0][1].others if series else "task"
    note = "other axes at the task's ranges" if others == "task" else "other axes at nominal"
    return (
        f'<figure class="card"><figcaption><b>{E(axis)}</b> <span class="muted">({note})</span></figcaption>'
        f"{legend}{_svg(''.join(parts), f'Success vs {axis}', w, h)}</figure>"
    )


# ------------------------------------------------------------------ tables and sections


def _rounds_table(rounds: Sequence[HistoryRound]) -> str:
    rows = []
    for r in rounds:
        rep = r.report
        fails = ", ".join(f"{k}={v}" for k, v in sorted(rep.failure_counts.items(), key=lambda kv: -kv[1])) or "none"
        rows.append(
            f"<tr><td>{r.round}</td><td>{E(r.source.version)}</td><td class=num>{rep.n_success}/{rep.n_rollouts}</td>"
            f"<td class=num>{_pct(rep.success_rate)}</td><td class=num>{_pct(rep.ci95[0])}–{_pct(rep.ci95[1])}</td>"
            f"<td>{E(fails)}</td></tr>"
        )
    return (
        "<div class=tw><table><thead><tr><th>Round</th><th>Version</th><th>Success</th><th>Rate</th><th>95% CI</th>"
        f"<th>Failures</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _rationales(rounds: Sequence[HistoryRound]) -> str:
    out = []
    for r in rounds:
        s = r.source
        parent = f" · parent {E(s.parent_version)}" if s.parent_version else ""
        text = E(s.rationale) if s.rationale else '<span class="muted">(no rationale recorded)</span>'
        out.append(
            f'<details{" open" if r is rounds[-1] else ""}><summary><b>{E(s.name)}@{E(s.version)}</b>'
            f'<span class="muted"> · round {r.round}{parent}</span></summary><p class="pre">{text}</p></details>'
        )
    return "".join(out)


def _failure_samples(rounds: Sequence[HistoryRound], per_round: int = 5) -> str:
    if not rounds:
        return ""
    last = rounds[-1].report
    if not last.failures:
        return '<p class="muted">No failures in the latest round.</p>'
    rows = "".join(
        f"<tr><td class=num>{f.seed}</td><td>{E(f.tag)}</td><td class=mono>{E(f.detail)}</td>"
        f"<td class=mono>{E(', '.join(f'{k}={v:.3g}' for k, v in f.randomization.items()))}</td></tr>"
        for f in last.failures[:per_round]
    )
    return (
        f"<p class=muted>Latest round ({E(last.skill_version)}), first {min(per_round, len(last.failures))} failures:</p>"
        f"<div class=tw><table><thead><tr><th>Seed</th><th>Tag</th><th>Detail</th><th>Randomization</th></tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _stress_table(reports: Sequence[RobustnessReport]) -> str:
    rows = [
        f"<tr><td>{E(r.skill)}@{E(r.skill_version)}</td><td class=num>×{r.stress_factor:g}</td>"
        f"<td class=num>{r.stress.n_success}/{r.stress.n_rollouts}</td><td class=num>{_pct(r.stress.success_rate)}</td>"
        f"<td class=num>{_pct(r.stress.ci95[0])}–{_pct(r.stress.ci95[1])}</td></tr>"
        for r in reports
        if r.stress is not None
    ]
    if not rows:
        return ""
    return (
        "<h3>Stress preset (every range widened)</h3><div class=tw><table><thead><tr><th>Skill</th><th>Factor</th><th>Success</th>"
        f"<th>Rate</th><th>95% CI</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _regression_table(results: Sequence[RegressionResult]) -> str:
    rows = "".join(
        f"<tr><td>{'<b class=good>✓ pass</b>' if r.passed else '<b class=bad>✗ fail</b>'}</td><td>{E(r.task_id)}</td>"
        f"<td>{E(r.skill)}</td><td>{E(r.baseline_version)} → {E(r.candidate_version)}</td>"
        f"<td class=num>{_pct(r.baseline_rate)}</td><td class=num>{_pct(r.candidate_rate)}</td>"
        f"<td class=num>{_pct(r.threshold)}</td><td>{E(r.reason)}</td></tr>"
        for r in results
    )
    return (
        "<div class=tw><table><thead><tr><th>Result</th><th>Task</th><th>Skill</th><th>Versions</th><th>Baseline</th>"
        f"<th>Candidate</th><th>Threshold</th><th>Note</th></tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _embed(path: str) -> str | None:
    p = Path(path)
    if not p.is_file() or p.stat().st_size > MAX_EMBED_BYTES:
        return None
    return "data:image/gif;base64," + base64.b64encode(p.read_bytes()).decode()


def _videos(rounds: Sequence[HistoryRound], extra: Sequence[tuple[str, str]]) -> str:
    items: list[tuple[str, str]] = list(extra)
    for r in rounds:
        for f in r.report.failures:
            if f.video:
                items.append((f"round {r.round} · {r.source.version} · seed {f.seed} · {f.tag}", f.video))
    cards, seen = [], set()
    for label, path in items:
        if path in seen:
            continue
        seen.add(path)
        src = _embed(path)
        body = f'<img src="{src}" alt="{E(label)}" loading="lazy">' if src else f'<p class="muted">missing or too large: {E(path)}</p>'
        cards.append(f'<figure class="card vid">{body}<figcaption>{E(label)}</figcaption></figure>')
    if not cards:
        return '<p class="muted">No episode videos (the env could not render, or none were recorded).</p>'
    return f'<div class="grid">{"".join(cards)}</div>'


CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--good:#006300;--bad:#d03b3b;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;--series-5:#e87ba4;
--series-6:#008300;--series-7:#6250d6;--series-8:#e34948}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;
--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--good:#0ca30c;--bad:#e66767;
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181;--series-6:#008300;
--series-7:#9085e9;--series-8:#e66767}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--good:#0ca30c;--bad:#e66767;--series-1:#3987e5;--series-2:#d95926;
--series-3:#199e70;--series-4:#c98500;--series-5:#d55181;--series-6:#008300;--series-7:#9085e9;--series-8:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}h3{font-size:15px;margin:20px 0 8px}
.muted,.sub{color:var(--ink2)}.empty{color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px;margin:0 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.grid .card{margin:0}
svg{width:100%;height:auto;display:block}
svg text{font-size:11px;fill:var(--ink2);font-variant-numeric:tabular-nums}
line.gl{stroke:var(--grid);stroke-width:1}
.ci{stroke:var(--ink2);stroke-width:1.5}.target{stroke:var(--muted);stroke-dasharray:4 4}
.line{fill:none;stroke:var(--series-1);stroke-width:2}.dot{fill:var(--series-1);stroke:var(--surface);stroke-width:2}
.band{opacity:.15}.hit{fill:transparent;cursor:default}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:0 0 6px;color:var(--ink2);font-size:12px}
.key i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface);border:1px solid var(--border);border-radius:8px}
.tw{overflow-x:auto;margin:0 0 12px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:600}
.num{font-variant-numeric:tabular-nums;white-space:nowrap}.mono{font-family:ui-monospace,monospace;font-size:12px;word-break:break-word}
.good{color:var(--good)}.bad{color:var(--bad)}
details{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 12px;margin:0 0 8px}
.pre{white-space:pre-wrap;margin:8px 0 0}
.vid img{width:100%;image-rendering:pixelated;border-radius:4px;background:var(--page)}
.vid figcaption{font-size:12px;color:var(--ink2);margin-top:6px}
figcaption{margin-bottom:6px}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);
border-radius:6px;padding:6px 8px;font-size:12px;max-width:360px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;z-index:9}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}.stat{background:var(--surface);border:1px solid var(--border);
border-radius:8px;padding:10px 14px;min-width:140px}.stat b{display:block;font-size:22px}
"""

JS = """
(()=>{const t=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const el=e.target.closest('[data-tip]');
if(!el){t.style.display='none';return}t.textContent=el.dataset.tip;t.style.display='block';
const w=t.offsetWidth,h=t.offsetHeight;let x=e.clientX+12,y=e.clientY+12;
if(x+w>innerWidth-8)x=e.clientX-w-12;if(y+h>innerHeight-8)y=e.clientY-h-12;
t.style.left=Math.max(8,x)+'px';t.style.top=Math.max(8,y)+'px'});})();
"""


def render_dashboard(
    history: Sequence[HistoryRound] | Sequence[object] | str | Path = (),
    robustness: Sequence[RobustnessReport] = (),
    regression: Sequence[RegressionResult] = (),
    videos: Sequence[tuple[str, str]] = (),
    title: str = "SkillForge eval",
    target: float | None = None,
) -> str:
    """Return the dashboard as one HTML string. `videos` are extra (label, gif path) pairs."""
    if isinstance(history, (str, Path)):
        rounds = load_history(history)
    elif history and all(isinstance(h, HistoryRound) for h in history):
        rounds = list(history)
    else:
        rounds = load_history(history) if history else []
    sections = []
    if rounds:
        last = rounds[-1].report
        best = max(rounds, key=lambda r: r.report.success_rate)
        sections.append(
            '<div class="stats">'
            f'<div class="stat"><span class="muted">Latest ({E(last.skill_version)})</span><b>{_pct(last.success_rate)}</b>'
            f'<span class="muted">{last.n_success}/{last.n_rollouts}, CI {_pct(last.ci95[0])}–{_pct(last.ci95[1])}</span></div>'
            f'<div class="stat"><span class="muted">Best round</span><b>r{best.round}</b><span class="muted">{E(best.source.version)} · {_pct(best.report.success_rate)}</span></div>'
            f'<div class="stat"><span class="muted">Rounds</span><b>{len(rounds)}</b><span class="muted">task {E(last.task_id)}</span></div>'
            "</div>"
        )
        sections.append(f'<h2>Success per round</h2><div class="card">{success_chart(rounds, target)}</div>')
        sections.append(f'<h2>Failure tags per round</h2><div class="card">{failure_chart(rounds)}</div>')
        sections.append(f"<h3>Table view</h3>{_rounds_table(rounds)}")
        sections.append(f"<h3>Failure samples</h3>{_failure_samples(rounds)}")
        sections.append(f"<h2>Versions and rationale</h2>{_rationales(rounds)}")
    if robustness:
        axes = list(dict.fromkeys(c.axis for r in robustness for c in r.curves))
        cards = "".join(curve_chart(a, robustness) for a in axes)
        n = len(robustness[0].seeds)
        sections.append(
            f'<h2>Robustness</h2><p class="muted">Success vs one randomization axis at a time, {n} seeds per point, '
            f"shaded band = 95% Wilson interval.</p><div class=grid>{cards}</div>{_stress_table(robustness)}"
        )
    if regression:
        sections.append(f"<h2>Regression</h2>{_regression_table(regression)}")
    if rounds or videos:
        sections.append(f"<h2>Episodes</h2>{_videos(rounds, videos)}")
    if not sections:
        sections.append('<p class="empty">Nothing to show: no history, robustness or regression data.</p>')
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{E(title)}</title><style>{CSS}</style></head><body><main>"
        f'<h1>{E(title)}</h1><p class="muted">Generated {stamp}. Sim results only.</p>'
        f'{"".join(sections)}</main><div id="tip" role="tooltip"></div><script>{JS}</script></body></html>'
    )


def write_dashboard(path: Path, **kw: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(**kw), encoding="utf-8")
    return path
