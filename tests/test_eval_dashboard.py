"""T4: the static dashboard renders from a fixture history, self-contained and escaped."""

import re
from pathlib import Path

import pytest
from eval_helpers import pick_place_source

from skillforge.eval.dashboard import load_history, render_dashboard, write_dashboard, write_history
from skillforge.eval.robustness import sweep
from skillforge.sim import StubEnv

FIXTURE = Path(__file__).parent / "fixtures" / "history.json"


@pytest.fixture
def robustness(stub_task):
    return sweep(
        StubEnv, pick_place_source(), stub_task, axes={"obs_noise_std": [0.0, 0.01, 0.02]}, seeds=range(10), workers=1
    )


def _no_external(html: str) -> None:
    refs = re.findall(r'(?:src|href)\s*=\s*"([^"]*)"', html)
    assert all(r.startswith(("data:", "#")) for r in refs), refs
    assert "http://" not in html.replace("http://www.w3.org", "") and "https://" not in html
    assert "<link" not in html and "@import" not in html


def test_dashboard_from_fixture_history(tmp_path, robustness):
    rounds = load_history(FIXTURE)
    assert [r.source.version for r in rounds] == ["0.1.0", "0.2.0", "0.3.0"]
    p = write_dashboard(tmp_path / "index.html", history=FIXTURE, robustness=[robustness], target=0.9)
    html = p.read_text()
    _no_external(html)
    assert html.startswith("<!doctype html>") and "<title>" in html
    # one CI whisker set and one dot per round
    assert html.count('class="dot"') >= 3 and "target 90%" in html
    # failure tags from every round appear in the legend
    for tag in ("timeout", "grasp_miss", "collision", "grasp_slip", "wrong_placement"):
        assert f"</i>{tag}</span>" in html
    # agent-written text is escaped, never executed
    assert "<script>alert" not in html and "&lt;script&gt;alert" in html
    assert "<b>timeout</b>" not in html
    assert "Slow down before closing the gripper &amp; wait 6 steps." in html
    # robustness section
    assert "obs_noise_std" in html and "Stress preset" in html
    assert "No episode videos" in html
    # prefers-color-scheme dark plus an explicit theme override
    assert "prefers-color-scheme:dark" in html and ':root[data-theme="dark"]' in html


def test_dashboard_embeds_gifs(tmp_path, stub_task):
    pytest.importorskip("imageio.v3")
    from eval_helpers import noisy

    from skillforge.eval import evaluate
    from skillforge.eval.bench import RenderStubEnv
    from skillforge.eval.video import record_selected
    from skillforge.skills import PickPlace

    task = noisy(stub_task, 0.03)
    report, eps = evaluate(RenderStubEnv, PickPlace(), task, range(12))
    recs, _, report = record_selected(RenderStubEnv, PickPlace(), task, eps, tmp_path / "v", report=report, max_frames=5)
    rounds = load_history(FIXTURE)
    rounds[-1] = rounds[-1].model_copy(update={"report": report})
    hist = write_history(rounds, tmp_path / "history.json")
    extra = [("success", r.path) for r in recs if r.episode.success] + [("dup", recs[0].path)]  # dup is shown once
    html = render_dashboard(history=hist, videos=extra)
    _no_external(html)
    n_gif = html.count('src="data:image/gif;base64,')
    assert n_gif == len({r.path for r in recs if r.path})


def test_dashboard_handles_empty_and_missing_inputs(tmp_path):
    html = render_dashboard()
    assert "Nothing to show" in html
    html = render_dashboard(history=load_history(FIXTURE), videos=[("gone", str(tmp_path / "nope.gif"))])
    assert "missing or too large" in html


def test_dashboard_accepts_forge_rounds(stub_task):
    from skillforge.agents import NullImprover
    from skillforge.forge import forge
    from skillforge.sim import make_env

    hist = forge(stub_task, pick_place_source(), NullImprover(), make_env, rounds=2, rollouts=5, target=1.01)
    html = render_dashboard(history=hist)
    assert "r0" in html and "r1" in html and "5/5" in html


def test_dashboard_reads_agents_loop_history(tmp_path, stub_task):
    """history.json written by agents.loop (version/rationale, no source) renders."""
    from skillforge.eval.dashboard import load_history, render_dashboard
    from skillforge.eval.harness import evaluate
    from skillforge.sim import StubEnv
    from skillforge.skills import PickPlace

    rep, _ = evaluate(StubEnv, PickPlace(), stub_task, range(3))
    doc = {
        "schema": 1,
        "rounds": [
            {"round": 0, "version": "0.2.0", "parent_version": None, "rationale": "seed",
             "accepted": True, "report": rep.model_dump(mode="json")},
            {"round": 1, "version": "0.2.1", "parent_version": "0.2.0", "rationale": "tweak tol",
             "accepted": True, "report": rep.model_dump(mode="json")},
        ],
    }
    p = tmp_path / "history.json"
    p.write_text(__import__("json").dumps(doc))
    rounds = load_history(p)
    assert [r.source.version for r in rounds] == ["0.2.0", "0.2.1"]
    assert rounds[1].source.rationale == "tweak tol"
    assert "0.2.1" in render_dashboard(history=p)
