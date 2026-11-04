"""T4: GIF capture through env.render, degrading cleanly when an env can't render."""

import numpy as np
import pytest
from eval_helpers import RaisingRenderEnv, noisy

from skillforge.eval import evaluate
from skillforge.eval.bench import RenderStubEnv
from skillforge.eval.video import record_episode, record_selected, select_episodes
from skillforge.sim import StubEnv
from skillforge.skills import PickPlace

iio = pytest.importorskip("imageio.v3")


def test_renderable_env_writes_a_gif(tmp_path, stub_task):
    env = RenderStubEnv(stub_task)
    _, (ep,) = evaluate(RenderStubEnv, PickPlace(), stub_task, [5])
    rec = record_episode(env, PickPlace(), stub_task, 5, tmp_path / "e.gif", every=1, max_frames=40, expected=ep)
    assert rec.path and rec.n_frames == 40 and rec.reason == ""
    frames = iio.imread(rec.path, index=None)
    assert 1 < frames.shape[0] <= 40 and frames.shape[1:3] == (64, 64)  # GIF merges identical frames
    assert len({f.tobytes() for f in frames}) > 1  # the gripper actually moves
    assert rec.episode.video == rec.path
    assert rec.episode.model_dump(exclude={"duration_s", "video"}) == ep.model_dump(exclude={"duration_s", "video"})


def test_stub_env_cannot_render_and_that_is_fine(tmp_path, stub_task):
    rec = record_episode(StubEnv(stub_task), PickPlace(), stub_task, 0, tmp_path / "x.gif")
    assert rec.path is None and "returned None" in rec.reason
    assert not (tmp_path / "x.gif").exists()


def test_render_that_raises_is_skipped(tmp_path, stub_task):
    rec = record_episode(RaisingRenderEnv(stub_task), PickPlace(), stub_task, 0, tmp_path / "x.gif")
    assert rec.path is None and "no GL context" in rec.reason
    assert rec.episode is not None and rec.episode.success  # the rollout itself still ran


def test_replay_mismatch_is_reported(tmp_path, stub_task):
    _, (ep,) = evaluate(RenderStubEnv, PickPlace(), stub_task, [1])
    fake = ep.model_copy(update={"n_steps": ep.n_steps + 7})
    rec = record_episode(RenderStubEnv(stub_task), PickPlace(), stub_task, 1, tmp_path / "e.gif", expected=fake)
    assert "replay mismatch" in rec.reason


def test_select_and_record_failures_and_a_success(tmp_path, stub_task):
    task = noisy(stub_task, 0.03)
    report, eps = evaluate(RenderStubEnv, PickPlace(), task, range(20))
    chosen = select_episodes(eps)
    tags = [e.failure for e in chosen if not e.success]
    assert len(tags) == len(set(tags)) == len(report.failure_counts)
    assert sum(e.success for e in chosen) == (1 if report.n_success else 0)
    recs, eps2, rep2 = record_selected(RenderStubEnv, PickPlace(), task, eps, tmp_path, report=report, max_frames=10)
    assert all(r.path for r in recs) and len(recs) == len(chosen)
    with_video = {e.seed for e in eps2 if e.video}
    assert with_video == {e.seed for e in chosen}
    assert any(f.video for f in rep2.failures)
    assert all(f.video is None or f.seed in with_video for f in rep2.failures)


def test_record_selected_stops_early_without_rendering(tmp_path, stub_task):
    _, eps = evaluate(StubEnv, PickPlace(), noisy(stub_task, 0.03), range(10))
    recs, eps2, rep = record_selected(StubEnv, PickPlace(), stub_task, eps, tmp_path)
    assert len(recs) == 1 and recs[0].path is None and rep is None
    assert all(e.video is None for e in eps2)


def test_grayscale_and_float_frames_are_accepted(tmp_path, stub_task):
    class GrayEnv(StubEnv):
        def render(self, camera=None):
            return np.full((8, 8), 0.5 * 255, dtype=np.float32)

    rec = record_episode(GrayEnv(stub_task), PickPlace(), stub_task, 0, tmp_path / "g.gif", max_frames=3)
    assert rec.n_frames == 3 and iio.imread(rec.path, index=None).shape[1:3] == (8, 8)
