"""T4: measure the parallel speedup, and envs that make the measurement honest.

StubEnv steps in microseconds, so pool start-up dominates any stub benchmark. `CostlyStubEnv`
adds a fixed amount of CPU work per step (a stand-in for a physics step) without changing
the episode, so it shows what the pool does once env.step costs something real.
`RenderStubEnv` draws a tiny top-down frame so video capture can be exercised without MuJoCo.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

from skillforge.contracts import SkillSource, TaskSpec
from skillforge.sim import StubEnv

from .harness import evaluate
from .parallel import MakeEnv, evaluate_parallel


class CostlyStubEnv(StubEnv):
    """StubEnv plus ~`work` matrix products of CPU per step. Same episodes as StubEnv."""

    work = 300

    def step(self, action):
        m = np.eye(32) * 1.0001
        for _ in range(self.work):
            m = m @ m
            m /= np.abs(m).max()
        return super().step(action)


class RenderStubEnv(StubEnv):
    """StubEnv with a 64x64 top-down render: EE in white, objects in red, regions outlined."""

    size = 64
    extent = ((0.2, 0.8), (-0.3, 0.5))  # world x, y shown

    def _px(self, x: float, y: float) -> tuple[int, int]:
        (x0, x1), (y0, y1) = self.extent
        c = int((y - y0) / (y1 - y0) * (self.size - 1))
        r = int((1 - (x - x0) / (x1 - x0)) * (self.size - 1))
        return max(0, min(self.size - 1, r)), max(0, min(self.size - 1, c))

    def render(self, camera=None):
        img = np.full((self.size, self.size, 3), 24, dtype=np.uint8)
        for reg in self.task.scene.regions:
            (r0, c0), (r1, c1) = (
                self._px(reg.center[0] + reg.half_extents[0], reg.center[1] - reg.half_extents[1]),
                self._px(reg.center[0] - reg.half_extents[0], reg.center[1] + reg.half_extents[1]),
            )
            img[r0 : r1 + 1, [c0, c1]] = (80, 160, 80)
            img[[r0, r1], c0 : c1 + 1] = (80, 160, 80)
        for p in self._obj.values():
            r, c = self._px(p[0], p[1])
            img[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2] = (220, 60, 60)
        r, c = self._px(self._ee[0], self._ee[1])
        shade = 255 if self._grip > 0.5 else 170
        img[r, max(0, c - 2) : c + 3] = shade
        img[max(0, r - 2) : r + 3, c] = shade
        return img


class BenchResult(BaseModel):
    env: str
    n_seeds: int
    workers: int
    serial_s: float
    parallel_s: float
    speedup: float
    identical: bool


def bench(
    make_env: MakeEnv, src: SkillSource, task: TaskSpec, seeds: Sequence[int], workers: int
) -> BenchResult:
    from skillforge.skills import load_skill

    t0 = time.perf_counter()
    rs, es = evaluate(make_env, load_skill(src), task, seeds)
    serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    rp, ep = evaluate_parallel(make_env, src, task, seeds, workers=workers)
    par = time.perf_counter() - t0
    strip = lambda eps: [e.model_dump(exclude={"duration_s"}) for e in eps]
    same = strip(es) == strip(ep) and rs.model_dump(exclude={"created_at"}) == rp.model_dump(exclude={"created_at"})
    return BenchResult(
        env=getattr(make_env, "__name__", str(make_env)),
        n_seeds=len(seeds),
        workers=workers,
        serial_s=round(serial, 3),
        parallel_s=round(par, 3),
        speedup=round(serial / par, 2),
        identical=same,
    )
