"""Test fixture: a StubEnv that misbehaves like a real gripper sometimes does.

StubEnv never misses or drops anything, so it cannot exercise retry and recovery. This subclass
injects, from a generator seeded by the episode seed (same seed, same faults):
- grasp misses: a close that would grasp the object fails with probability `p_miss`;
- slips: a held object is dropped with probability `p_slip` per grasp, at a random step
  between `slip_after` steps after grasping; it lands at its resting height below the EE;
- extra observation noise (`extra_noise`, metres std) on object positions;
- `max_misses` / `max_slips` cap how many faults one episode gets (None = no cap);
- `report_yaw=True`: observed object quats carry the sampled `object_yaw_jitter` as a yaw
  (StubEnv records the yaw sample but always reports the identity quat);
- an honest gripper width (`honest_width=True`): while holding, the reported width is the
  object's width instead of StubEnv's 0, as a physical parallel gripper would report.

Success is still judged on TRUE positions by StubEnv.
"""

from __future__ import annotations

import functools

import numpy as np

from skillforge.sim import StubEnv


class PerturbedStubEnv(StubEnv):
    def __init__(
        self,
        task,
        p_miss: float = 0.0,
        p_slip: float = 0.0,
        slip_after: tuple[int, int] = (2, 20),
        extra_noise: float = 0.0,
        honest_width: bool = False,
        max_misses: int | None = None,
        max_slips: int | None = None,
        report_yaw: bool = False,
    ):
        self.max_misses, self.max_slips, self.report_yaw = max_misses, max_slips, report_yaw
        self.p_miss, self.p_slip, self.slip_after = p_miss, p_slip, slip_after
        self.extra_noise, self.honest_width = extra_noise, honest_width
        super().__init__(task)

    @classmethod
    def factory(cls, **kw):
        """A `make_env(task)` callable for `evaluate`."""
        return functools.partial(cls, **kw)

    def reset(self, seed: int):
        # Own generator so StubEnv's draws (jitter, noise, latency) stay identical to the plain stub.
        self._prng = np.random.default_rng([seed, 0x5F1])
        self._slip_at: int | None = None
        self._held_steps = 0
        self.n_misses = self.n_slips = 0
        return super().reset(seed)

    def _apply(self, act):
        events = super()._apply(act)
        grabbed = [e for e in events if e.startswith("grasp:")]
        if grabbed:
            miss_ok = self.max_misses is None or self.n_misses < self.max_misses
            slip_ok = self.max_slips is None or self.n_slips < self.max_slips
            if self._prng.random() < self.p_miss and miss_ok:
                self._held = None
                events = [e for e in events if not e.startswith("grasp:")] + ["grasp_miss"]
                self.n_misses += 1
            else:
                self._held_steps = 0
                lo, hi = self.slip_after
                self._slip_at = int(self._prng.integers(lo, hi + 1)) if self._prng.random() < self.p_slip and slip_ok else None
        if self._held is not None and not grabbed:
            self._held_steps += 1
            if self._slip_at is not None and self._held_steps >= self._slip_at:
                name = self._held
                self._held, self._slip_at = None, None
                self._obj[name][2] = self._start[name][2]
                events.append("grasp_slip")
                self.n_slips += 1
        return events

    def _observe(self):
        obs = super()._observe()
        if self.extra_noise > 0:
            for o in obs.objects.values():
                o.pos = o.pos + self._prng.normal(0, self.extra_noise, 3)
        if self.report_yaw:
            yaw = self._sample["object_yaw_jitter"]
            for o in obs.objects.values():
                o.quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        if self.honest_width and self._held is not None and self._grip > 0.5:
            spec = self.task.scene.object(self._held)
            obs.gripper_width = float(2 * min(spec.size[0], spec.size[1]))
        return obs
