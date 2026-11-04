"""T4: robustness sweeps and the stress preset."""

import math

import pytest
from eval_helpers import pick_place_source

from skillforge.contracts import RandomizationSpec, Range
from skillforge.eval import EvalPool
from skillforge.eval.robustness import (
    AXES,
    DEFAULT_GRID,
    RobustnessReport,
    pin_axis,
    stress_spec,
    sweep,
)
from skillforge.sim import StubEnv


def test_pin_axis_keeps_or_resets_other_axes(stub_task):
    t = pin_axis(stub_task, "obs_noise_std", 0.006)
    assert t.scene.randomization.obs_noise_std == Range.fixed(0.006)
    assert t.scene.randomization.object_xy_jitter == stub_task.scene.randomization.object_xy_jitter
    t2 = pin_axis(stub_task, "obs_noise_std", 0.006, others="nominal")
    assert t2.scene.randomization.object_xy_jitter == Range.fixed(0.0)
    assert stub_task.scene.randomization.obs_noise_std == Range(lo=0.0, hi=0.004)  # original untouched
    with pytest.raises(KeyError):
        pin_axis(stub_task, "gravity", 1.0)


def test_default_grid_covers_every_axis():
    assert set(DEFAULT_GRID) == set(AXES)
    assert all(len(v) == 6 for v in DEFAULT_GRID.values())
    assert DEFAULT_GRID["obs_noise_std"][0] == 0.0 and math.isclose(DEFAULT_GRID["obs_noise_std"][-1], 0.01)


def test_stress_widens_around_nominal_and_clamps():
    rnd = RandomizationSpec(
        object_xy_jitter=Range(lo=0.0, hi=0.05),
        friction_scale=Range(lo=0.6, hi=1.2),
        mass_scale=Range(lo=1.2, hi=1.6),  # entirely above nominal: push the far end out
        obs_noise_std=Range(lo=0.001, hi=0.004),
    )
    s = stress_spec(rnd, 2.0)
    assert s.object_xy_jitter == Range(lo=0.0, hi=0.1)
    assert math.isclose(s.friction_scale.lo, 0.2) and math.isclose(s.friction_scale.hi, 1.4)
    assert math.isclose(s.mass_scale.lo, 1.2) and math.isclose(s.mass_scale.hi, 2.2)
    assert s.obs_noise_std.lo == 0.001 and math.isclose(s.obs_noise_std.hi, 0.008)
    assert s.actuator_gain_scale == Range.fixed(1.0)  # fixed at nominal stays fixed
    big = stress_spec(RandomizationSpec(friction_scale=Range(lo=0.5, hi=1.0)), 3.0)
    assert big.friction_scale.lo == 0.05  # clamped: friction can't go non-positive
    with pytest.raises(ValueError):
        stress_spec(rnd, 0.5)


def test_sweep_obs_noise_curve_falls_off(stub_task):
    vals = [0.0, 0.01, 0.03]
    rep = sweep(StubEnv, pick_place_source(), stub_task, axes={"obs_noise_std": vals}, seeds=range(30), workers=1)
    c = rep.curve("obs_noise_std")
    assert [p.value for p in c.points] == vals
    rates = [p.success_rate for p in c.points]
    assert rates[0] == 1.0 and rates[-1] < rates[0]
    for p in c.points:
        assert p.n == 30 and p.ci95[0] <= p.success_rate <= p.ci95[1]
        assert sum(p.failure_counts.values()) == p.n - p.n_success
    assert c.breaking_point(0.9) in (0.01, 0.03)
    assert rep.stress is not None and rep.stress_factor == 2.0
    again = RobustnessReport.model_validate_json(rep.model_dump_json())
    assert again.curve("obs_noise_std") == c
    assert "obs_noise_std" in rep.summary()


def test_sweep_parallel_matches_serial(stub_task):
    kw = dict(axes={"obs_noise_std": [0.0, 0.02], "action_latency_steps": [0, 4]}, seeds=range(16), stress_factor=None)
    a = sweep(StubEnv, pick_place_source(), stub_task, workers=1, **kw)
    with EvalPool(2) as pool:
        b = sweep(StubEnv, pick_place_source(), stub_task, pool=pool, **kw)
    assert a.curves == b.curves and b.stress is None
