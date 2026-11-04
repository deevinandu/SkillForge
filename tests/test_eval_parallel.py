"""T4: parallel eval gives exactly the serial result for the same seeds."""

import pytest
from eval_helpers import make_stub, noisy, pick_place_source

from skillforge.eval import EvalPool, evaluate, evaluate_parallel
from skillforge.eval.parallel import _chunks
from skillforge.sim import StubEnv, make_env
from skillforge.skills import load_skill


def _strip(eps):
    return [e.model_dump(exclude={"duration_s"}) for e in eps]


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_parallel_equals_serial(stub_task, workers):
    task = noisy(stub_task)
    src = pick_place_source()
    seeds = list(range(37))  # uneven split across chunks on purpose
    rs, es = evaluate(StubEnv, load_skill(src), task, seeds)
    rp, ep = evaluate_parallel(StubEnv, src, task, seeds, workers=workers)
    assert rs.n_success < len(seeds), "fixture should produce some failures"
    assert _strip(es) == _strip(ep)
    assert rs.model_dump(exclude={"created_at"}) == rp.model_dump(exclude={"created_at"})


def test_parallel_keeps_seed_order_and_is_repeatable(stub_task):
    task, src = noisy(stub_task), pick_place_source()
    seeds = [17, 3, 99, 3, 42, 0, 8]
    _, a = evaluate_parallel(make_env, src, task, seeds, workers=3)
    _, b = evaluate_parallel(make_env, src, task, seeds, workers=2)
    assert [e.seed for e in a] == seeds
    assert _strip(a) == _strip(b)
    assert _strip([a[1]]) == _strip([a[3]])  # same seed, same episode


def test_pool_reuse_across_tasks_and_skills(stub_task):
    src_a, src_b = pick_place_source(), pick_place_source("0.2.0", tol=0.02)
    t1, t2 = stub_task, noisy(stub_task)
    with EvalPool(2) as pool:
        got = [pool.evaluate(StubEnv, s, t, range(12))[0] for t in (t1, t2) for s in (src_a, src_b)]
    want = [evaluate(StubEnv, load_skill(s), t, range(12))[0] for t in (t1, t2) for s in (src_a, src_b)]
    strip = lambda rs: [r.model_dump(exclude={"created_at"}) for r in rs]
    assert strip(got) == strip(want)
    assert {r.skill_version for r in got} == {"0.1.0", "0.2.0"}


def test_crashing_skill_in_workers_is_a_failure(stub_task):
    src = pick_place_source()
    boom = src.model_copy(update={"code": src.code + "\nPickPlace.act = lambda self, obs: 1 / 0\n"})
    report, _ = evaluate_parallel(StubEnv, boom, stub_task, range(6), workers=2)
    assert report.failure_counts == {"exception": 6}


def test_unpicklable_make_env_is_a_clear_error(stub_task):
    with pytest.raises(TypeError, match="module-level"):
        evaluate_parallel(lambda t: StubEnv(t), pick_place_source(), stub_task, range(4), workers=2)


def test_serial_mode_accepts_any_callable(stub_task):
    report, _ = evaluate_parallel(lambda t: StubEnv(t), pick_place_source(), stub_task, range(4), workers=1)
    assert report.n_rollouts == 4


def test_module_level_function_env_and_empty_seeds(stub_task):
    report, eps = evaluate_parallel(make_stub, pick_place_source(), stub_task, [], workers=2)
    assert eps == [] and report.n_rollouts == 0 and report.skill == "pick_place"


def test_chunks_cover_seeds_in_order():
    seeds = list(range(10))
    for n in (1, 3, 4, 10, 50):
        parts = _chunks(seeds, n)
        assert [s for p in parts for s in p] == seeds
        assert max(map(len, parts)) - min(map(len, parts)) <= 1
