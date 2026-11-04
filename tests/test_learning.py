"""T5 learning: dataset engine, BC training, LearnedSkill. Everything here runs on StubEnv.

The MuJoCo end-to-end smoke test is opt-in (SKILLFORGE_SLOW=1) so `pytest -q` stays quick.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from skillforge.contracts import Action, Skill
from skillforge.learning.dataset import CollectConfig, DartNoise, Dataset, collect, collect_to
from skillforge.learning.features import (
    Layout,
    action_vector,
    chunk_targets,
    encode_actions,
    history_index,
    policy_features,
    quat_to_rot6,
    rot6_to_quat,
    state_names,
    vector_action,
)

STUB = "skillforge.sim:StubEnv"
slow = pytest.mark.skipif(not os.environ.get("SKILLFORGE_SLOW"), reason="set SKILLFORGE_SLOW=1 for slow tests")


def _cfg(**kw) -> CollectConfig:
    return CollectConfig(make_env=STUB, **kw)


# ---------------------------------------------------------------- features


def test_rot6_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        back = rot6_to_quat(quat_to_rot6(q))
        assert min(np.linalg.norm(back - q), np.linalg.norm(back + q)) < 1e-6
    down = np.array([0.0, 1.0, 0.0, 0.0])  # w = 0: the case a w-sign canonicalisation gets wrong
    assert abs(abs(np.dot(rot6_to_quat(quat_to_rot6(down)), down)) - 1) < 1e-9


def test_action_vector_round_trip():
    a = Action(ee_pos=np.array([0.1, 0.2, 0.3]), ee_quat=np.array([0.0, 1.0, 0.0, 0.0]), gripper=1.0)
    b = vector_action(action_vector(a))
    assert np.allclose(b.ee_pos, a.ee_pos) and np.allclose(b.ee_quat, a.ee_quat) and b.gripper == 1.0


def test_chunks_are_relative_to_current_ee_and_masked_at_the_end(stub_task):
    names = state_names(stub_task, 3)
    lay = Layout(names, "cube")
    T = 5
    st = np.zeros((T, len(names)))
    st[:, lay.ee_pos[0]] = np.arange(T) * 0.01  # ee moves 1 cm per step in x
    st[:, lay.ee_quat[0]] = 1.0
    st[:, lay.obj_quat[0]] = 1.0
    acts = np.zeros((T, 8))
    acts[:, 0] = st[:, lay.ee_pos[0]] + 0.002  # always 2 mm ahead of the ee
    acts[:, 3] = 1.0
    enc = encode_actions(acts, st, lay, pos_clip=0.05)
    y, m = chunk_targets(enc, st, lay, horizon=3)
    assert y.shape == (T, 3, 10) and m.shape == (T, 3)
    assert np.allclose(y[0, :, 0], [0.002, 0.012, 0.022])  # ee_{t+k} - ee_t + 2 mm
    assert m[:3].all() and not m[4, 1] and not m[3, 2]
    assert history_index(4, 3).tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3]]
    f = policy_features(st, lay)
    assert f.shape[0] == T and np.isfinite(f).all()


def test_encode_clips_far_targets(stub_task):
    names = state_names(stub_task, 3)
    lay = Layout(names, "cube")
    st = np.zeros((1, len(names)))
    acts = np.array([[1.0, 0, 0, 1, 0, 0, 0, 0]])
    enc = encode_actions(acts, st, lay, pos_clip=0.05)
    assert np.isclose(np.linalg.norm(enc[0, :3]), 0.05)


# ---------------------------------------------------------------- dataset


def test_collect_is_deterministic_and_records_the_teacher(stub_task):
    a = collect(stub_task, [3, 4], _cfg())
    b = collect(stub_task, [3, 4], _cfg())
    for (ra, xa), (rb, xb) in zip(a, b, strict=True):
        assert ra["seed"] == rb["seed"] and ra["length"] == rb["length"]
        for k in xa:
            assert np.array_equal(xa[k], xb[k]), k
    rec, arr = a[0]
    assert rec["success"] and rec["source"] == "teacher"
    T = rec["length"]
    assert arr["observation.state"].shape == (T, len(state_names(stub_task, 3)))
    assert arr["action"].shape == (T, 8) and np.array_equal(arr["action"], arr["action.executed"])
    assert arr["label_valid"].all()


def test_dart_noise_perturbs_executed_actions_only(stub_task):
    (rec, arr), = collect(stub_task, [1], _cfg(source="dart", dart=DartNoise(pos_std=0.02, p=1.0, segment=4)))
    diff = np.abs(arr["action"] - arr["action.executed"])
    assert diff[:, :3].max() > 1e-3  # positions perturbed
    assert diff[:, 3:].max() == 0  # orientation and gripper untouched


def test_dataset_layout_append_and_success_only(stub_task, tmp_path):
    root = tmp_path / "ds"
    ds, summ = collect_to(root, stub_task, range(3), _cfg())
    assert summ["kept"] == 3 and ds.info["total_episodes"] == 3
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["format"] == "skillforge.dataset/1" and info["teacher"]["version"] == "0.2.0"
    assert info["features"]["observation.state"]["names"] == ds.state_names
    assert (root / "meta" / "stats.json").exists()
    assert (root / ds.episodes[0]["file"]).exists()
    # append: a second pass lands in the same dataset with continuing indices
    ds2, _ = collect_to(root, stub_task, range(10, 12), _cfg(source="dart", dart=DartNoise()))
    assert [e["episode_index"] for e in ds2.episodes] == [0, 1, 2, 3, 4]
    assert len(ds2.info["collections"]) == 2
    assert Dataset(root).load(4)["action"].shape[1] == 8
    # success_only drops failures (a 3-step budget fails every episode)
    short = stub_task.model_copy(update={"max_steps": 3})
    _, s = collect_to(tmp_path / "short", short, range(2), _cfg(success_only=True))
    assert s["kept"] == 0 and s["success"] == 0


# ---------------------------------------------------------------- policy + skill (torch)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    pytest.importorskip("torch")
    import json as _json
    from pathlib import Path

    from skillforge.contracts import TaskSpec
    from skillforge.learning.policy import TrainConfig, train

    task = TaskSpec.model_validate(_json.loads((Path(__file__).parent.parent / "tasks" / "stub_pick_cube_to_bin.json").read_text()))
    d = tmp_path_factory.mktemp("learn")
    ds, _ = collect_to(d / "ds", task, range(6), _cfg())
    cfg = TrainConfig(epochs=1, hidden=(32, 32), horizon=4, batch_size=64, threads=1, val_frac=0.2)
    meta = train([ds], d / "pol" / "policy.pt", cfg, log=None)
    return task, ds, d / "pol" / "policy.pt", meta, cfg


def test_train_one_epoch_writes_a_self_contained_checkpoint(trained):
    from skillforge.learning.policy import LoadedPolicy

    task, ds, ckpt, meta, cfg = trained
    assert ckpt.exists() and ckpt.with_suffix(".json").exists()
    assert meta["epochs_run"] == 1 and meta["n_frames"] > 0 and meta["teacher"]["version"] == "0.2.0"
    pol = LoadedPolicy(ckpt)
    assert pol.x_mu.shape == (meta["in_dim"],) and pol.y_sd.shape == (10,)
    y = pol.predict(np.zeros(meta["in_dim"]))
    assert y.shape == (4, 10)


def test_training_is_deterministic(trained, tmp_path):
    import torch

    from skillforge.learning.policy import train

    task, ds, ckpt, meta, cfg = trained
    train([ds], tmp_path / "again.pt", cfg, log=None)
    a = torch.load(ckpt, weights_only=False)["state_dict"]
    b = torch.load(tmp_path / "again.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_learned_skill_round_trip_through_the_harness(trained):
    from skillforge.eval.harness import evaluate
    from skillforge.learning.skill import LearnedSkill, learned_source
    from skillforge.sim import StubEnv
    from skillforge.skills import load_skill

    task, ds, ckpt, meta, cfg = trained
    sk = LearnedSkill(ckpt)
    assert isinstance(sk, Skill) and sk.version.startswith("bc-")
    rep, eps = evaluate(StubEnv, sk, task, [100, 101])
    assert rep.n_rollouts == 2 and rep.skill == "learned_pick_place"
    # same checkpoint as a SkillSource (what the parallel harness ships to workers)
    src = learned_source(ckpt, ensemble_m=None)
    loaded = load_skill(src)
    assert loaded.version == sk.version and loaded.ensemble_m is None
    rep2, _ = evaluate(StubEnv, loaded, task, [100])
    rep3, _ = evaluate(StubEnv, LearnedSkill(ckpt, ensemble_m=None), task, [100])
    assert rep2.n_success == rep3.n_success  # deterministic given seed + checkpoint
    with pytest.raises(ValueError):
        sk.reset(task.model_copy(update={"id": "other_task"}))


def test_dagger_relabels_learner_rollouts(trained):
    task, ds, ckpt, meta, cfg = trained
    c = _cfg(source="dagger:r1", learner_checkpoint=str(ckpt), learner_overrides={"ensemble_m": 0.1})
    (rec, arr), = collect(task, [200], c)
    assert rec["source"] == "dagger:r1"
    # the env executed the learner, the label is the shadow teacher's
    assert arr["action"].shape == arr["action.executed"].shape
    assert not np.array_equal(arr["action"], arr["action.executed"])
    again = collect(task, [200], c)[0][1]
    assert all(np.array_equal(arr[k], again[k]) for k in arr)


def test_dagger_sync_gate_drops_out_of_phase_labels(trained):
    task, ds, ckpt, meta, cfg = trained
    base = dict(source="dagger:r1", learner_checkpoint=str(ckpt))
    (_, keep_all), = collect(task, [201], _cfg(sync_tol=None, **base))
    (_, strict), = collect(task, [201], _cfg(sync_tol=1e-6, sync_patience=0, **base))
    assert keep_all["label_valid"][0]  # without the gate only an outright teacher failure invalidates
    assert not strict["label_valid"].any()  # a 1-epoch learner never matches the teacher exactly
    assert np.array_equal(keep_all["action"], strict["action"])  # gating changes masks, not labels


def test_compare_reports_teacher_vs_learned_with_paired_counts(trained):
    from skillforge.learning.experiment import compare

    task, ds, ckpt, meta, cfg = trained
    res = compare(task, str(ckpt), range(300, 303), workers=1, stress_factor=2.0, cache_teacher=False)
    assert set(res["results"]) == {"nominal", "stress_x2"}
    for r in res["results"].values():
        assert r["teacher"]["n"] == r["learned"]["n"] == 3
        assert sum(r["paired"].values()) == 3
        lo, hi = r["teacher"]["ci95"]
        assert 0 <= lo <= r["teacher"]["rate"] <= hi <= 1
    assert isinstance(res["gate"]["met"], bool)


def test_cli_parses_every_subcommand():
    from skillforge.learning.experiment import parser

    p = parser()
    assert p.parse_args(["collect", "--name", "x", "--dart", "0.01"]).dart == 0.01
    assert p.parse_args(["train", "--data", "a", "b", "--out", "o.pt"]).data == ["a", "b"]
    assert p.parse_args(["dagger", "--data", "a", "--policy", "p.pt", "--ensemble-m", "none"]).ensemble_m is None
    assert p.parse_args(["compare", "--policy", "p.pt"]).seed_start == 1_000_000


@slow
def test_mujoco_collect_train_eval_smoke(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("torch")
    from skillforge.eval.harness import evaluate
    from skillforge.learning.experiment import load_task
    from skillforge.learning.policy import TrainConfig, train
    from skillforge.learning.skill import LearnedSkill
    from skillforge.sim import make_env

    task = load_task("franka_pick_cube_to_bin")
    ds, s = collect_to(tmp_path / "ds", task, range(4), CollectConfig(success_only=True))
    assert s["kept"] >= 1
    train([ds], tmp_path / "p.pt", TrainConfig(epochs=1, hidden=(64,), threads=1), log=None)
    rep, _ = evaluate(make_env, LearnedSkill(tmp_path / "p.pt"), task, [1_000_000])
    assert rep.n_rollouts == 1
