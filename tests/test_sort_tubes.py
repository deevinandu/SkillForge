"""T2: sort tubes into racks by the cap colour the wrist camera sees (colours shuffled per seed)."""

from pathlib import Path

import numpy as np
import pytest

from skillforge.contracts import Skill, TaskSpec
from skillforge.skills import SortTubes
from skillforge.skills.sort_tubes import BACKGROUND_RGB, REFERENCE_RGB, classify

TASKS = Path(__file__).resolve().parents[1] / "tasks"


def load(name: str) -> TaskSpec:
    return TaskSpec.model_validate_json((TASKS / f"{name}.json").read_text())


def test_classify_reads_colours_and_rejects_the_table():
    cands = ["red", "blue"]
    assert classify(np.array(REFERENCE_RGB["red"]), cands) == "red"
    assert classify(np.array(REFERENCE_RGB["blue"]), cands) == "blue"
    assert classify(np.array(REFERENCE_RGB["blue"]) * 0.8, cands) == "blue"  # darker, same hue
    for bg in BACKGROUND_RGB:
        assert classify(np.array(bg), cands) is None  # no tube under the camera: nothing to read


def test_task_shuffles_colours_and_asks_for_the_wrist_camera():
    t = load("franka_sort_tubes_by_color")
    assert t.scene.randomization.shuffle_colors and "wrist" in t.scene.obs_cameras
    tubes = [o for o in t.scene.objects if not o.fixed]
    assert sorted(o.color for o in tubes) == ["blue", "blue", "red", "red"]
    assert all(o.color == "grey" for o in t.scene.objects if o.fixed)  # racks never look like a cap


def test_skill_setup_and_slots():
    t = load("franka_sort_tubes_by_color")
    sk = SortTubes()
    sk.reset(t)
    assert isinstance(sk, Skill)
    assert set(sk.rack_of) == {"red", "blue"}
    for slots in sk.free.values():
        assert len(slots) == 2 and abs(slots[1][1] - slots[0][1]) == pytest.approx(sk.slot_pitch)
    with pytest.raises(ValueError):
        SortTubes().reset(load("franka_fill_tube_rack"))  # no color_in_region conditions
    no_cam = t.model_copy(update={"scene": t.scene.model_copy(update={"obs_cameras": ()})})
    with pytest.raises(ValueError, match="wrist"):
        SortTubes().reset(no_cam)


@pytest.fixture(scope="module")
def mj():
    from skillforge.sim import default_mujoco_gl

    default_mujoco_gl()
    pytest.importorskip("mujoco")
    from skillforge.sim.mujoco_env import MujocoEnv

    return MujocoEnv


def test_sorts_by_what_it_sees(mj):
    from skillforge.eval.harness import run_episode

    t = load("franka_sort_tubes_by_color")
    env = mj(t)
    try:
        if env.render("wrist") is None:
            pytest.skip("no GL context: the skill cannot see")
        drawn = []
        for seed in (0, 1):
            sk = SortTubes()
            ep = run_episode(env, sk, t, seed)
            assert sk.seen == env._colors  # every colour read from the image, as drawn this episode
            assert ep.success, (seed, ep.failure, sk.log)
            drawn.append(dict(env._colors))
        assert drawn[0] != drawn[1]  # the shuffle changed which tube is which colour
    finally:
        env.close()
