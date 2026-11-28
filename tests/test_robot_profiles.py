"""Robot profiles (sim/robots.py): lookup by RobotSpec.source, and MujocoEnv reading only the profile."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from skillforge.contracts import Action, TaskSpec
from skillforge.sim.robots import PandaProfile, profile_for, register_profile, registered_sources

TASKS = Path(__file__).resolve().parents[1] / "tasks"


def _task(tid: str = "franka_pick_cube_to_bin") -> TaskSpec:
    return TaskSpec.model_validate_json((TASKS / f"{tid}.json").read_text())


def test_panda_is_registered_and_is_the_fallback():
    assert "menagerie:franka_emika_panda" in registered_sources()
    assert isinstance(profile_for("menagerie:franka_emika_panda"), PandaProfile)
    # a source nobody registered (e.g. a local copy of panda.xml) still gets the Panda's names
    assert isinstance(profile_for("mjcf:/somewhere/panda.xml"), PandaProfile)


def test_register_profile_maps_a_source():
    @dataclasses.dataclass(frozen=True)
    class Other(PandaProfile):
        name: str = "other"

    register_profile("mjcf:/tmp/other.xml", Other)
    try:
        assert profile_for("mjcf:/tmp/other.xml").name == "other"
    finally:
        from skillforge.sim import robots

        robots._REGISTRY.pop("mjcf:/tmp/other.xml")


def test_gripper_ctrl_default_maps_open_and_closed():
    lo_closed, hi_open = 0.0, 255.0
    p = PandaProfile()
    assert p.gripper_ctrl(0.0, np.array([lo_closed, hi_open])) == hi_open
    assert p.gripper_ctrl(1.0, np.array([lo_closed, hi_open])) == lo_closed


mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def env_cls():
    from skillforge.sim.assets import AssetError
    from skillforge.sim.mujoco_env import MujocoEnv

    try:
        MujocoEnv(_task())
    except AssetError as e:
        pytest.skip(f"menagerie Panda not available: {e}")
    return MujocoEnv


def test_env_reads_names_and_home_from_the_profile(env_cls):
    env = env_cls(_task())
    p = env.profile
    assert isinstance(p, PandaProfile)
    np.testing.assert_allclose(env.reset(0).qpos, p.home, atol=1e-9)
    # open at reset: the measured width is the task's max width
    assert env._gripper_width() == pytest.approx(env.task.scene.robot.max_gripper_width, abs=1e-6)
    tcp = env.model.site(env.task.scene.robot.ee_site)
    assert tcp.pos[2] == pytest.approx(p.tcp_offset)


def test_a_profile_passed_in_overrides_the_registry(env_cls):
    home = (0.0, 0.0, 0.0, -1.7, 0.0, 1.7, 0.7853)
    env = env_cls(_task(), profile=dataclasses.replace(PandaProfile(), home=home))
    np.testing.assert_allclose(env.reset(3).qpos, home, atol=1e-9)
    obs, _ = env.step(Action(mode="joint", joint_targets=np.array(home), gripper=1.0))
    assert obs.gripper_width < env.task.scene.robot.max_gripper_width  # closing


def test_a_model_without_the_profile_names_is_refused(env_cls):
    bad = dataclasses.replace(PandaProfile(), hand_body="no_such_hand")
    with pytest.raises(NotImplementedError, match="no_such_hand"):
        env_cls(_task(), profile=bad)
