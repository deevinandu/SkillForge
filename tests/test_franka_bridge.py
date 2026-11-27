"""T7: the frankapy bridge, client against a real server process in --fake mode (no robot, no ROS)."""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from skillforge.real import SIM_HOME, BridgeError, FrankaBridgeRobotIO, RobotIO

SERVER = Path(__file__).parent.parent / "scripts" / "frankapy_bridge_server.py"
SPEED = 0.5  # rad/s, the server's --max-joint-speed in these tests
WATCHDOG = 0.3


@pytest.fixture
def server():
    p = subprocess.Popen([sys.executable, str(SERVER), "--fake", "--port", "0", "--max-joint-speed", str(SPEED),
                          "--watchdog", str(WATCHDOG)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    line = p.stdout.readline()
    assert "bridge listening on" in line, line
    port = int(line.split(":")[1].split()[0])
    yield port
    p.terminate()
    p.wait(5)


def connect(port, **kw):
    return FrankaBridgeRobotIO(port=port, **kw)


def stream_to(robot, q, n=1, dt=0.05, gripper=0.0):
    for _ in range(n):
        robot.send_joint_targets(np.asarray(q, dtype=float), gripper)
        time.sleep(dt)


def test_sim_home_matches_the_simulator():
    pytest.importorskip("mujoco")
    from skillforge.sim.mujoco_env import PandaProfile

    assert SIM_HOME == PandaProfile.home


def test_client_is_a_robot_io_and_home_goes_to_the_sim_home(server):
    with connect(server) as robot:
        assert isinstance(robot, RobotIO)
        assert robot.info["mode"] == "fake" and robot.info["joints"] == 7
        robot.home()
        s = robot.read_state()
        np.testing.assert_allclose(s.q, SIM_HOME, atol=1e-9)
        assert s.gripper_width == pytest.approx(0.08)


def test_targets_before_home_are_refused(server):
    with connect(server) as robot, pytest.raises(BridgeError, match="not streaming"):
        robot.send_joint_targets(np.array(SIM_HOME), 0.0)


def test_a_far_target_is_limited_to_the_joint_speed(server):
    with connect(server) as robot:
        robot.home()
        far = np.array(SIM_HOME) + [0.5, 0, 0, 0, 0, 0, 0]
        stream_to(robot, far, n=1, dt=0.0)
        s = robot.read_state()
        moved = s.q[0] - SIM_HOME[0]
        assert 0 < moved <= SPEED * 0.1 + 1e-9  # at most one (capped) control period's worth
        assert "bridge:clamped" in s.events


def test_targets_are_kept_off_the_joint_limits(server):
    with connect(server) as robot:
        robot.home()
        beyond = np.array(SIM_HOME)
        beyond[3] = 0.5  # joint 4's upper limit is -0.0698
        for _ in range(80):  # long enough to reach the limit at the speed cap
            stream_to(robot, beyond, dt=0.01)
        q4 = robot.read_state().q[3]
        assert q4 <= -0.0698 - 0.05 + 1e-9


def test_home_outside_the_joint_limits_is_refused(server):
    bad = list(SIM_HOME)
    bad[3] = 0.0
    with connect(server, home_q=bad) as robot, pytest.raises(BridgeError, match="joint limits"):
        robot.home()


def test_the_watchdog_stops_streaming_and_later_targets_fail(server):
    with connect(server) as robot:
        robot.home()
        stream_to(robot, SIM_HOME, n=2)
        time.sleep(WATCHDOG + 0.3)
        assert "bridge:watchdog" in robot.read_state().events
        with pytest.raises(BridgeError, match="not streaming"):
            robot.send_joint_targets(np.array(SIM_HOME), 0.0)
        robot.home()  # home is the way back
        stream_to(robot, SIM_HOME)


def test_the_robot_ending_the_skill_stops_streaming(server):
    with connect(server) as robot:
        robot.home()
        robot._call({"cmd": "fault", "reason": "virtual wall"})
        time.sleep(0.25)  # the server checks the skill at 10 Hz
        assert "bridge:robot_ended_skill" in robot.read_state().events
        with pytest.raises(BridgeError):
            robot.send_joint_targets(np.array(SIM_HOME), 0.0)


def test_stop_holds_until_home(server):
    with connect(server) as robot:
        robot.home()
        robot.stop()
        assert "bridge:stop" in robot.read_state().events
        with pytest.raises(BridgeError):
            robot.send_joint_targets(np.array(SIM_HOME), 0.0)


def test_gripper_closes_and_opens_on_the_crossing(server):
    with connect(server) as robot:
        robot.home()
        stream_to(robot, SIM_HOME, gripper=1.0)
        assert robot.read_state().gripper_width == pytest.approx(0.0)
        stream_to(robot, SIM_HOME, gripper=0.2)
        assert robot.read_state().gripper_width == pytest.approx(0.08)


def test_a_disconnected_client_leaves_the_arm_stopped(server):
    robot = connect(server)
    robot.home()
    robot.close()
    time.sleep(0.1)
    with connect(server) as again:
        r = again._call({"cmd": "state"})
        assert r["streaming"] is False and "bridge:client_disconnected" in r["events"]


def test_bad_messages_get_errors_not_a_dead_server(server):
    with connect(server) as robot:
        robot.home()
        with pytest.raises(ValueError):
            robot.send_joint_targets(np.zeros(6), 0.0)
        with pytest.raises(BridgeError, match="finite"):
            robot._call({"cmd": "targets", "q": [float("nan")] * 7, "gripper": 0})
        with pytest.raises(BridgeError, match="unknown"):
            robot._call({"cmd": "dance"})
        stream_to(robot, SIM_HOME)  # still serving


def test_no_server_is_a_clear_error():
    with pytest.raises(BridgeError, match="is frankapy_bridge_server.py running"):
        FrankaBridgeRobotIO(port=1)


def test_real_env_runs_over_the_bridge(server):
    """RealEnv + the sim's controller + the bridge: an ee-mode skill step reaches the fake arm."""
    pytest.importorskip("mujoco")
    from skillforge.contracts import Action, TaskSpec
    from skillforge.real import AutoOperator, MujocoController, RealEnv, SafetyLimits

    task = TaskSpec.model_validate(json.loads((Path(__file__).parent.parent / "tasks" / "franka_lift_cube.json").read_text()))
    limits = SafetyLimits()
    env = RealEnv(task, connect(server), estimator=None, controller=MujocoController(task, max_ee_speed=limits.ee_speed),
                  operator=AutoOperator(), limits=limits, realtime=True)
    try:
        obs = env.reset(0)
        np.testing.assert_allclose(obs.qpos, SIM_HOME, atol=1e-9)
        start = obs.ee_pos.copy()
        goal = start + [0.0, 0.0, -0.03]
        for _ in range(20):  # 1 s at 20 Hz
            obs, info = env.step(Action(mode="ee", ee_pos=goal, ee_quat=obs.ee_quat, gripper=0.0))
            assert not info.terminated
        assert obs.ee_pos[2] < start[2] - 0.01  # moved down through IK -> bridge -> fake arm
    finally:
        env.close()
