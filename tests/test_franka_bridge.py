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


def test_a_pause_after_home_holds_and_streaming_resumes(server):
    """2026-11-27 on the arm: a prompt between home and the first target let the watchdog end
    the skill, and every later target was refused. A pause must hold, not stop."""
    with connect(server) as robot:
        robot.home()
        time.sleep(WATCHDOG + 0.3)  # nobody sends anything (an operator prompt)
        s = robot.read_state()
        assert "bridge:watchdog_hold" in s.events
        np.testing.assert_allclose(s.q, SIM_HOME, atol=1e-9)  # held
        stream_to(robot, SIM_HOME, n=3)  # accepted again
        assert robot._call({"cmd": "state"})["streaming"] is True


def test_the_first_target_after_a_hold_cannot_jump(server):
    with connect(server) as robot:
        robot.home()
        stream_to(robot, SIM_HOME, n=2)
        time.sleep(WATCHDOG + 0.3)
        far = np.array(SIM_HOME) + [0.5, 0, 0, 0, 0, 0, 0]
        stream_to(robot, far, n=1, dt=0.0)
        moved = robot.read_state().q[0] - SIM_HOME[0]
        assert 0 < moved <= SPEED * 0.05 + 1e-9  # one control period's worth from the measured joints


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


def test_frankapy_args_are_plain_floats():
    """frankapy asserts `type(run_time) is float`: a numpy.float64 duration failed on the arm
    ("Incorrect time type") on 2026-11-27. The server converts everything before calling it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("bridge_server", SERVER)
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    now = np.array([0.418, 0.529, -0.476, -1.786, 0.388, 2.212, -0.433])  # the pose of the failed run
    duration = max(4.0, np.abs(np.array(SIM_HOME) - now).max() / 0.3)
    assert type(duration) is not float  # the bug's input: numpy.float64
    joints, d = srv.frankapy_joints_args(np.array(SIM_HOME), duration)
    assert type(d) is float and type(joints) is list and all(type(x) is float for x in joints)


def _server_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bridge_server", SERVER)
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    return srv


PARKED = [0.418, 0.529, -0.476, -1.786, 0.388, 2.212, -0.433]  # 2026-11-27: flange 9.99 cm from the front wall


def test_wall_check_matches_the_robot_and_clears_sim_home():
    srv = _server_module()
    problem = srv.wall_problem(PARKED)
    assert "flange" in problem and "front wall" in problem and "9.9 cm" in problem  # the robot logged 0.0999721 m
    assert srv.wall_problem(list(SIM_HOME)) is None


def test_server_kinematics_match_mujoco():
    mujoco = pytest.importorskip("mujoco")
    from skillforge.contracts import TaskSpec
    from skillforge.sim import make_env

    srv = _server_module()
    task = TaskSpec.model_validate(json.loads((Path(__file__).parent.parent / "tasks" / "franka_lift_cube.json").read_text()))
    env = make_env(task)
    env.reset(0)
    m, d = env.model, env.data
    rng = np.random.default_rng(0)
    try:
        for _ in range(20):
            q = np.array(srv.Q_MIN) + rng.random(7) * (np.array(srv.Q_MAX) - np.array(srv.Q_MIN))
            d.qpos[:7] = q
            mujoco.mj_kinematics(m, d)
            base = d.xpos[m.body("link0").id]
            got = np.array(srv.frame_positions(list(q)))
            want = np.array([d.xpos[m.body(n).id] - base for n in [f"link{i}" for i in range(1, 8)] + ["hand"]])
            np.testing.assert_allclose(got, want, atol=1e-6)
    finally:
        env.close()


def test_home_from_inside_a_wall_margin_is_refused_before_moving(server):
    with connect(server) as robot:
        robot.home()
        stream_to(robot, PARKED, n=1, dt=0.0)  # clamped: walk there at the speed cap
        for _ in range(120):
            stream_to(robot, PARKED, dt=0.03)
        before = robot.read_state().q
        assert np.abs(before - PARKED).max() < 1e-6
        robot.stop()
        with pytest.raises(BridgeError, match="flange is 9.9 cm from the front wall"):
            robot.home()
        np.testing.assert_allclose(robot.read_state().q, before)  # nothing moved


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


def test_one_status_flicker_does_not_end_streaming():
    """2026-11-27, skill 12: franka-interface's status flickered, the ROS action was aborted and the
    old check ended streaming while the robot kept executing. Now only the robot's own mode counts,
    and it must be out of 'executing' for 3 checks in a row."""
    srv = _server_module()
    arm = object.__new__(srv.FrankapyArm)  # no ROS: drive the state by hand
    arm._not_moving, arm.last_done_reason = 0, ""

    def at(mode):
        arm.mode, arm.mode_t = mode, time.time()
        return arm.skill_done()

    assert not at(2) and not at(1) and not at(2)  # a one-sample blip
    assert not at(1) and not at(1)
    assert at(1) and "robot_mode 1" in arm.last_done_reason  # third in a row: really idle
    arm._not_moving = 0
    arm.mode, arm.mode_t = 2, time.time() - 2.0  # state messages stopped
    assert arm.skill_done() and "no robot_state" in arm.last_done_reason


def test_server_survives_a_process_wide_socket_timeout(tmp_path):
    """rospy sets socket.setdefaulttimeout(); the idle server died in accept() after it."""
    code = ("import socket, runpy, sys; socket.setdefaulttimeout(0.3); "
            f"sys.argv = ['srv', '--fake', '--port', '0']; runpy.run_path({str(SERVER)!r}, run_name='__main__')")
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        port = int(p.stdout.readline().split(":")[1].split()[0])
        time.sleep(1.0)  # idle well past the default timeout
        assert p.poll() is None, "server died while idle"
        with connect(port) as robot:
            robot.home()
            time.sleep(0.8)  # a client pausing past the timeout, as at a prompt
            stream_to(robot, SIM_HOME)
    finally:
        p.terminate()
        p.wait(5)
