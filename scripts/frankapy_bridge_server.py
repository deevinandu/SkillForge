#!/usr/bin/env python3
"""Bridge: SkillForge's `RobotIO` (Python >= 3.10) to a Franka through frankapy (Python 3.8).

frankapy and its ROS 1 stack only run on the lab's Python 3.8; SkillForge needs 3.10. This
server runs next to frankapy and speaks newline-delimited JSON over a local TCP socket;
`skillforge.real.franka_bridge.FrankaBridgeRobotIO` is the client. Standalone on purpose: it
imports nothing from skillforge and nothing newer than Python 3.8 (stdlib, plus frankapy
unless --fake).

    # after start_control_pc.sh, robot blue, a person at the e-stop:
    source ~/Documents/frankapy/catkin_ws/devel/setup.bash
    PYTHONPATH=$HOME/Documents/frankapy:$PYTHONPATH /usr/bin/python3 scripts/frankapy_bridge_server.py
    # no robot at all (tests, protocol checks):
    python3 scripts/frankapy_bridge_server.py --fake

How the arm is driven: `home` moves to the given joints with a normal (blocking) frankapy
skill, opens the gripper, then starts ONE long dynamic joint-impedance skill
(`goto_joints(dynamic=True)`, as frankapy's examples/run_dynamic_joints.py). Each `targets`
message is published to that skill as a JointPositionSensorMessage. `stop` ends the skill;
the arm holds where it is until the next `home`.

Safety here, independent of the client (the client's RealEnv has its own limits too):
- joint targets are clamped to the joint limits (with a margin) and to `--max-joint-speed`
  relative to the last target sent;
- if no target arrives for `--watchdog` seconds the arm HOLDS its last target (the skill keeps
  running), and the next target is speed-limited from the measured joints;
- if the robot ends the skill by itself (virtual wall, reflex, e-stop), streaming stops and
  every later `targets` is refused with an error until `home`, so the client fails loudly;
- the socket listens on 127.0.0.1 only, and serves one client at a time.
franka-interface's own virtual walls and libfranka's reflexes still apply on top.

Protocol (one JSON object per line each way; every reply has "ok"):
  {"cmd": "hello"}                       -> {"ok", "mode", "joints": 7, "limits": {...}}
  {"cmd": "state"}                       -> {"ok", "t", "q", "qvel", "gripper_width", "ee_force", "events", "streaming"}
  {"cmd": "targets", "q": [7], "gripper": g}   (g: 0 open .. 1 closed) -> {"ok"} or {"ok": false, "error"}
  {"cmd": "stop"}                        -> {"ok"}
  {"cmd": "home", "q": [7]}              -> {"ok"} once the arm is there and streaming (blocks)
  {"cmd": "fault", "reason": "..."}      (--fake only: pretend the robot ended the skill)
"""

# ruff: noqa: B905 - runs on the lab's Python 3.8, which has no zip(strict=)
import argparse
import json
import math
import socket
import threading
import time

# Panda joint limits (rad), from the Franka datasheet; frankapy's FrankaConstants has the same.
Q_MIN = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
Q_MAX = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
MAX_WIDTH = 0.08
MODE_MOVE = 2  # libfranka RobotMode kMove: executing a motion (a skill)


# franka-interface's virtual walls (control PC, termination_handler.h): plane normal, point on
# plane, name; and the clearance each checked frame needs (Joint2..Joint7, then the flange).
# A skill stops as soon as any of these frames is closer than its threshold, including at its
# first control step, so a pose inside the margin cannot start any skill.
WALLS = [((1, 0, 0), (0.75, 0, 0), "front wall (x = 0.75 m)"),
         ((0, 1, 0), (0, 0.47, 0), "left wall (y = +0.47 m)"),
         ((0, 1, 0), (0, -0.47, 0), "right wall (y = -0.47 m)"),
         ((0, 0, 1), (0, 0, -0.015), "table (z = -0.015 m)"),
         ((0, 0, 1), (0, 0, 1.25), "ceiling (z = 1.25 m)"),
         ((1, 0, 0), (-0.46, 0, 0), "back wall (x = -0.46 m)")]
WALL_FRAMES = ["joint2", "joint3", "joint4", "joint5", "joint6", "joint7 (wrist)", "flange"]
WALL_CLEARANCE = [0.11, 0.11, 0.08, 0.08, 0.07, 0.07, 0.10]

# Panda kinematics, modified DH (Craig), as in Franka's documentation: a, d, alpha per joint.
_DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2), (0.0825, 0.0, math.pi / 2),
       (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2), (0.088, 0.0, math.pi / 2)]
_FLANGE_D = 0.107


def _matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def frame_positions(q):
    """Origins of joint frames 1..7 and the flange in the base frame, pure Python."""
    T = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    out = []
    for (a, d, alpha), th in zip(_DH, q):
        ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(th), math.sin(th)
        T = _matmul(T, [[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])
        out.append((T[0][3], T[1][3], T[2][3]))
    T = _matmul(T, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, _FLANGE_D], [0, 0, 0, 1]])
    out.append((T[0][3], T[1][3], T[2][3]))
    return out  # joint1..joint7, flange


def wall_problem(q):
    """None if every checked frame clears every wall as franka-interface requires, else a sentence."""
    pos = frame_positions(q)[1:]  # franka-interface checks Joint2..Flange
    for name, p, need in zip(WALL_FRAMES, pos, WALL_CLEARANCE):
        for n, p0, wall in WALLS:
            dist = abs(sum(ni * (pi - p0i) for ni, pi, p0i in zip(n, p, p0)))
            if dist < need:
                return f"the {name} is {dist * 100:.1f} cm from the {wall}; it must stay {need * 100:.0f} cm away"
    return None


def frankapy_joints_args(joints, duration):
    """`joints` and `duration` as frankapy's skill_list accepts them, checked the same way.

    frankapy asserts `type(run_time) is float or int` and `type(joints) is list`, so a
    numpy.float64 duration (e.g. from arithmetic on `get_joints()`) is rejected with
    "Incorrect time type". The fake arm calls this too, so the tests catch such a value."""
    joints = [float(x) for x in joints]
    duration = float(duration)
    assert type(duration) is float and type(joints) is list and len(joints) == 7
    assert all(type(x) is float and math.isfinite(x) for x in joints)
    return joints, duration


def clamp_targets(q, last, dt, max_speed, margin):
    """`q` clamped to the joint limits (shrunk by `margin`) and to `max_speed` rad/s from `last`."""
    out, clipped = [], False
    step = max_speed * max(dt, 1e-3)
    for i, (qi, li) in enumerate(zip(q, last)):
        lo, hi = Q_MIN[i] + margin, Q_MAX[i] - margin
        v = min(max(qi, li - step), li + step)
        v = min(max(v, lo), hi)
        clipped = clipped or abs(v - qi) > 1e-9
        out.append(v)
    return out, clipped


class FakeArm:
    """No robot: joints go straight to each (already clamped) target. For tests."""

    def __init__(self):
        self.q = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
        self.qvel = [0.0] * 7
        self.width = MAX_WIDTH
        self.skill_running = False
        self.fault = None

    def _get_joints(self):
        """Like frankapy's get_joints(): a numpy array when numpy is installed, so arithmetic on
        it gives numpy.float64, which frankapy's type checks reject."""
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            return list(self.q)
        return np.array(self.q)

    def goto_joints_blocking(self, q):
        now = self._get_joints()
        q, _ = frankapy_joints_args(q, max(4.0, max(abs(a - b) for a, b in zip(q, now)) / 0.3))
        self.q = q
        self.qvel = [0.0] * 7

    def open_gripper(self):
        self.width = MAX_WIDTH

    def close_gripper(self):
        self.width = 0.0

    def start_stream(self):
        frankapy_joints_args(self._get_joints(), 3600.0)
        self.skill_running, self.fault = True, None

    def publish(self, q, dt):
        self.qvel = [(a - b) / max(dt, 1e-3) for a, b in zip(q, self.q)]
        self.q = list(q)

    def stop(self):
        self.skill_running = False
        self.qvel = [0.0] * 7

    def skill_done(self):
        return not self.skill_running

    def state(self):
        return list(self.q), list(self.qvel), self.width, None


class FrankapyArm:
    """The real arm through frankapy. Every call happens under the server's lock."""

    def __init__(self, k_gains=None, session_s=3600.0, robot_state_topic="/robot_state_publisher_node_1/robot_state",
                 gripper_topic="/franka_gripper_1/joint_states"):
        import rospy
        from franka_interface_msgs.msg import SensorDataGroup
        from frankapy import FrankaArm, SensorDataMessageType
        from frankapy import FrankaConstants as FC
        from frankapy.proto import JointPositionSensorMessage
        from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg

        self._rospy, self._FC, self._SDT = rospy, FC, SensorDataMessageType
        self._JPSM, self._make, self._p2r = JointPositionSensorMessage, make_sensor_group_msg, sensor_proto2ros_msg
        from franka_interface_msgs.msg import RobotState
        from frankapy.proto import ShouldTerminateSensorMessage

        self._STSM = ShouldTerminateSensorMessage
        self.fa = FrankaArm()  # calls rospy.init_node
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
        self.k_gains = k_gains
        self.session_s = session_s
        self._id = 0
        self._t0 = time.time()
        # The robot's own mode, straight from franka-interface's state publisher: 2 = executing a
        # skill (libfranka kMove), 1 = idle. This, not the ROS action's result, says whether the
        # skill runs: the action server aborts a goal when franka-interface's status flickers to
        # "not ready" while the robot carries on with the skill (seen 2026-11-27, skill 12).
        self.mode, self.mode_t, self._not_moving = None, 0.0, 0
        rospy.Subscriber(robot_state_topic, RobotState, self._on_state, queue_size=1)
        # robot_state's gripper_width is not filled in on this setup (reads 0.0 with the gripper
        # open); the gripper node publishes its finger positions, whose sum is the opening.
        from sensor_msgs.msg import JointState

        self._width, self._width_t = None, 0.0
        rospy.Subscriber(gripper_topic, JointState, self._on_gripper, queue_size=1)
        self.last_done_reason = ""

    def ros_down(self):
        return self._rospy.is_shutdown()

    def _on_state(self, msg):
        self.mode, self.mode_t = int(msg.robot_mode), time.time()
        self._state = (list(msg.q), list(msg.dq), float(msg.gripper_width), list(msg.O_F_ext_hat_K[:3]))

    def _on_gripper(self, msg):
        if len(msg.position) >= 2:
            self._width, self._width_t = float(msg.position[0] + msg.position[1]), time.time()

    def robot_busy(self):
        """True while the robot executes a skill (mode 2, from a fresh state message)."""
        return self.mode == MODE_MOVE and time.time() - self.mode_t < 0.5

    def _wait_mode(self, busy, timeout):
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.robot_busy() == busy:
                return True
            time.sleep(0.02)
        return False

    def goto_joints_blocking(self, q):
        if self.robot_busy():  # a skill still runs on the robot (e.g. a stream whose action was aborted)
            self.stop()
            if not self._wait_mode(False, 3.0):
                raise RuntimeError("the robot is still executing a skill and did not stop; e-stop and restart")
        now = self.fa.get_joints()
        dist = max(abs(a - b) for a, b in zip(q, now))
        q, duration = frankapy_joints_args(q, max(4.0, dist / 0.3))  # at most 0.3 rad/s on any joint
        self.fa.goto_joints(q, duration=duration, ignore_virtual_walls=False)
        err = max(abs(a - b) for a, b in zip(q, self.fa.get_joints()))
        if err > 0.02:
            raise RuntimeError(f"home not reached: max joint error {err:.3f} rad (wall, reflex or e-stop?)")

    def open_gripper(self):
        self.fa.open_gripper(block=True)

    def close_gripper(self):
        self.fa.goto_gripper(0.0, grasp=True, force=20.0, block=False)

    def open_gripper_async(self):
        self.fa.goto_gripper(MAX_WIDTH, block=False)

    def start_stream(self):
        q, duration = frankapy_joints_args(self.fa.get_joints(), self.session_s)
        kw = {"k_gains": [float(k) for k in self.k_gains]} if self.k_gains else {}
        self.fa.goto_joints(q, duration=duration, dynamic=True, buffer_time=10,
                            ignore_virtual_walls=False, **kw)
        self._id, self._t0 = 0, time.time()
        self._not_moving = 0
        if not self._wait_mode(True, 3.0):
            raise RuntimeError("the streaming skill did not start on the robot within 3 s")

    def publish(self, q, dt):
        self._id += 1
        msg = self._JPSM(id=self._id, timestamp=time.time() - self._t0, joints=list(q))
        self.pub.publish(self._make(trajectory_generator_sensor_msg=self._p2r(msg, self._SDT.JOINT_POSITION)))

    def stop(self):
        """End the skill on the robot. The skill reads a ShouldTerminate sensor message every
        control step, which works even when the ROS action was already aborted (cancelling the
        action alone does not reach the robot then). Then tidy frankapy's own bookkeeping."""
        for _ in range(5):
            msg = self._STSM(timestamp=time.time() - self._t0, should_terminate=True)
            self.pub.publish(self._make(termination_handler_sensor_msg=self._p2r(msg, self._SDT.SHOULD_TERMINATE)))
            time.sleep(0.02)
        stopped = self._wait_mode(False, 2.0)
        if self.fa._in_skill:
            try:
                self.fa._client.cancel_goal()  # non-blocking: stop_skill() would wait on the action
            except Exception as e:
                print("cancel_goal:", e, flush=True)
            self.fa._in_skill = False
        if not stopped:
            print("warning: robot still in a skill 2 s after the terminate message", flush=True)

    def skill_done(self):
        """True once the robot has left mode 2 for 3 checks in a row (0.3 s at the server's 10 Hz),
        or its state messages stopped. One flicker does not end streaming."""
        if time.time() - self.mode_t > 1.0:
            self.last_done_reason = "no robot_state message for 1 s"
            return True
        self._not_moving = 0 if self.mode == MODE_MOVE else self._not_moving + 1
        if self._not_moving >= 3:
            self.last_done_reason = f"robot_mode {self.mode} (not executing a skill) for 0.3 s"
            return True
        return False

    def state(self):
        """From the robot_state topic (~100 Hz, already subscribed): no service calls. On the arm
        (2026-11-27, trial 4) four frankapy service calls per read took 17 ms median, 95 ms at the
        95th percentile and up to 140 ms, which made RealEnv's steps late and uneven. Falls back to
        the service calls only if the topic has gone quiet."""
        st = getattr(self, "_state", None)
        if st is not None and time.time() - self.mode_t < 0.2:
            q, dq, _, f = st
            if self._width is not None and time.time() - self._width_t < 1.0:
                width = self._width
            else:
                width = float(self.fa.get_gripper_width())
            return [float(x) for x in q], [float(x) for x in dq], width, [float(x) for x in f]
        q = [float(x) for x in self.fa.get_joints()]
        qvel = [float(x) for x in self.fa.get_joint_velocities()]
        width = float(self.fa.get_gripper_width())
        ft = self.fa.get_ee_force_torque()
        return q, qvel, width, [float(x) for x in ft[:3]]


class Bridge:
    def __init__(self, arm, max_joint_speed, watchdog_s, margin, fake, publish_hz=250.0, log_path=None, period=0.05):
        self.arm, self.max_speed, self.watchdog_s, self.margin, self.fake = arm, max_joint_speed, watchdog_s, margin, fake
        self.lock = threading.RLock()
        # Interpolation. franka-interface's PassThroughJointTrajectoryGenerator sets the desired
        # joints to each message AS IS, and the joint-impedance controller pulls hard toward it:
        # a client at 20 Hz moving at 0.1 m/s makes 5 mm steps 20 times a second, and the arm
        # jerks (seen on the arm 2026-11-27, first pyramid trial). So each target starts a
        # segment from where the command is now to the new target, over one client period, and a
        # thread publishes points along it at `publish_hz`. 0 = pass targets straight through.
        self.publish_hz = publish_hz
        self.period = period  # the client's nominal control period (s): each target's segment lasts this long
        self.seg = None  # (q_from, q_to, t0, duration)
        self.cmd_q = None  # the last joints published
        self._log = open(log_path, "a", buffering=1) if log_path else None  # line-buffered: readable while running
        self.streaming = False
        self.last_q = None
        self.last_t = 0.0
        self.holding = False  # the watchdog fired: the arm holds its last target until the next one
        self.stop_reason = "never started: call home"
        self.grip_closed = False
        self.events = []
        self.t0 = time.time()
        self._alive = True
        threading.Thread(target=self._watch, daemon=True).start()
        if self.publish_hz > 0:
            threading.Thread(target=self._publisher, daemon=True).start()

    def _record(self, kind, q):
        if self._log is not None:
            self._log.write(json.dumps([round(time.time() - self.t0, 4), kind, [round(x, 5) for x in q]]) + "\n")

    def _point(self, now):
        q0, q1, t0, T = self.seg
        a = min(max((now - t0) / T, 0.0), 1.0)
        return [x + a * (y - x) for x, y in zip(q0, q1)], a >= 1.0

    def _publisher(self):
        period = 1.0 / self.publish_hz
        nxt = time.time()
        while self._alive:
            nxt += period
            with self.lock:
                if self.streaming and self.seg is not None:
                    q, done = self._point(time.time())
                    self.arm.publish(q, period)
                    self.cmd_q = q
                    self._record("pub", q)
                    if done:
                        self.seg = None  # the robot keeps the last desired joints
            time.sleep(max(0.0, nxt - time.time()))
            if time.time() - nxt > 0.5:  # fell far behind (machine busy): resync rather than burst
                nxt = time.time()

    def _halt(self, event, detail=""):
        """Stop streaming (under the lock). The arm holds where it is until `home`."""
        if self.streaming:
            self.streaming = False
            self.seg = None
            self.stop_reason = event + (f" ({detail})" if detail else "")
            try:
                self.arm.stop()
            finally:
                self.events.append(event)
                print("stopped:", self.stop_reason, flush=True)

    def _watch(self):
        """Watchdog at 50 Hz; whether the robot ended the skill at 10 Hz (a ROS service call).

        A gap in targets means HOLD, not stop: the dynamic skill keeps its last target, which is
        the safe behaviour and what RealEnv's own watchdog expects. Ending the skill instead
        would turn any pause (a prompt, a slow perception step) into a failure. The next target
        after a gap is speed-limited from the arm's measured joints, so it cannot jump."""
        n = 0
        while self._alive:
            time.sleep(0.02)
            n += 1
            with self.lock:
                if self.streaming and not self.holding and time.time() - self.last_t > self.watchdog_s:
                    self.holding = True
                    self.events.append("bridge:watchdog_hold")
                elif self.streaming and n % 5 == 0 and self.arm.skill_done():
                    self._halt("bridge:robot_ended_skill", getattr(self.arm, "last_done_reason", ""))

    def handle(self, msg):
        cmd = msg.get("cmd")
        if getattr(self.arm, "ros_down", lambda: False)():
            with self.lock:
                self.streaming = False
            return {"ok": False, "error": "this bridge lost its ROS connection: another frankapy program (another bridge, "
                    "guide_mode.py, ...) started a node with the same name and ROS shut this one down. Stop the "
                    "other program, then restart this bridge server."}
        with self.lock:
            if cmd == "hello":
                return {"ok": True, "mode": "fake" if self.fake else "frankapy", "joints": 7,
                        "limits": {"max_joint_speed": self.max_speed, "watchdog_s": self.watchdog_s, "margin": self.margin}}
            if cmd == "state":
                q, qvel, width, force = self.arm.state()
                self._record("measured", list(q) + [width] + (list(force) if force else []))  # q, width, F
                events, self.events = self.events, []
                return {"ok": True, "t": time.time() - self.t0, "q": q, "qvel": qvel, "gripper_width": width,
                        "ee_force": force, "events": events, "streaming": self.streaming}
            if cmd == "targets":
                if not self.streaming:
                    return {"ok": False, "error": f"not streaming since {self.stop_reason}: call home first"}
                q = [float(x) for x in msg["q"]]
                if len(q) != 7 or not all(math.isfinite(x) for x in q):
                    return {"ok": False, "error": "q must be 7 finite numbers"}
                now = time.time()
                if self.holding:  # resume from where the arm is, as if the last target were sent just now
                    self.last_q = [float(x) for x in self.arm.state()[0]]
                    self.last_t = now - 0.05
                    self.cmd_q, self.seg = list(self.last_q), None
                    self.holding = False
                dt = min(now - self.last_t, 0.1)
                if self.publish_hz > 0:
                    # Each target starts a segment of one NOMINAL period from where the command is
                    # now, and the speed cap is measured from that point over that period. (Before:
                    # from the previous target over the measured gap. Targets arriving in bursts were
                    # clipped hard, the arm fell 4-5 cm behind, then lurched to catch up at ~0.8 m/s
                    # and hit a can: lab, 2026-11-27, trial 4.)
                    start = self._point(now)[0] if self.seg is not None else (self.cmd_q or self.last_q)
                    q, clipped = clamp_targets(q, start, self.period, self.max_speed, self.margin)
                    self._record("target", q)
                    self.seg = (start, q, now, self.period)
                else:
                    q, clipped = clamp_targets(q, self.last_q, dt, self.max_speed, self.margin)
                    self._record("target", q)
                    self.arm.publish(q, dt)
                    self.cmd_q = q
                self.last_q, self.last_t = q, now
                if clipped:
                    self.events.append("bridge:clamped")
                closed = float(msg.get("gripper", 0.0)) > 0.5
                if closed != self.grip_closed:
                    self.grip_closed = closed
                    if closed:
                        self.arm.close_gripper()
                    elif hasattr(self.arm, "open_gripper_async"):
                        self.arm.open_gripper_async()
                    else:
                        self.arm.open_gripper()
                return {"ok": True, "clamped": clipped}
            if cmd == "stop":
                self._halt("bridge:stop")
                return {"ok": True}
            if cmd == "home":
                q = [float(x) for x in msg["q"]]
                if len(q) != 7 or any(not (Q_MIN[i] + self.margin <= q[i] <= Q_MAX[i] - self.margin) for i in range(7)):
                    return {"ok": False, "error": "home joints missing or outside the joint limits"}
                self._halt("bridge:home")
                self.events = [e for e in self.events if e != "bridge:home"]
                now = [float(x) for x in self.arm.state()[0]]
                problem = wall_problem(now)
                if problem:
                    return {"ok": False, "error": "the arm starts inside franka-interface's virtual-wall margin, so no "
                            f"skill can start: {problem}. Move it back by hand first (e-stop down for white mode, "
                            "squeeze the grey wrist buttons), then release the e-stop."}
                for k in range(1, 21):  # frankapy interpolates in joint space: check the path, not just the ends
                    s = k / 20
                    problem = wall_problem([a + s * (b - a) for a, b in zip(now, q)])
                    if problem:
                        return {"ok": False, "error": f"the move home would pass inside a virtual-wall margin at {s:.0%} "
                                f"of the way: {problem}. Move the arm nearer home by hand first."}
                try:
                    self.arm.goto_joints_blocking(q)
                    self.arm.open_gripper()
                    self.grip_closed = False
                    self.arm.start_stream()
                except Exception as e:  # report, don't die: the client decides what to do
                    return {"ok": False, "error": f"home failed: {e}"}
                self.last_q, self.last_t = [float(x) for x in self.arm.state()[0]], time.time()
                self.cmd_q, self.seg = list(self.last_q), None
                self.holding = False
                self.streaming = True
                return {"ok": True}
            if cmd == "fault" and self.fake:
                self.arm.stop()
                return {"ok": True}
            return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    def close(self):
        self._alive = False
        with self.lock:
            self._halt("bridge:shutdown")


def listen(host, port):
    """Claim the port BEFORE frankapy starts its ROS node: frankapy's node has a fixed name, and a
    second copy of this server would shut down the first one's node (ROS allows one node per name)
    before failing on the port (2026-11-27). Now a second copy fails here, harmlessly."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.settimeout(None)  # rospy sets a process-wide socket timeout; an idle server must not die of it
    try:
        srv.bind((host, port))
    except OSError as e:
        raise SystemExit(f"port {port} is taken ({e}): a bridge server is probably already running. Use that one.") from None
    srv.listen(1)
    return srv


def serve(bridge, host, port, ready=None, srv=None):
    srv = srv or listen(host, port)
    print("bridge listening on {}:{} ({})".format(host, srv.getsockname()[1], "fake" if bridge.fake else "frankapy"), flush=True)
    if ready:
        ready(srv.getsockname()[1])
    try:
        while True:
            conn, addr = srv.accept()
            conn.settimeout(None)  # a client may pause for minutes at an operator prompt
            print("client connected", addr, flush=True)
            with conn, conn.makefile("rwb") as f:
                for line in f:
                    try:
                        reply = bridge.handle(json.loads(line))
                    except Exception as e:
                        reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    f.write((json.dumps(reply) + "\n").encode())
                    f.flush()
            with bridge.lock:  # client gone: never leave the arm streaming unattended
                bridge._halt("bridge:client_disconnected")
            print("client disconnected", flush=True)
    finally:
        bridge.close()
        srv.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765, help="0 picks a free port (printed)")
    ap.add_argument("--fake", action="store_true", help="no robot: joints follow targets exactly")
    ap.add_argument("--max-joint-speed", type=float, default=0.5, help="rad/s on any joint, relative to the last target")
    ap.add_argument("--watchdog", type=float, default=0.5, help="hold (and report it) if no target arrives for this long (s)")
    ap.add_argument("--margin", type=float, default=0.05, help="rad kept clear of each joint limit")
    ap.add_argument("--publish-hz", type=float, default=250.0,
                    help="interpolate between client targets and publish at this rate (0 = pass targets straight through)")
    ap.add_argument("--log", help="append every target and published point (JSON lines) to this file")
    ap.add_argument("--period", type=float, default=0.05, help="the client's control period (s); 0.05 = RealEnv at 20 Hz")
    a = ap.parse_args()
    srv = listen(a.host, a.port)
    arm = FakeArm() if a.fake else FrankapyArm()
    serve(Bridge(arm, a.max_joint_speed, a.watchdog, a.margin, a.fake, a.publish_hz, a.log, a.period), a.host, a.port, srv=srv)


if __name__ == "__main__":
    main()
