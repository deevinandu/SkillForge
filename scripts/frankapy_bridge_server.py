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
- a watchdog stops the skill if no target arrives for `--watchdog` seconds while streaming;
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

    def goto_joints_blocking(self, q):
        self.q = list(q)
        self.qvel = [0.0] * 7

    def open_gripper(self):
        self.width = MAX_WIDTH

    def close_gripper(self):
        self.width = 0.0

    def start_stream(self):
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

    def __init__(self, k_gains=None, session_s=3600.0):
        import rospy
        from franka_interface_msgs.msg import SensorDataGroup
        from frankapy import FrankaArm, SensorDataMessageType
        from frankapy import FrankaConstants as FC
        from frankapy.proto import JointPositionSensorMessage
        from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg

        self._rospy, self._FC, self._SDT = rospy, FC, SensorDataMessageType
        self._JPSM, self._make, self._p2r = JointPositionSensorMessage, make_sensor_group_msg, sensor_proto2ros_msg
        self.fa = FrankaArm()  # calls rospy.init_node
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
        self.k_gains = k_gains
        self.session_s = session_s
        self._id = 0
        self._t0 = time.time()

    def goto_joints_blocking(self, q):
        now = self.fa.get_joints()
        dist = max(abs(a - b) for a, b in zip(q, now))
        duration = max(4.0, dist / 0.3)  # at most 0.3 rad/s on any joint
        self.fa.goto_joints(list(q), duration=duration, ignore_virtual_walls=False)
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
        q = self.fa.get_joints()
        kw = {"k_gains": self.k_gains} if self.k_gains else {}
        self.fa.goto_joints(list(q), duration=self.session_s, dynamic=True, buffer_time=10,
                            ignore_virtual_walls=False, **kw)
        self._id, self._t0 = 0, time.time()

    def publish(self, q, dt):
        self._id += 1
        msg = self._JPSM(id=self._id, timestamp=time.time() - self._t0, joints=list(q))
        self.pub.publish(self._make(trajectory_generator_sensor_msg=self._p2r(msg, self._SDT.JOINT_POSITION)))

    def stop(self):
        self.fa.stop_skill()

    def skill_done(self):
        # ignore_errors=False: the default waits (blocks) until franka-interface recovers from an
        # error, which would freeze the watchdog. Any error means the skill is over.
        try:
            return self.fa.is_skill_done(ignore_errors=False)
        except Exception as e:
            print("skill ended with error:", e, flush=True)
            return True

    def state(self):
        q = [float(x) for x in self.fa.get_joints()]
        qvel = [float(x) for x in self.fa.get_joint_velocities()]
        width = float(self.fa.get_gripper_width())
        ft = self.fa.get_ee_force_torque()
        return q, qvel, width, [float(x) for x in ft[:3]]


class Bridge:
    def __init__(self, arm, max_joint_speed, watchdog_s, margin, fake):
        self.arm, self.max_speed, self.watchdog_s, self.margin, self.fake = arm, max_joint_speed, watchdog_s, margin, fake
        self.lock = threading.RLock()
        self.streaming = False
        self.last_q = None
        self.last_t = 0.0
        self.grip_closed = False
        self.events = []
        self.t0 = time.time()
        self._alive = True
        threading.Thread(target=self._watch, daemon=True).start()

    def _halt(self, event):
        """Stop streaming (under the lock). The arm holds where it is until `home`."""
        if self.streaming:
            self.streaming = False
            try:
                self.arm.stop()
            finally:
                self.events.append(event)
                print("stopped:", event, flush=True)

    def _watch(self):
        """Watchdog at 50 Hz; whether the robot ended the skill at 10 Hz (a ROS service call)."""
        n = 0
        while self._alive:
            time.sleep(0.02)
            n += 1
            with self.lock:
                if self.streaming and time.time() - self.last_t > self.watchdog_s:
                    self._halt("bridge:watchdog")
                elif self.streaming and n % 5 == 0 and self.arm.skill_done():
                    self._halt("bridge:robot_ended_skill")

    def handle(self, msg):
        cmd = msg.get("cmd")
        with self.lock:
            if cmd == "hello":
                return {"ok": True, "mode": "fake" if self.fake else "frankapy", "joints": 7,
                        "limits": {"max_joint_speed": self.max_speed, "watchdog_s": self.watchdog_s, "margin": self.margin}}
            if cmd == "state":
                q, qvel, width, force = self.arm.state()
                events, self.events = self.events, []
                return {"ok": True, "t": time.time() - self.t0, "q": q, "qvel": qvel, "gripper_width": width,
                        "ee_force": force, "events": events, "streaming": self.streaming}
            if cmd == "targets":
                if not self.streaming:
                    return {"ok": False, "error": "not streaming: call home first (stopped, or the robot ended the skill)"}
                q = [float(x) for x in msg["q"]]
                if len(q) != 7 or not all(math.isfinite(x) for x in q):
                    return {"ok": False, "error": "q must be 7 finite numbers"}
                now = time.time()
                dt = min(now - self.last_t, 0.1)
                q, clipped = clamp_targets(q, self.last_q, dt, self.max_speed, self.margin)
                self.arm.publish(q, dt)
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
                try:
                    self.arm.goto_joints_blocking(q)
                    self.arm.open_gripper()
                    self.grip_closed = False
                    self.arm.start_stream()
                except Exception as e:  # report, don't die: the client decides what to do
                    return {"ok": False, "error": f"home failed: {e}"}
                self.last_q, self.last_t = self.arm.state()[0], time.time()
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


def serve(bridge, host, port, ready=None):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print("bridge listening on {}:{} ({})".format(host, srv.getsockname()[1], "fake" if bridge.fake else "frankapy"), flush=True)
    if ready:
        ready(srv.getsockname()[1])
    try:
        while True:
            conn, addr = srv.accept()
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
    ap.add_argument("--watchdog", type=float, default=0.5, help="stop if no target arrives for this long (s)")
    ap.add_argument("--margin", type=float, default=0.05, help="rad kept clear of each joint limit")
    a = ap.parse_args()
    arm = FakeArm() if a.fake else FrankapyArm()
    serve(Bridge(arm, a.max_joint_speed, a.watchdog, a.margin, a.fake), a.host, a.port)


if __name__ == "__main__":
    main()
