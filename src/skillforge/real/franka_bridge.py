"""T7: `RobotIO` for a Franka through frankapy, via a small bridge server.

frankapy runs on the lab's Python 3.8 and ROS 1; SkillForge needs Python 3.10. So the arm side
is a separate process, `scripts/frankapy_bridge_server.py`, run with frankapy's Python, and this
client talks to it over a local socket (newline-delimited JSON, one request, one reply).

The server enforces its own limits (joint speed, joint-limit margin, a watchdog, and refusing
targets once the robot has ended the skill), independent of `RealEnv`'s. When it refuses, this
client raises `BridgeError`, so a skill running in `RealEnv` stops instead of carrying on with
an arm that no longer follows.

`home()` goes to the SIMULATOR's home posture (`PandaProfile.home`), not frankapy's
`reset_joints()` pose, so a real episode starts where the matched sim episode does.

Tested against the server's --fake mode (no robot). Not yet run on the arm.
"""

from __future__ import annotations

import json
import socket

import numpy as np

from .core import RobotState

# PandaProfile.home in sim/mujoco_env.py; duplicated (and tested equal) so importing this module
# does not need MuJoCo.
SIM_HOME = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853)


class BridgeError(RuntimeError):
    """The bridge server refused a command or the connection failed."""


class FrankaBridgeRobotIO:
    """`RobotIO` over the frankapy bridge server.

    Joint targets are sent every control period by `RealEnv`; the server publishes them to one
    dynamic joint-impedance skill that `home()` starts. Gripper: above 0.5 closes (a grasp),
    below opens; the change is sent once, on the crossing.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, *, home_q=SIM_HOME, timeout: float = 2.0,
                 home_timeout: float = 120.0):
        self.home_q = [float(x) for x in home_q]
        if len(self.home_q) != 7:
            raise ValueError("home_q must have 7 joints")
        self.timeout, self.home_timeout = timeout, home_timeout
        try:
            self._sock = socket.create_connection((host, port), timeout=timeout)
        except OSError as e:
            raise BridgeError(f"cannot reach the bridge at {host}:{port} ({e}); is frankapy_bridge_server.py running?") from e
        self._f = self._sock.makefile("rwb")
        self.info = self._call({"cmd": "hello"})

    def _call(self, msg: dict, timeout: float | None = None) -> dict:
        self._sock.settimeout(timeout or self.timeout)
        try:
            self._f.write((json.dumps(msg) + "\n").encode())
            self._f.flush()
            line = self._f.readline()
        except OSError as e:
            raise BridgeError(f"bridge connection failed during {msg['cmd']!r}: {e}") from e
        if not line:
            raise BridgeError(f"bridge closed the connection during {msg['cmd']!r}")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise BridgeError(f"bridge refused {msg['cmd']!r}: {reply.get('error')}")
        return reply

    # -- RobotIO -------------------------------------------------------------

    def read_state(self) -> RobotState:
        r = self._call({"cmd": "state"})
        return RobotState(
            t=float(r["t"]),
            q=np.asarray(r["q"], dtype=float),
            qvel=np.asarray(r["qvel"], dtype=float),
            gripper_width=float(r["gripper_width"]),
            ee_force=None if r.get("ee_force") is None else np.asarray(r["ee_force"], dtype=float),
            events=list(r.get("events", [])),
        )

    def send_joint_targets(self, q: np.ndarray, gripper: float) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != (7,):
            raise ValueError(f"expected 7 joint targets, got shape {q.shape}")
        self._call({"cmd": "targets", "q": q.tolist(), "gripper": float(gripper)})

    def stop(self) -> None:
        self._call({"cmd": "stop"})

    def home(self) -> None:
        self._call({"cmd": "home", "q": self.home_q}, timeout=self.home_timeout)

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Disconnect. The server stops the arm's skill when its client goes away."""
        try:
            self._f.close()
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> FrankaBridgeRobotIO:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
