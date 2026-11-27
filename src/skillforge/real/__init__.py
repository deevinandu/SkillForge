"""T7: the real-robot env. ROS-free core, safety limits, controllers, fakes and the RealSense camera source.

Not here yet: the ROS 2 node, the session runner, system ID and the sim2real scoreboard.
Nothing in this package has run on a real arm.
"""

from .controller import MujocoController, PointController
from .core import (
    AutoOperator,
    Controller,
    Operator,
    Placement,
    PlacementAborted,
    RealEnv,
    RobotIO,
    RobotState,
    draw_placement,
)
from .fakes import (
    FakeCell,
    FakeRobotIO,
    MujocoTwin,
    make_depth_twin_env,
    make_fake_cell_env,
    make_sim_speed_twin_env,
    make_twin_env,
)
from .franka_bridge import SIM_HOME, BridgeError, FrankaBridgeRobotIO
from .realsense import (
    RealSenseCamera,
    RealSenseFrameSource,
    calibration_overlay,
    list_devices,
    load_extrinsics,
    save_extrinsics,
)
from .safety import GuardedSkill, Safety, SafetyLimits, Watchdog, Workspace

__all__ = [
    "SIM_HOME",
    "AutoOperator",
    "BridgeError",
    "Controller",
    "FrankaBridgeRobotIO",
    "FakeCell",
    "FakeRobotIO",
    "GuardedSkill",
    "MujocoController",
    "MujocoTwin",
    "Operator",
    "Placement",
    "PlacementAborted",
    "PointController",
    "RealEnv",
    "RealSenseCamera",
    "RealSenseFrameSource",
    "RobotIO",
    "RobotState",
    "Safety",
    "SafetyLimits",
    "Watchdog",
    "Workspace",
    "calibration_overlay",
    "draw_placement",
    "list_devices",
    "load_extrinsics",
    "make_depth_twin_env",
    "make_fake_cell_env",
    "make_sim_speed_twin_env",
    "make_twin_env",
    "save_extrinsics",
]
