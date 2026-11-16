"""T7: the real-robot env. ROS-free core, safety limits, controllers and fakes.

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
from .safety import GuardedSkill, Safety, SafetyLimits, Watchdog, Workspace

__all__ = [
    "AutoOperator",
    "Controller",
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
    "RobotIO",
    "RobotState",
    "Safety",
    "SafetyLimits",
    "Watchdog",
    "Workspace",
    "draw_placement",
    "make_depth_twin_env",
    "make_fake_cell_env",
    "make_sim_speed_twin_env",
    "make_twin_env",
]
