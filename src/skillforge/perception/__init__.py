"""T6: pose estimation from camera frames, and the env wrapper that feeds it to skills."""

from .env import PerceivedEnv, make_depth_env, make_noisy_depth_env, make_oracle_env
from .error_model import ObjectError, PerceptionReport, measure
from .estimators import DepthEstimator, Estimate, OracleEstimator, PoseEstimator
from .frames import (
    CameraFrame,
    FrameSource,
    SimFrameSource,
    VirtualCamera,
    backproject,
    overview_camera,
    project,
)

__all__ = [
    "CameraFrame",
    "DepthEstimator",
    "Estimate",
    "FrameSource",
    "ObjectError",
    "OracleEstimator",
    "PerceivedEnv",
    "PerceptionReport",
    "PoseEstimator",
    "SimFrameSource",
    "VirtualCamera",
    "backproject",
    "make_depth_env",
    "make_noisy_depth_env",
    "make_oracle_env",
    "measure",
    "overview_camera",
    "project",
]
