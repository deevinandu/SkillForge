"""T6: pose estimation from camera frames, and the env wrapper that feeds it to skills."""

from .env import (
    PerceivedEnv,
    make_color_depth_env,
    make_depth_env,
    make_dr_color_depth_env,
    make_dr_depth_env,
    make_noisy_depth_env,
    make_oracle_env,
    make_perceived_env,
)
from .error_model import ObjectError, PerceptionReport, measure
from .estimators import DepthEstimator, Estimate, OracleEstimator, PoseEstimator
from .frames import (
    CameraFrame,
    FrameSource,
    SimFrameSource,
    VirtualCamera,
    VisualRandomization,
    backproject,
    overview_camera,
    project,
)
from .vlm_judge import Confusion, Verdict, VLMJudge

__all__ = [
    "CameraFrame",
    "Confusion",
    "DepthEstimator",
    "Estimate",
    "FrameSource",
    "ObjectError",
    "OracleEstimator",
    "PerceivedEnv",
    "PerceptionReport",
    "PoseEstimator",
    "SimFrameSource",
    "VLMJudge",
    "Verdict",
    "VirtualCamera",
    "VisualRandomization",
    "backproject",
    "make_color_depth_env",
    "make_depth_env",
    "make_dr_color_depth_env",
    "make_dr_depth_env",
    "make_noisy_depth_env",
    "make_oracle_env",
    "make_perceived_env",
    "measure",
    "overview_camera",
    "project",
]
