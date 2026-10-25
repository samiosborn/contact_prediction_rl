# src/contact_prediction_rl/tasks/reorientation/mdp/rewards.py

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from contact_prediction_rl.tasks.reorientation.orientation import orientation_error_rad

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return summed squared error from a scalar target after wrapping joint angles to the principal interval."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


def goal_error_rad(env: ManagerBasedRLEnv, goal_xyzw: tuple[float, float, float, float]) -> torch.Tensor:
    cube_quat = env.scene["object"].data.root_quat_w.torch
    return orientation_error_rad(cube_quat, cube_quat.new_tensor(goal_xyzw).expand_as(cube_quat))


def object_dropped(env: ManagerBasedRLEnv, start_pos_w: tuple[float, float, float], max_displacement_m: float = 0.04) -> torch.Tensor:
    """Detect leaving the reset grasp, including falling onto the table."""
    object_pos = env.scene["object"].data.root_pos_w.torch
    initial = object_pos.new_tensor(start_pos_w) + env.scene.env_origins
    return torch.linalg.vector_norm(object_pos - initial, dim=-1) > max_displacement_m


def goal_reached(env: ManagerBasedRLEnv, goal_xyzw: tuple[float, float, float, float], start_pos_w: tuple[float, float, float], tolerance_deg: float = 5.0) -> torch.Tensor:
    return (goal_error_rad(env, goal_xyzw) < math.radians(tolerance_deg)) & ~object_dropped(env, start_pos_w)


def orientation_reward(env: ManagerBasedRLEnv, goal_xyzw: tuple[float, float, float, float], goal_degrees: float = 15.0) -> torch.Tensor:
    return -goal_error_rad(env, goal_xyzw) / math.radians(goal_degrees)


def success_reward(env: ManagerBasedRLEnv, goal_xyzw: tuple[float, float, float, float], start_pos_w: tuple[float, float, float]) -> torch.Tensor:
    return goal_reached(env, goal_xyzw, start_pos_w).float()


def drop_penalty(env: ManagerBasedRLEnv, start_pos_w: tuple[float, float, float]) -> torch.Tensor:
    return object_dropped(env, start_pos_w).float()
