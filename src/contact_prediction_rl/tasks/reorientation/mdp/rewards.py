# src/contact_prediction_rl/tasks/reorientation/mdp/rewards.py

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from contact_prediction_rl.robots.hx5_d20_patches import fingertip_patch_contact
from contact_prediction_rl.tasks.reorientation.diagnostics import ReorientationEpisodeMetrics
from contact_prediction_rl.tasks.reorientation.metrics import ReorientationSuccessThresholds, grasp_status, reorientation_step_metrics
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



def orientation_progress_reward(
    env: ManagerBasedRLEnv,
    goal_xyzw: tuple[float, float, float, float],
    start_pos_w: tuple[float, float, float],
    goal_degrees: float = 15.0,
    max_grasp_displacement_m: float = 0.04,
) -> torch.Tensor:
    """Dense, bounded orientation quality only while the cube remains grasped.

    Zero at the 15-degree start pose, one at the goal; never rewards a
    correctly oriented cube that has lost opposing fingertip contact.
    Unlike the previous always-negative error penalty, the policy gains
    nothing by causing an early time-out (time-out is now the only reset).
    """
    error = goal_error_rad(env, goal_xyzw)
    cube_pos = env.scene["object"].data.root_pos_w.torch
    origin = cube_pos.new_tensor(start_pos_w) + env.scene.env_origins
    held = grasp_status(
        cube_pos, origin, fingertip_patch_contact(env), max_displacement_m=max_grasp_displacement_m
    )["grasp_retained"]
    quality = (1.0 - error / math.radians(goal_degrees)).clamp(0.0, 1.0)
    return quality * held.float()


class StableGoalReward(ManagerTermBase):
    """Reward stable goal occupancy, logging episode metrics at time-out resets.

    Each invocation is one environment control step. Metrics are computed
    independently from reward weights in reorientation/metrics.py; only the
    returned stable-goal indicator is used as a reward.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        sustained_steps = int(cfg.params.get("sustained_steps", 10))
        self.tracker = ReorientationEpisodeMetrics(env.num_envs, sustained_steps=sustained_steps, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        summary = self.tracker.summarise_completed(env_ids)
        if summary:
            self._env.extras.setdefault("log", {}).update(summary)
        self.tracker.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        goal_xyzw: tuple[float, float, float, float],
        start_pos_w: tuple[float, float, float],
        goal_tolerance_deg: float = 5.0,
        max_hold_displacement_m: float = 0.02,
        max_grasp_displacement_m: float = 0.04,
        max_linear_speed_m_s: float = 0.10,
        max_angular_speed_rad_s: float = 0.50,
        sustained_steps: int = 10,
    ) -> torch.Tensor:
        """Return stable goal occupancy; never terminates the episode."""
        thresholds = ReorientationSuccessThresholds(
            goal_tolerance_deg=goal_tolerance_deg,
            max_hold_displacement_m=max_hold_displacement_m,
            max_grasp_displacement_m=max_grasp_displacement_m,
            max_linear_speed_m_s=max_linear_speed_m_s,
            max_angular_speed_rad_s=max_angular_speed_rad_s,
            sustained_steps=sustained_steps,
        )
        cube = env.scene["object"].data
        pos = cube.root_pos_w.torch
        state = reorientation_step_metrics(
            pos,
            cube.root_quat_w.torch,
            cube.root_lin_vel_w.torch,
            cube.root_ang_vel_w.torch,
            fingertip_patch_contact(env),
            start_pos_w=pos.new_tensor(start_pos_w) + env.scene.env_origins,
            goal_quat_w=pos.new_tensor(goal_xyzw),
            thresholds=thresholds,
        )
        self.tracker.observe(state)
        return state["stable_goal"].float()
