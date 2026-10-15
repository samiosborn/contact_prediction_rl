# src/contact_prediction_rl/tasks/lift/mdp/rewards.py

"""Reward terms that shape the grasp of the Franka + HX5-D20 hand."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse

from isaaclab_tasks.contrib.lift import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Reuse scene selectors across reward evaluations
_OBJECT = SceneEntityCfg("object")
_EE_FRAME = SceneEntityCfg("ee_frame")
_ROBOT = SceneEntityCfg("robot")
_PANDA_HAND = SceneEntityCfg("robot", body_names=["panda_hand"])


def _hand_to_object_distance(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg, grasp_offset_z: float = 0.0) -> torch.Tensor:
    """Distance [m] between the grasp centre of the hand and the part of the object the hand is meant to hold.

    The offset is along world +Z, not the rotated object axis: zero for a cube
    and half the extra height for an upright block gripped on its upper part.
    """
    object = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    held_part = object.data.root_pos_w.torch.clone()
    held_part[:, 2] += grasp_offset_z
    return torch.linalg.norm(held_part - ee_frame.data.target_pos_w.torch[..., 0, :], dim=1)


def grasp_point_reaching(
    env: ManagerBasedRLEnv,
    std: float,
    grasp_offset_z: float = 0.0,
    object_cfg: SceneEntityCfg = _OBJECT,
    ee_frame_cfg: SceneEntityCfg = _EE_FRAME,
) -> torch.Tensor:
    """Reward bringing the grasp centre of the hand to the part of the object to hold (tanh kernel)."""
    return 1.0 - torch.tanh(_hand_to_object_distance(env, object_cfg, ee_frame_cfg, grasp_offset_z) / std)


def hand_closed_near_object(
    env: ManagerBasedRLEnv,
    std: float,
    grasp_offset_z: float = 0.0,
    action_name: str = "gripper_action",
    object_cfg: SceneEntityCfg = _OBJECT,
    ee_frame_cfg: SceneEntityCfg = _EE_FRAME,
    closure_index: int = 0,
) -> torch.Tensor:
    """Reward a closed hand while the grasp centre is close to the part to hold (tanh kernel).

    ``closure`` is the commanded fraction, not realised joint closure. For
    synergy actions, ``closure_index`` selects the relevant shape (0 is curl).
    Without a closure fraction, a negative raw action counts as closed.
    """
    distance = _hand_to_object_distance(env, object_cfg, ee_frame_cfg, grasp_offset_z)
    term = env.action_manager.get_term(action_name)
    closed = term.closure[:, closure_index] if hasattr(term, "closure") else (term.raw_actions[:, 0] < 0).float()
    return closed * (1.0 - torch.tanh(distance / std))


def hand_gravity_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    target_gravity: tuple[float, float, float],
    near_std: float,
    grasp_offset_z: float = 0.0,
    asset_cfg: SceneEntityCfg = _PANDA_HAND,
    object_cfg: SceneEntityCfg = _OBJECT,
    ee_frame_cfg: SceneEntityCfg = _EE_FRAME,
) -> torch.Tensor:
    """Reward holding the hand at the orientation in which the closed hand keeps the cube.

    Whether the grasp holds depends on the direction of gravity in the hand frame, so this compares that direction
    with ``target_gravity``. The reward is weighted by the proximity of the grasp centre to the object, so it only
    matters near the cube.
    """
    robot = env.scene[asset_cfg.name]
    quat = robot.data.body_link_quat_w.torch[:, asset_cfg.body_ids[0]]
    down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    gravity = quat_apply_inverse(quat, down)
    target = torch.tensor(target_gravity, device=env.device)
    alignment = 1.0 - torch.tanh(torch.linalg.norm(gravity - target, dim=1) / std)
    distance = _hand_to_object_distance(env, object_cfg, ee_frame_cfg, grasp_offset_z)
    return alignment * (1.0 - torch.tanh(distance / near_std))


def object_lifted_in_hand(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    max_distance: float,
    grasp_offset_z: float = 0.0,
    object_cfg: SceneEntityCfg = _OBJECT,
    ee_frame_cfg: SceneEntityCfg = _EE_FRAME,
) -> torch.Tensor:
    """Reward the object being above the minimal height *and* within ``max_distance`` of the grasp centre.

    Proximity is a geometric proxy for holding, not a contact or force test.
    """
    object = env.scene[object_cfg.name]
    in_hand = _hand_to_object_distance(env, object_cfg, ee_frame_cfg, grasp_offset_z) < max_distance
    return (in_hand & (object.data.root_pos_w.torch[:, 2] > minimal_height)).float()


def object_lift_progress_in_hand(
    env: ManagerBasedRLEnv,
    rest_height: float,
    target_height: float,
    max_distance: float,
    grasp_offset_z: float = 0.0,
    object_cfg: SceneEntityCfg = _OBJECT,
    ee_frame_cfg: SceneEntityCfg = _EE_FRAME,
) -> torch.Tensor:
    """Reward the height of the object above its resting height while it is in the hand, as a fraction of the lift.

    Grows linearly from 0 at ``rest_height`` to 1 at ``target_height`` [m] and stays at 1 above it. Unlike the on/off
    lift reward, every millimetre counts, so a policy holding the object on the table is led to lift it gradually.
    """
    object = env.scene[object_cfg.name]
    in_hand = _hand_to_object_distance(env, object_cfg, ee_frame_cfg, grasp_offset_z) < max_distance
    progress = (object.data.root_pos_w.torch[:, 2] - rest_height) / (target_height - rest_height)
    return in_hand.float() * progress.clamp(0.0, 1.0)


class object_goal_distance_in_hand(mdp.object_goal_distance):
    """Goal-tracking reward (and success metric) that only counts while the object is held in the hand.

    Same as :class:`~isaaclab_tasks.contrib.lift.mdp.object_goal_distance`, except that the object must also be within
    ``max_distance`` of the grasp centre, which rejects distant objects but does not prove a force-supported grasp.

    When ``success_threshold`` is given, it also logs ``Metrics/held_at_goal_at_end``: the share of finished episodes
    whose last step had the object held within the threshold of the goal. ``Metrics/success_rate`` only needs the
    object to pass near the goal once.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._held_now = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        # rewards are computed before resets, so ``_held_now`` still describes the last step of the finished episode
        if self._track_success:
            self._env.extras.setdefault("log", {})["Metrics/held_at_goal_at_end"] = self._held_now[env_ids].float().mean().item()
        super().reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimal_height: float,
        command_name: str,
        max_distance: float,
        grasp_offset_z: float = 0.0,
        robot_cfg: SceneEntityCfg = _ROBOT,
        object_cfg: SceneEntityCfg = _OBJECT,
        ee_frame_cfg: SceneEntityCfg = _EE_FRAME,
        success_threshold: float | None = None,
    ) -> torch.Tensor:
        robot = env.scene[robot_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
        object_pos_w = env.scene[object_cfg.name].data.root_pos_w.torch
        distance = torch.linalg.norm(des_pos_w - object_pos_w, dim=1)
        in_hand = _hand_to_object_distance(env, object_cfg, ee_frame_cfg, grasp_offset_z) < max_distance
        is_lifted = (object_pos_w[:, 2] > minimal_height) & in_hand
        if success_threshold is not None:
            self._held_now = is_lifted & (distance < success_threshold)
            self._succeeded |= self._held_now
        return is_lifted.float() * (1 - torch.tanh(distance / std))


def hand_residual_penalty(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """Mean squared size of the per-joint corrections of the hand, between 0 and 1 (use with a negative weight).

    Each correction is taken as a fraction of the largest one the action allows, so the value is 0 when the hand
    follows the prior posture and 1 when every joint is at its largest correction. It pulls joints that do not matter
    for the grasp back toward the prior.
    """
    term = env.action_manager.get_term(action_name)
    return term.normalised_residual.pow(2).mean(dim=1)
