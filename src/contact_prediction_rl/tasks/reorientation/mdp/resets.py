"""Restore measured fingers while retaining commanded grasp targets."""

from __future__ import annotations

import torch


def restore_grasp_joint_state(env, env_ids: torch.Tensor, finger_joint_pos: dict[str, float]) -> None:
    """Runs after reset_scene_to_default; PD targets remain at the grasp command."""
    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    for name, value in finger_joint_pos.items():
        joint_pos[:, robot.joint_names.index(name)] = value
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)
