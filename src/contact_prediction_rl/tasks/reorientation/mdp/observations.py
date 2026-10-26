"""Privileged object state for goal-conditioned Reorientation."""

from __future__ import annotations

import torch

from contact_prediction_rl.tasks.reorientation.orientation import quat_conjugate, quat_multiply, quat_rotate


def object_state_palm(env) -> torch.Tensor:
    """13D: object pose and velocities relative to panda_hand."""
    obj = env.scene["object"].data
    robot = env.scene["robot"]
    hand_id = next(i for i, name in enumerate(robot.body_names) if name.split("/")[-1] == "panda_hand")
    hand = robot.data.body_link_pose_w.torch[:, hand_id]
    inverse = quat_conjugate(hand[:, 3:7])
    pos = quat_rotate(inverse, obj.root_pos_w.torch - hand[:, :3])
    rot = quat_multiply(inverse, obj.root_quat_w.torch)
    lin_vel = quat_rotate(inverse, obj.root_lin_vel_w.torch)
    ang_vel = quat_rotate(inverse, obj.root_ang_vel_w.torch)
    return torch.cat((pos, rot, lin_vel, ang_vel), dim=-1)


def goal_error_quat(env, goal_xyzw: tuple[float, float, float, float]) -> torch.Tensor:
    """Goal relative to current cube orientation, XYZW."""
    current = env.scene["object"].data.root_quat_w.torch
    goal = current.new_tensor(goal_xyzw).expand_as(current)
    delta = quat_multiply(quat_conjugate(current), goal)
    return torch.where(delta[:, 3:4] < 0, -delta, delta)


def remaining_time(env) -> torch.Tensor:
    return (1.0 - env.episode_length_buf.float() / env.max_episode_length).clamp(0.0, 1.0).unsqueeze(-1)
