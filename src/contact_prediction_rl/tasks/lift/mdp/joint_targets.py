# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Target dynamics of the per-joint hand action, in plain torch (no simulator needed)."""

from __future__ import annotations

import torch


def max_target_change(lower: torch.Tensor, upper: torch.Tensor, step_dt: float, closing_time: float) -> torch.Tensor:
    """Largest change [rad] of each joint target in one step: its whole range in ``closing_time`` seconds.

    With this rate every joint can go from one limit to the other in the time the single-closure hand needs for a
    full close, so a fast close of the whole hand that pushes a tall block over is still not possible.
    """
    return (upper - lower) * step_dt / closing_time


def step_joint_targets(
    targets: torch.Tensor,
    actions: torch.Tensor,
    max_change: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Move each joint target by ``action * max_change`` and keep it inside the joint limits.

    An action of 0 holds the posture, and an action of +1 or -1 moves the joint at its largest rate. Actions are
    clamped to [-1, 1].

    Args:
        targets: Current joint targets, shape (num_envs, num_joints).
        actions: Policy actions, shape (num_envs, num_joints).
        max_change: Largest change per step, shape (num_joints,).
        lower: Lower joint limits, shape (num_joints,).
        upper: Upper joint limits, shape (num_joints,).
    """
    return torch.clamp(targets + actions.clamp(-1.0, 1.0) * max_change, min=lower, max=upper)


def closure_along_shape(targets: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """How far the joint targets are along a posture ``shape``, in [0, 1] (0 for the open hand, 1 for the full shape).

    This is the projection of the targets onto the shape, so joints the shape does not move do not count. It
    lets a hand with free joints report the same "closure" as the single-closure hand, which the closed-hand reward
    uses.

    Args:
        targets: Joint targets, shape (num_envs, num_joints).
        shape: Joint displacement of the posture at closure 1, shape (num_joints,).

    Returns:
        Closure, shape (num_envs, 1).
    """
    return ((targets @ shape) / (shape @ shape)).clamp(0.0, 1.0).unsqueeze(-1)
