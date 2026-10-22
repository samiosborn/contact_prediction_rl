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


def residual_bounds(lower: torch.Tensor, upper: torch.Tensor, authority: float) -> torch.Tensor:
    """Largest size [rad] of the per-joint correction: ``authority`` times the joint range."""
    return authority * (upper - lower)


def step_residuals(
    residuals: torch.Tensor,
    actions: torch.Tensor,
    bounds: torch.Tensor,
    max_change: torch.Tensor,
) -> torch.Tensor:
    """Move each per-joint correction toward ``action * bound`` by at most ``max_change``.

    The correction is *not* integrated: an action of 0 asks for no correction, so the hand returns to the prior
    posture, and an action of +1 or -1 asks for the largest correction in that direction. Actions are clamped to
    [-1, 1]. The rate limit keeps the fingers from jumping.

    Args:
        residuals: Current corrections [rad], shape (num_envs, num_joints).
        actions: Policy actions, shape (num_envs, num_joints).
        bounds: Largest correction per joint, shape (num_joints,).
        max_change: Largest change per step, shape (num_joints,).
    """
    desired = actions.clamp(-1.0, 1.0) * bounds
    return residuals + torch.minimum(torch.maximum(desired - residuals, -max_change), max_change)


def prior_residual_targets(
    open_posture: torch.Tensor,
    prior_shape: torch.Tensor,
    closure: torch.Tensor,
    residuals: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Joint targets ``open + closure * prior + residual``, kept inside the allowed range of every joint.

    Args:
        open_posture: Joint values of the open hand, shape (num_joints,).
        prior_shape: Joint displacement of the prior posture at closure 1, shape (num_joints,).
        closure: Closure of the prior posture in [0, 1], shape (num_envs, 1).
        residuals: Per-joint corrections [rad], shape (num_envs, num_joints).
        lower: Lowest allowed joint value, shape (num_joints,).
        upper: Highest allowed joint value, shape (num_joints,).
    """
    return torch.clamp(open_posture + closure * prior_shape + residuals, min=lower, max=upper)
