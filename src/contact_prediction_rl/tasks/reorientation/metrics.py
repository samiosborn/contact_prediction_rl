"""Reusable, batched definitions of successful cube reorientation.

Pure physics-state measurements: no reward weights, episode counters, simulator
side effects, or PPO dependencies. Each returned tensor has shape [num_envs].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .orientation import orientation_error_rad


@dataclass(frozen=True)
class ReorientationSuccessThresholds:
    """Initial task thresholds; speed limits must be verified on Newton rollouts."""

    goal_tolerance_deg: float = 5.0
    max_hold_displacement_m: float = 0.02
    max_grasp_displacement_m: float = 0.04
    max_linear_speed_m_s: float = 0.10
    max_angular_speed_rad_s: float = 0.50
    sustained_steps: int = 10

    def __post_init__(self) -> None:
        values = (
            self.goal_tolerance_deg,
            self.max_hold_displacement_m,
            self.max_grasp_displacement_m,
            self.max_linear_speed_m_s,
            self.max_angular_speed_rad_s,
        )
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Success thresholds must be finite and positive")
        if self.max_hold_displacement_m > self.max_grasp_displacement_m:
            raise ValueError("Stable-hold displacement must not exceed grasp displacement")
        if self.sustained_steps < 1:
            raise ValueError("sustained_steps must be positive")


def opposing_fingertip_contact(contacts: torch.Tensor) -> torch.Tensor:
    """Thumb plus at least one other fingertip in contact with the object."""
    if contacts.ndim != 2 or contacts.shape[1] != 5:
        raise ValueError("Expected five fingertip-object contact labels [N, 5]")
    active = contacts.bool()
    return active[:, 0] & active[:, 1:].any(dim=-1)


def grasp_status(
    cube_pos_w: torch.Tensor,
    start_pos_w: torch.Tensor,
    contacts: torch.Tensor,
    *,
    max_displacement_m: float = 0.04,
) -> dict[str, torch.Tensor]:
    """Displacement and opposing-fingertip retention from world-frame states."""
    if cube_pos_w.ndim != 2 or cube_pos_w.shape[1] != 3 or start_pos_w.shape != cube_pos_w.shape:
        raise ValueError("Expected matching world positions [N, 3]")
    if max_displacement_m <= 0:
        raise ValueError("max_displacement_m must be positive")
    displacement = torch.linalg.vector_norm(cube_pos_w - start_pos_w, dim=-1)
    opposing = opposing_fingertip_contact(contacts)
    return {
        "cube_displacement_m": displacement,
        "opposing_contact": opposing,
        "grasp_retained": opposing & (displacement <= max_displacement_m),
        "object_dropped": displacement > max_displacement_m,
    }


def reorientation_step_metrics(
    cube_pos_w: torch.Tensor,
    cube_quat_w: torch.Tensor,
    cube_lin_vel_w: torch.Tensor,
    cube_ang_vel_w: torch.Tensor,
    contacts: torch.Tensor,
    *,
    start_pos_w: torch.Tensor,
    goal_quat_w: torch.Tensor,
    thresholds: ReorientationSuccessThresholds,
) -> dict[str, torch.Tensor]:
    """Evaluate instantaneous goal, contact and stability predicates.

    Caller supplies observed physics tensors; this never advances simulation.
    The extra goal tolerance and velocity gates do NOT terminate episodes.
    """
    n = cube_pos_w.shape[0]
    if cube_quat_w.shape != (n, 4) or cube_lin_vel_w.shape != (n, 3) or cube_ang_vel_w.shape != (n, 3):
        raise ValueError("Expected cube quaternion [N,4] and velocities [N,3]")
    if goal_quat_w.shape not in ((4,), (n, 4)):
        raise ValueError("Expected goal quaternion [4] or [N,4]")
    grasp = grasp_status(cube_pos_w, start_pos_w, contacts, max_displacement_m=thresholds.max_grasp_displacement_m)
    angular_error = orientation_error_rad(cube_quat_w, goal_quat_w.expand(n, 4))
    linear_speed = torch.linalg.vector_norm(cube_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.vector_norm(cube_ang_vel_w, dim=-1)
    near_goal = angular_error <= math.radians(thresholds.goal_tolerance_deg)
    stable_goal = (
        near_goal
        & grasp["opposing_contact"]
        & (grasp["cube_displacement_m"] <= thresholds.max_hold_displacement_m)
        & (linear_speed <= thresholds.max_linear_speed_m_s)
        & (angular_speed <= thresholds.max_angular_speed_rad_s)
    )
    return {
        **grasp,
        "goal_error_rad": angular_error,
        "near_goal": near_goal,
        "stable_goal": stable_goal,
        "linear_speed_m_s": linear_speed,
        "angular_speed_rad_s": angular_speed,
    }
