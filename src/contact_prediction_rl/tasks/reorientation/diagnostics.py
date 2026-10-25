"""Read-only cube rotation, drift and grasp measurements for Reorientation."""

from __future__ import annotations

import math

import torch

from .orientation import orientation_error_rad, signed_twist_rad


def measure_reorientation(
    cube_pos_w: torch.Tensor,
    cube_quat_w: torch.Tensor,
    fingertip_contact: torch.Tensor,
    *,
    start_pos_w: torch.Tensor,
    start_quat_w: torch.Tensor,
    goal_quat_w: torch.Tensor,
    axis_world: torch.Tensor,
    max_drift_m: float = 0.04,
    goal_tolerance_deg: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Measure genuine physics state; no changes to task rewards or termination.

    start_pos_w is [N,3] including env origins; quaternions and axis may be
    single [4]/[3] tensors or batched. Unlike training's current success
    predicate, diagnostic success additionally requires opposing finger contact.
    """
    n = cube_pos_w.shape[0]
    if cube_pos_w.shape != (n, 3) or cube_quat_w.shape != (n, 4) or fingertip_contact.shape != (n, 5):
        raise ValueError("Expected cube [N,3], quaternion [N,4] and five contact labels [N,5]")
    if start_pos_w.shape != (n, 3) or max_drift_m <= 0 or goal_tolerance_deg <= 0:
        raise ValueError("Invalid start positions or success thresholds")
    for name, value, width in (
        ("start_quat_w", start_quat_w, 4),
        ("goal_quat_w", goal_quat_w, 4),
        ("axis_world", axis_world, 3),
    ):
        if value.shape not in ((width,), (n, width)):
            raise ValueError(f"{name} must be [{width}] or [N,{width}]")
    if not bool(((fingertip_contact == 0) | (fingertip_contact == 1)).all()):
        raise ValueError("Contact labels must be binary")
    initial = start_quat_w.expand(n, 4)
    goal = goal_quat_w.expand(n, 4)
    axis = axis_world.expand(n, 3)
    error = torch.rad2deg(orientation_error_rad(cube_quat_w, goal))
    initial_error = torch.rad2deg(orientation_error_rad(initial, goal))
    twist = torch.rad2deg(signed_twist_rad(initial, cube_quat_w, axis))
    displacement = torch.linalg.vector_norm(cube_pos_w - start_pos_w, dim=-1)
    opposing = fingertip_contact[:, 0].bool() & fingertip_contact[:, 1:].bool().any(dim=1)
    retained = opposing & (displacement <= max_drift_m)
    return {
        "goal_error_deg": error,
        "goal_progress_deg": initial_error - error,
        "signed_twist_deg": twist,
        "cube_displacement_m": displacement,
        "opposing_contact": opposing,
        "grasp_retained": retained,
        "diagnostic_success": (error <= goal_tolerance_deg) & retained,
    }


class ReorientationTracker:
    """Summarise one continuous rollout per environment without auto-resets."""

    def __init__(self, num_envs: int, device: str | torch.device = "cpu") -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.steps = 0
        self.min_error = torch.full((num_envs,), float("inf"), device=device)
        self.final_error = torch.full((num_envs,), float("nan"), device=device)
        self.min_twist = torch.full((num_envs,), float("inf"), device=device)
        self.max_twist = torch.full((num_envs,), -float("inf"), device=device)
        self.max_drift = torch.zeros(num_envs, device=device)
        self.opposing_steps = torch.zeros(num_envs, device=device)
        self.retained_steps = torch.zeros(num_envs, device=device)
        self.success_streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_success_streak = torch.zeros_like(self.success_streak)

    def observe(self, metrics: dict[str, torch.Tensor]) -> None:
        for key in (
            "goal_error_deg", "signed_twist_deg", "cube_displacement_m",
            "opposing_contact", "grasp_retained", "diagnostic_success",
        ):
            if key not in metrics or metrics[key].shape != (self.num_envs,):
                raise ValueError(f"Missing or invalid Reorientation metric: {key}")
        error = metrics["goal_error_deg"]
        twist = metrics["signed_twist_deg"]
        drift = metrics["cube_displacement_m"]
        if not all(bool(torch.isfinite(v).all()) for v in (error, twist, drift)):
            raise ValueError("Non-finite cube measurements")
        self.steps += 1
        self.min_error = torch.minimum(self.min_error, error)
        self.final_error = error.clone()
        self.min_twist = torch.minimum(self.min_twist, twist)
        self.max_twist = torch.maximum(self.max_twist, twist)
        self.max_drift = torch.maximum(self.max_drift, drift)
        self.opposing_steps += metrics["opposing_contact"].float()
        self.retained_steps += metrics["grasp_retained"].float()
        hit = metrics["diagnostic_success"].bool()
        self.success_streak = torch.where(hit, self.success_streak + 1, 0)
        self.max_success_streak = torch.maximum(self.max_success_streak, self.success_streak)

    def summary(self) -> dict[str, float]:
        if not self.steps:
            return {}
        return {
            "mean_minimum_goal_error_deg": self.min_error.mean().item(),
            "mean_final_goal_error_deg": self.final_error.mean().item(),
            "mean_max_positive_twist_deg": self.max_twist.mean().item(),
            "mean_max_negative_twist_deg": self.min_twist.mean().item(),
            "mean_max_cube_displacement_m": self.max_drift.mean().item(),
            "opposing_contact_fraction": (self.opposing_steps / self.steps).mean().item(),
            "grasp_retained_fraction": (self.retained_steps / self.steps).mean().item(),
            "ever_reached_goal_fraction": (self.max_success_streak > 0).float().mean().item(),
            "held_goal_10_steps_fraction": (self.max_success_streak >= 10).float().mean().item(),
            "at_goal_at_end_fraction": (self.success_streak > 0).float().mean().item(),
        }
