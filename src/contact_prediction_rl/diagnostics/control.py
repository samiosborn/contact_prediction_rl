"""Action and joint-controller measurements; no action semantics are assumed."""

from __future__ import annotations

import torch


def action_statistics(actions: torch.Tensor, *, action_limit: float | None = None) -> dict[str, float]:
    """Summarise raw continuous actions; action units depend on the action term."""
    if actions.ndim != 2 or not torch.isfinite(actions).all():
        raise ValueError("Actions must be finite [num_envs, action_dim]")
    result = {
        "mean_abs_action": actions.abs().mean().item(),
        "rms_action": actions.square().mean().sqrt().item(),
        "max_abs_action": actions.abs().max().item(),
    }
    if action_limit is not None:
        if action_limit <= 0:
            raise ValueError("action_limit must be positive")
        result["fraction_at_action_limit"] = (actions.abs() >= action_limit - 1e-6).float().mean().item()
    return result


def joint_tracking_statistics(
    measured: torch.Tensor,
    commanded: torch.Tensor,
    *,
    limits: torch.Tensor | None = None,
    limit_margin_rad: float = 0.05,
) -> dict[str, float]:
    """Compare actual joint positions with physical joint targets (both radians).

    The caller must translate the action through its actual task action term:
    Reach, Lift's synergies and Reorientation do NOT share action units.
    limits may have shape [joints, 2] or [envs, joints, 2].
    """
    if measured.ndim != 2 or measured.shape != commanded.shape or not torch.isfinite(measured).all() or not torch.isfinite(commanded).all():
        raise ValueError("Measured and commanded joints must be finite matching [envs, joints] tensors")
    error = measured - commanded
    result = {
        "mean_abs_joint_target_error_rad": error.abs().mean().item(),
        "max_abs_joint_target_error_rad": error.abs().max().item(),
    }
    if limits is not None:
        if limit_margin_rad < 0 or limits.shape not in ((measured.shape[1], 2), (*measured.shape, 2)):
            raise ValueError("Invalid joint limits or margin")
        lower, upper = limits[..., 0], limits[..., 1]
        if not bool((upper > lower).all()):
            raise ValueError("Joint upper limits must exceed lower limits")
        margin = torch.minimum(measured - lower, upper - measured)
        result["minimum_joint_limit_margin_rad"] = margin.min().item()
        result["fraction_near_joint_limit"] = (margin <= limit_margin_rad).float().mean().item()
    return result
