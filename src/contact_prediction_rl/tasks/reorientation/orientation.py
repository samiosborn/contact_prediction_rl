"""Batched XYZW quaternion geometry shared by Reorientation diagnostics and rewards.

Isaac Lab 3 uses XYZW for object poses, link poses and math utilities.
All angles in this module are radians; inputs may have arbitrary batch dimensions.
"""

from __future__ import annotations

import math

import torch


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av, aw = a[..., :3], a[..., 3:4]
    bv, bw = b[..., :3], b[..., 3:4]
    return torch.cat((aw * bv + bw * av + torch.cross(av, bv, dim=-1), aw * bw - (av * bv).sum(dim=-1, keepdim=True)), dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = q[..., :3]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., 3:4] * t + torch.cross(qv, t, dim=-1)


def axis_angle_quat(axis: torch.Tensor, radians: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    if torch.any(norm < 1e-8):
        raise ValueError("Rotation axis cannot be zero")
    half = 0.5 * radians.unsqueeze(-1)
    return torch.cat((axis / norm * torch.sin(half), torch.cos(half)), dim=-1)


def goal_orientation(initial: torch.Tensor, axis_world: torch.Tensor, radians: torch.Tensor) -> torch.Tensor:
    """Apply a signed world-frame rotation to the starting object orientation."""
    return quat_multiply(axis_angle_quat(axis_world, radians), initial)


def orientation_error_rad(current: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Shortest SO(3) geodesic angle, invariant to quaternion sign."""
    current = torch.nn.functional.normalize(current, dim=-1)
    goal = torch.nn.functional.normalize(goal, dim=-1)
    dot = (current * goal).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.atan2(torch.sqrt(torch.clamp(1.0 - dot * dot, min=0.0)), dot)


def signed_twist_rad(initial: torch.Tensor, current: torch.Tensor, axis_world: torch.Tensor) -> torch.Tensor:
    """Signed twist component of the relative rotation about a world axis."""
    delta = quat_multiply(current, quat_conjugate(initial))
    projection = (delta[..., :3] * torch.nn.functional.normalize(axis_world, dim=-1)).sum(dim=-1)
    phase = 2.0 * torch.atan2(projection, delta[..., 3])
    return torch.atan2(torch.sin(phase), torch.cos(phase))


# Scalar helpers retain the feasibility script's tuple interface.
def _tensor(value):
    return torch.as_tensor(value, dtype=torch.float64).reshape(1, -1)


def quat_mul_xyzw(a, b) -> tuple[float, float, float, float]:
    return tuple(quat_multiply(_tensor(a), _tensor(b))[0].tolist())


def axis_angle_xyzw(axis, degrees: float) -> tuple[float, float, float, float]:
    return tuple(axis_angle_quat(_tensor(axis), torch.tensor([math.radians(degrees)], dtype=torch.float64))[0].tolist())


def rotate_vector_xyzw(q, vec) -> tuple[float, float, float]:
    return tuple(quat_rotate(_tensor(q), _tensor(vec))[0].tolist())


def angle_error_deg(q, goal) -> float:
    return math.degrees(float(orientation_error_rad(_tensor(q), _tensor(goal))[0]))


def signed_twist_deg(initial, current, axis_world) -> float:
    return math.degrees(float(signed_twist_rad(_tensor(initial), _tensor(current), _tensor(axis_world))[0]))
