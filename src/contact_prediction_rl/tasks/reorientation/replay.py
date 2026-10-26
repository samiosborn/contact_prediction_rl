"""Convert verified scripted finger trajectories into the training action ordering."""

from __future__ import annotations

import math

import torch

from .feasibility import ROLL_PATTERNS, apply_rotation_pattern, finger_gait_phases


def rotation_waypoints(
    closed: list[float],
    joint_names: list[str],
    *,
    controller: str,
    pattern: str,
    amplitude_rad: float,
    rotate_steps: int = 150,
    ramp_steps: int = 90,
    gait_cycles: int = 1,
) -> list[list[float]]:
    """Generate the same interpolated target sequence as the Newton feasibility script."""
    if pattern not in ROLL_PATTERNS or controller not in ("gait", "offset"):
        raise ValueError("Unknown rotation pattern or controller")
    if not math.isfinite(amplitude_rad) or min(rotate_steps, ramp_steps, gait_cycles) <= 0 or ramp_steps > rotate_steps:
        raise ValueError("Invalid amplitude, step counts or ramp")
    if len(closed) != len(joint_names) or len(closed) != 20:
        raise ValueError("A full ordered 20-joint closed grasp is required")

    if controller == "offset":
        commanded = apply_rotation_pattern(closed, joint_names, pattern, amplitude_rad)
        return [
            [a + (b - a) * min(step / ramp_steps, 1.0) for a, b in zip(closed, commanded, strict=True)]
            for step in range(1, rotate_steps + 1)
        ]

    if rotate_steps < 8 * gait_cycles:
        raise ValueError("Gait requires at least eight steps per cycle")
    phases = finger_gait_phases(closed, joint_names, pattern, amplitude_rad)
    outputs: list[list[float]] = []
    previous = list(closed)
    for cycle in range(gait_cycles):
        cycle_start = round(cycle * rotate_steps / gait_cycles)
        cycle_end = round((cycle + 1) * rotate_steps / gait_cycles)
        previous_end = cycle_start
        accumulated = 0.0
        for phase_index, phase in enumerate(phases):
            accumulated += phase.fraction
            boundary = cycle_end if phase_index == len(phases) - 1 else round(
                cycle_start + accumulated * (cycle_end - cycle_start)
            )
            n_steps = boundary - previous_end
            if n_steps < 1:
                raise ValueError("Too few timesteps for this gait phase")
            target = list(phase.targets)
            for step in range(1, n_steps + 1):
                alpha = step / n_steps
                outputs.append([a + alpha * (b - a) for a, b in zip(previous, target, strict=True)])
            previous = target
            previous_end = boundary
    if len(outputs) != rotate_steps:
        raise RuntimeError("Gait interpolation did not produce the requested step count")
    return outputs


def targets_to_actions(targets: torch.Tensor, *, offset: torch.Tensor | float, scale: torch.Tensor | float) -> torch.Tensor:
    """Invert Isaac Lab's joint-position action mapping: target = offset + scale * action.

    Use the *live action term's* resolved offset and scale, in its own joint
    ordering. No global assumption about the 20 finger indices or action units.
    """
    if targets.ndim != 2 or not torch.isfinite(targets).all():
        raise ValueError("Targets must be finite [envs, joints]")
    try:
        offset = torch.broadcast_to(torch.as_tensor(offset, device=targets.device, dtype=targets.dtype), targets.shape)
        scale = torch.broadcast_to(torch.as_tensor(scale, device=targets.device, dtype=targets.dtype), targets.shape)
    except RuntimeError as exc:
        raise ValueError("Offset/scale do not broadcast to commanded targets") from exc
    if not torch.isfinite(offset).all() or not torch.isfinite(scale).all() or bool((scale == 0).any()):
        raise ValueError("Invalid action offset or zero/nonfinite scale")
    return (targets - offset) / scale
