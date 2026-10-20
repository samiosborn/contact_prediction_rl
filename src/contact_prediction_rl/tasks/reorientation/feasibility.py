"""Diagnostic grasp and rotation trajectories; no Isaac Lab or PPO dependency.

Candidate poses are not verified training resets. Geometric contact patches,
policy inputs/outputs and PPO remain independent of this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FINGERS = ("thumb", "index", "middle", "ring", "little")
# A known reachable posture used only as the initial guess for the
# compiled-Newton palm-up IK solve. It is NOT the trial's reset pose.
ARM_IK_SEED = (0.3697, -0.2897, -0.3216, -2.0356, -0.0921, 1.7594, -2.277)

# Hand-link conventions: local +X is the palm normal; local +Z points
# along the fingers. World +Z = palm up, world +X = fingers forward.
PALM_NORMAL_WORLD = (0.0, 0.0, 1.0)
FINGERS_FORWARD_WORLD = (1.0, 0.0, 0.0)

# Search near the central palm first (shorter +Z and neutral lateral Y).
# The previously verified stable grasp (0.050, 0.025, 0.100) is retained
# within the first three centres as a fallback; no new pose is prevalidated.
GRASP_CENTRES = (
    (0.050, 0.000, 0.070),
    (0.045, 0.000, 0.085),
    (0.050, 0.025, 0.100),  # stable in the prior palm-up Newton run
    (0.050, 0.000, 0.100),
    (0.045, -0.025, 0.080),
    (0.045, 0.025, 0.080),
    (0.040, 0.000, 0.060),
    (0.050, 0.000, 0.055),
    (0.040, 0.000, 0.120),
    (0.040, -0.025, 0.120),
    (0.040, 0.025, 0.120),
)
DIAGNOSTIC_THUMB = (0.4, -0.8, 1.0, 1.0)
DIAGNOSTIC_FINGER = (0.0, 1.8, 1.4, 1.4)

# Additive finger-position directions. Both signs and several amplitudes are
# tested; these patterns are hypotheses, not a demonstrated rotation policy.
ROLL_PATTERNS = {
    "spread": {5: -1.0, 9: -0.5, 13: 0.5, 17: 1.0},
    "differential_curl": {6: 0.75, 7: 0.5, 10: 0.75, 11: 0.5,
                          14: -0.75, 15: -0.5, 18: -0.75, 19: -0.5},
    "thumb_index": {1: 0.5, 2: -0.6, 6: -0.75, 7: -0.4, 10: -0.3, 14: 0.3},
}


@dataclass(frozen=True)
class GraspCandidate:
    centre_hand_m: tuple[float, float, float]
    posture: str = "lift"

    def label(self) -> str:
        return f"{self.posture}@{','.join(f'{v:.3f}' for v in self.centre_hand_m)}"


def grasp_targets(joint_names: list[str], posture: str) -> list[float]:
    """20 joint-position commands in the actual action-term joint order."""
    if set(joint_names) != {f"finger_r_joint{i}" for i in range(1, 21)} or len(joint_names) != 20:
        raise ValueError("Expected the 20 uniquely named HX5-D20 finger action joints")
    if posture == "lift":
        from contact_prediction_rl.robots.franka_hx5d20 import FRANKA_HX5D20_HAND_CLOSED, resolve_joint_posture

        return resolve_joint_posture(FRANKA_HX5D20_HAND_CLOSED, joint_names)
    if posture == "diagnostic":
        commands = {f"finger_r_joint{i + 1}": value for i, value in enumerate(DIAGNOSTIC_THUMB)}
        for finger in range(1, 5):
            commands.update({f"finger_r_joint{4 * finger + i + 1}": value for i, value in enumerate(DIAGNOSTIC_FINGER)})
        return [commands[name] for name in joint_names]
    raise ValueError(f"Unknown grasp posture: {posture}")


def apply_rotation_pattern(closed: list[float], joint_names: list[str], pattern: str, signed_amplitude_rad: float) -> list[float]:
    """Desired finger targets; clamp to articulation joint limits in the runner."""
    if len(closed) != 20 or len(joint_names) != 20:
        raise ValueError("Expected 20-D closed posture and action-joint names")
    result = list(closed)
    for joint_no, direction in ROLL_PATTERNS[pattern].items():
        index = joint_names.index(f"finger_r_joint{joint_no}")
        result[index] += signed_amplitude_rad * direction
    return result


@dataclass(frozen=True)
class GaitPhase:
    """One finger-position waypoint and relative share of a rotation cycle."""

    name: str
    fraction: float
    targets: tuple[float, ...]


def finger_gait_phases(closed: list[float], joint_names: list[str], pattern: str, signed_amplitude_rad: float) -> tuple[GaitPhase, ...]:
    """Create a staged unload / tangential sweep / regrasp / recovery motion.

    One non-thumb finger reduces curl while another and the thumb continue
    opposing the cube. A tangential finger displacement is applied during
    the sweep; the finger reloads *before* recovering the sweep. All motions
    are continuous joint-position interpolations, never object teleports.
    """
    if len(closed) != 20 or len(joint_names) != 20:
        raise ValueError("Finger gait requires a 20-dimensional hand target")
    if pattern not in ROLL_PATTERNS:
        raise ValueError(f"Unknown finger gait pattern: {pattern}")
    if not math.isfinite(signed_amplitude_rad):
        raise ValueError("Non-finite finger gait amplitude")

    # The thumb is held during the unloading phase. The other finger remains
    # a potential opposing contact, but actual contact is checked in physics.
    unloading = {
        "spread": (6, 7, 8),
        "differential_curl": (10, 11, 12),
        "thumb_index": (6, 7, 8),
    }[pattern]
    relaxed = list(closed)
    magnitude = 0.75 * abs(signed_amplitude_rad)
    for joint_no in unloading:
        index = joint_names.index(f"finger_r_joint{joint_no}")
        relaxed[index] = max(0.0, closed[index] - magnitude)

    sweep = apply_rotation_pattern(relaxed, joint_names, pattern, signed_amplitude_rad)
    reload = apply_rotation_pattern(closed, joint_names, pattern, signed_amplitude_rad)
    return (
        GaitPhase("unload", 0.20, tuple(relaxed)),
        GaitPhase("sweep", 0.30, tuple(sweep)),
        GaitPhase("regrasp", 0.25, tuple(reload)),
        GaitPhase("recover", 0.25, tuple(closed)),
    )


def quat_mul_xyzw(a, b) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def axis_angle_xyzw(axis, degrees: float) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(x * x for x in axis))
    if length < 1e-8:
        raise ValueError("Rotation axis cannot be zero")
    h = math.radians(degrees) / 2
    return (*(math.sin(h) * x / length for x in axis), math.cos(h))


def rotate_vector_xyzw(q, vec) -> tuple[float, float, float]:
    qinv = (-q[0], -q[1], -q[2], q[3])
    rotated = quat_mul_xyzw(quat_mul_xyzw(q, (*vec, 0.0)), qinv)
    return rotated[:3]


def angle_error_deg(q, goal) -> float:
    """Shortest SO(3) quaternion distance; this diagnostic does not fold cube symmetries."""
    inverse = (-goal[0], -goal[1], -goal[2], goal[3])
    delta = quat_mul_xyzw(q, inverse)
    return math.degrees(2 * math.atan2(math.sqrt(sum(v * v for v in delta[:3])), abs(delta[3])))


def signed_twist_deg(initial, current, axis_world) -> float:
    """Signed component of the object's relative orientation around a world axis."""
    inverse = (-initial[0], -initial[1], -initial[2], initial[3])
    delta = quat_mul_xyzw(current, inverse)
    projection = sum(x * y for x, y in zip(delta[:3], axis_world, strict=True))
    radians = 2 * math.atan2(projection, delta[3])
    return math.degrees((radians + math.pi) % (2 * math.pi) - math.pi)
