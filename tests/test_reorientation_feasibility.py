"""CPU-only tests of the diagnostic grasp/rotation helper functions."""

import math

import pytest

from contact_prediction_rl.tasks.reorientation.feasibility import (
    ARM_IK_SEED,
    FINGERS_FORWARD_WORLD,
    PALM_NORMAL_WORLD,
    ROLL_PATTERNS,
    angle_error_deg,
    apply_rotation_pattern,
    axis_angle_xyzw,
    grasp_targets,
    quat_mul_xyzw,
    rotate_vector_xyzw,
    signed_twist_deg,
)


def test_quaternion_target_rotation_and_signed_component():
    initial = axis_angle_xyzw((0, 0, 1), 20)
    axis = (1, 0, 0)
    goal = quat_mul_xyzw(axis_angle_xyzw(axis, 45), initial)
    assert angle_error_deg(initial, goal) == pytest.approx(45)
    assert angle_error_deg(goal, goal) == pytest.approx(0, abs=1e-10)
    assert signed_twist_deg(initial, goal, axis) == pytest.approx(45)
    assert signed_twist_deg(initial, goal, (-1, 0, 0)) == pytest.approx(-45)
    assert rotate_vector_xyzw(axis_angle_xyzw((0, 0, 1), 90), (1, 0, 0)) == pytest.approx((0, 1, 0), abs=1e-8)
    with pytest.raises(ValueError):
        axis_angle_xyzw((0, 0, 0), 15)


def test_action_order_and_rotation_templates():
    names = [f"finger_r_joint{i}" for i in range(20, 0, -1)]
    diagnostic = grasp_targets(names, "diagnostic")
    assert len(diagnostic) == 20
    assert diagnostic[names.index("finger_r_joint2")] == -0.8
    assert diagnostic[names.index("finger_r_joint10")] == 1.8
    for pattern in ROLL_PATTERNS:
        targets = apply_rotation_pattern(diagnostic, names, pattern, 0.2)
        assert len(targets) == 20
        assert all(math.isfinite(x) for x in targets)
        assert targets != diagnostic
    with pytest.raises(ValueError):
        grasp_targets(names[:-1], "diagnostic")


def test_palm_up_axis_targets_and_ik_seed():
    """Constrain the intended Franka orientation, independent of the hand posture."""
    assert PALM_NORMAL_WORLD == (0.0, 0.0, 1.0)
    assert FINGERS_FORWARD_WORLD == (1.0, 0.0, 0.0)
    assert sum(x * y for x, y in zip(PALM_NORMAL_WORLD, FINGERS_FORWARD_WORLD, strict=True)) == 0.0
    assert len(ARM_IK_SEED) == 7
    assert all(math.isfinite(x) for x in ARM_IK_SEED)
