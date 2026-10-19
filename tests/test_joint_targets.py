# tests/test_joint_targets.py

"""Tests for the target dynamics of the per-joint hand action (plain torch, no simulator)."""

import pytest
import torch

from contact_prediction_rl.robots.franka_hx5d20 import (
    FRANKA_HX5D20_HAND_SYNERGIES,
    resolve_joint_posture,
)
from contact_prediction_rl.tasks.lift.mdp.joint_targets import (
    closure_along_shape,
    max_target_change,
    step_joint_targets,
)

STEP_DT = 0.02
CLOSING_TIME = 2.0
JOINT_NAMES = [f"finger_r_joint{i}" for i in range(1, 21)]
# lower and upper limits [rad] of finger_r_joint1 ... finger_r_joint20 in the HX5-D20 URDF
LIMITS = [(-1.57, 1.57), (-1.57, 0.0), (0.0, 1.57), (0.0, 1.57)] + 4 * [
    (-0.6, 0.6),
    (0.0, 2.0),
    (0.0, 1.57),
    (0.0, 1.57),
]
LOWER = torch.tensor([lo for lo, _ in LIMITS])
UPPER = torch.tensor([hi for _, hi in LIMITS])
MAX_CHANGE = max_target_change(LOWER, UPPER, STEP_DT, CLOSING_TIME)


def test_a_joint_covers_its_range_in_the_closing_time():
    targets = LOWER.repeat(1, 1)
    steps = 0

    while (targets < UPPER - 1e-4).any():
        targets = step_joint_targets(targets, torch.ones(1, 20), MAX_CHANGE, LOWER, UPPER)
        steps += 1

    assert steps * STEP_DT == pytest.approx(CLOSING_TIME, abs=STEP_DT)


def test_zero_action_holds_the_posture():
    targets = torch.rand(4, 20) * (UPPER - LOWER) + LOWER

    held = step_joint_targets(targets, torch.zeros(4, 20), MAX_CHANGE, LOWER, UPPER)

    assert held == pytest.approx(targets.numpy())


def test_targets_stay_inside_the_joint_limits():
    generator = torch.Generator().manual_seed(0)
    targets = torch.zeros(8, 20)

    for _ in range(500):
        actions = torch.randn(8, 20, generator=generator) * 3.0
        targets = step_joint_targets(targets, actions, MAX_CHANGE, LOWER, UPPER)

        assert (targets >= LOWER - 1e-6).all() and (targets <= UPPER + 1e-6).all()


def test_change_per_step_is_limited_and_actions_are_clamped():
    targets = torch.zeros(1, 20) + 0.5 * (LOWER + UPPER)

    moved = step_joint_targets(targets, torch.full((1, 20), 100.0), MAX_CHANGE, LOWER, UPPER)

    assert (moved - targets)[0] == pytest.approx(MAX_CHANGE.numpy(), abs=1e-6)


def test_closure_is_zero_for_the_open_hand_and_one_for_the_curl_posture():
    curl = torch.tensor(resolve_joint_posture(FRANKA_HX5D20_HAND_SYNERGIES["curl"], JOINT_NAMES))

    assert closure_along_shape(torch.zeros(1, 20), curl).item() == 0.0
    assert closure_along_shape(curl.repeat(1, 1), curl).item() == pytest.approx(1.0)
    assert closure_along_shape(0.5 * curl.repeat(1, 1), curl).item() == pytest.approx(0.5)


def test_closure_ignores_the_joints_the_curl_shape_does_not_move():
    curl = torch.tensor(resolve_joint_posture(FRANKA_HX5D20_HAND_SYNERGIES["curl"], JOINT_NAMES))
    thumb = torch.tensor(resolve_joint_posture(FRANKA_HX5D20_HAND_SYNERGIES["thumb"], JOINT_NAMES))

    assert closure_along_shape(thumb.repeat(1, 1), curl).item() == 0.0
    assert closure_along_shape((curl + thumb).repeat(1, 1), curl).item() == pytest.approx(1.0)
