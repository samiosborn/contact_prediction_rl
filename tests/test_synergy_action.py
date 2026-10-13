# tests/test_synergy_action.py

"""Tests for the closure dynamics of the synergy hand action (plain torch, no simulator)."""

import pytest
import torch

from contact_prediction_rl.robots.franka_hx5d20 import (
    FRANKA_HX5D20_HAND_CLOSED,
    FRANKA_HX5D20_HAND_SYNERGIES,
    resolve_joint_posture,
)
from contact_prediction_rl.tasks.lift.mdp.synergies import (
    step_closures,
    synergy_posture,
    target_closures,
)

STEP_DT = 0.02
CLOSING_TIME = 2.0
MAX_CHANGE = STEP_DT / CLOSING_TIME
JOINT_NAMES = [f"finger_r_joint{i}" for i in range(1, 21)]


def binary_closure_step(closure: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """The closure update of RateLimitedBinaryJointPositionAction, copied as the reference."""
    close = actions < 0
    return closure + (close.float() - closure).clamp(-MAX_CHANGE, MAX_CHANGE)


def test_target_closures_close_on_negative_actions_only():
    actions = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0, -3.0]])

    assert target_closures(actions).tolist() == [[1.0, 0.5, 0.0, 0.0, 0.0, 1.0]]


def test_one_synergy_with_binary_actions_matches_the_binary_hand_action():
    generator = torch.Generator().manual_seed(0)
    closure = torch.zeros(16, 1)
    reference = torch.zeros(16, 1)

    for _ in range(300):
        actions = torch.where(torch.rand(16, 1, generator=generator) < 0.5, -1.0, 1.0)
        closure = step_closures(closure, actions, MAX_CHANGE)
        reference = binary_closure_step(reference, actions)

        assert closure == pytest.approx(reference.numpy(), abs=1e-6)


def test_closing_takes_the_closing_time_and_changes_at_a_limited_rate():
    closure = torch.zeros(1, 3)
    full_close = torch.full((1, 3), -1.0)
    steps = 0

    while closure.min() < 1.0 - 1e-6:
        new = step_closures(closure, full_close, MAX_CHANGE)

        assert (new - closure).abs().max() <= MAX_CHANGE + 1e-7
        closure, steps = new, steps + 1

    assert steps * STEP_DT == pytest.approx(CLOSING_TIME, abs=STEP_DT)


def test_half_closure_command_stops_at_half_closure():
    closure = torch.zeros(1, 1)

    for _ in range(200):
        closure = step_closures(closure, torch.tensor([[-0.5]]), MAX_CHANGE)

    assert closure.item() == pytest.approx(0.5, abs=1e-6)


def test_synergies_are_independent():
    closure = torch.zeros(1, 3)
    actions = torch.tensor([[-1.0, 0.0, -0.5]])

    for _ in range(300):
        closure = step_closures(closure, actions, MAX_CHANGE)

    assert closure.tolist()[0] == pytest.approx([1.0, 0.0, 0.5], abs=1e-6)


def posture_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    open_posture = torch.zeros(len(JOINT_NAMES))
    shapes = torch.tensor([resolve_joint_posture(p, JOINT_NAMES) for p in FRANKA_HX5D20_HAND_SYNERGIES.values()])
    return open_posture, shapes


def test_full_curl_and_thumb_give_the_closed_posture():
    open_posture, shapes = posture_tensors()
    closures = torch.tensor([[1.0, 1.0, 0.0]])
    expected = torch.tensor(resolve_joint_posture(FRANKA_HX5D20_HAND_CLOSED, JOINT_NAMES))

    assert synergy_posture(open_posture, shapes, closures)[0] == pytest.approx(expected.numpy(), abs=1e-6)


def test_zero_closure_is_the_open_posture_and_closures_scale_linearly():
    open_posture, shapes = posture_tensors()

    assert synergy_posture(open_posture, shapes, torch.zeros(2, 3)).abs().max() == 0.0

    full = synergy_posture(open_posture, shapes, torch.tensor([[1.0, 1.0, 1.0]]))
    half = synergy_posture(open_posture, shapes, torch.tensor([[0.5, 0.5, 0.5]]))

    assert half == pytest.approx((0.5 * full).numpy(), abs=1e-6)
