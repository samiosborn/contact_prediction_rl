"""CPU tests of Reorientation-specific state measurements."""

import pytest
import torch

from contact_prediction_rl.tasks.reorientation.diagnostics import ReorientationTracker, measure_reorientation
from contact_prediction_rl.tasks.reorientation.orientation import axis_angle_quat


def test_rotation_diagnostics_detect_retained_and_lost_grasp():
    n = 2
    initial = torch.tensor([0., 0., 0., 1.])
    axis = torch.tensor([0., 0., 1.])
    goal = axis_angle_quat(axis.unsqueeze(0), torch.tensor([torch.pi / 12]))[0]
    positions = torch.tensor([[0., 0., 0.], [0.05, 0., 0.]])
    quats = goal.repeat(n, 1)
    contacts = torch.tensor([[1., 1., 0., 0., 0.], [1., 0., 0., 0., 0.]])
    metrics = measure_reorientation(
        positions, quats, contacts, start_pos_w=torch.zeros(n, 3),
        start_quat_w=initial, goal_quat_w=goal, axis_world=axis,
    )
    assert metrics["goal_error_deg"].abs().max().item() == pytest.approx(0., abs=1e-3)
    assert metrics["signed_twist_deg"].tolist() == pytest.approx([15., 15.], abs=1e-3)
    assert metrics["goal_progress_deg"].tolist() == pytest.approx([15., 15.], abs=1e-3)
    assert metrics["diagnostic_success"].tolist() == [True, False]

    tracker = ReorientationTracker(n)
    for _ in range(10):
        tracker.observe(metrics)
    summary = tracker.summary()
    assert summary["held_goal_10_steps_fraction"] == pytest.approx(0.5)
    assert summary["ever_reached_goal_fraction"] == pytest.approx(0.5)
    assert summary["mean_max_cube_displacement_m"] == pytest.approx(0.025)


def test_bad_contact_dimensions_fail():
    with pytest.raises(ValueError):
        measure_reorientation(
            torch.zeros(2, 3), torch.zeros(2, 4), torch.zeros(2, 4),
            start_pos_w=torch.zeros(2, 3), start_quat_w=torch.zeros(4),
            goal_quat_w=torch.zeros(4), axis_world=torch.tensor([1., 0., 0.]),
        )
