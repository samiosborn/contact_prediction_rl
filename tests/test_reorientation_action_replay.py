"""CPU checks for scripted trajectories and exact inversion of joint-position action scaling."""

import pytest
import torch

from contact_prediction_rl.tasks.reorientation.feasibility import apply_rotation_pattern
from contact_prediction_rl.tasks.reorientation.replay import rotation_waypoints, targets_to_actions


def test_offset_ramp_matches_feasibility_targets():
    names = [f"finger_r_joint{i}" for i in range(20, 0, -1)]
    closed = [0.1 * i for i in range(20)]
    points = rotation_waypoints(closed, names, controller="offset", pattern="thumb_index",
                               amplitude_rad=0.6, rotate_steps=150, ramp_steps=90)
    expected = apply_rotation_pattern(closed, names, "thumb_index", 0.6)
    assert len(points) == 150
    assert points[0] == pytest.approx([a + (b - a) / 90 for a, b in zip(closed, expected, strict=True)])
    assert points[89] == pytest.approx(expected)
    assert points[-1] == pytest.approx(expected)


def test_gait_waypoints_return_to_snapshot_posture_and_validate_counts():
    names = [f"finger_r_joint{i}" for i in range(1, 21)]
    closed = [0.5] * 20
    points = rotation_waypoints(closed, names, controller="gait", pattern="spread", amplitude_rad=-0.6)
    assert len(points) == 150
    assert points[-1] == pytest.approx(closed)
    assert any(any(abs(v - a) > 0.1 for v, a in zip(p, closed, strict=True)) for p in points)
    with pytest.raises(ValueError):
        rotation_waypoints(closed, names, controller="gait", pattern="spread", amplitude_rad=0.2, rotate_steps=4)


def test_live_action_offset_and_scale_round_trip_for_arbitrary_joint_order():
    desired = torch.tensor([[1., -0.4], [0.8, 0.2]])
    offset = torch.tensor([[0.5, 0.5], [0.4, -0.1]])
    scale = torch.tensor([[0.25, 0.5]])
    action = targets_to_actions(desired, offset=offset, scale=scale)
    torch.testing.assert_close(action * scale + offset, desired)
    assert action[0, 0].item() == pytest.approx(2.0)
    with pytest.raises(ValueError):
        targets_to_actions(desired, offset=offset, scale=torch.zeros(2))


def test_scalar_action_scale_and_offset_are_supported():
    desired = torch.tensor([[0.6, -0.4]])
    actions = targets_to_actions(desired, offset=0.1, scale=0.25)
    torch.testing.assert_close(actions, torch.tensor([[2.0, -2.0]]))
