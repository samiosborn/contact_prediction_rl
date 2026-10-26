"""CPU coverage for task-agnostic rollout and control diagnostics."""

import pytest
import torch

from contact_prediction_rl.diagnostics.control import action_statistics, joint_tracking_statistics
from contact_prediction_rl.diagnostics.rollout import EpisodeTracker


def test_interleaved_episodes_are_reset_independently():
    tracker = EpisodeTracker(2)
    assert tracker.summary() == {}
    tracker.update(torch.tensor([1., 2.]), torch.tensor([False, False]), success=torch.tensor([True, False]), goal_error=torch.tensor([3., 5.]))
    completed = tracker.update(torch.tensor([3., 4.]), torch.tensor([True, False]), success=torch.tensor([False, True]), goal_error=torch.tensor([2., 4.]))
    assert len(completed) == 1
    assert (completed[0].env_id, completed[0].length, completed[0].reward) == (0, 2, 4.)
    assert completed[0].success_ever and not completed[0].success_at_end
    assert completed[0].minimum_goal_error == pytest.approx(2.)
    assert completed[0].longest_success_streak == 1

    completed = tracker.update(torch.tensor([5., 6.]), torch.tensor([False, True]), time_outs=torch.tensor([False, True]), success=torch.tensor([False, True]), goal_error=torch.tensor([9., 3.]))
    assert completed[0].env_id == 1
    assert completed[0].reward == pytest.approx(12.)
    assert completed[0].success_at_end and completed[0].timed_out
    assert completed[0].minimum_goal_error == pytest.approx(3.)
    completed = tracker.update(torch.tensor([7., 8.]), torch.tensor([True, False]), goal_error=torch.tensor([8., 7.]))
    assert completed[0].length == 2
    assert completed[0].reward == pytest.approx(12.)
    assert completed[0].first_goal_error == pytest.approx(9.)
    assert tracker.summary()["episodes"] == 3


def test_mixed_success_streaks_and_bad_shapes():
    tracker = EpisodeTracker(1)
    for hit in (True, True, False, True, True, True):
        tracker.update(torch.ones(1), torch.zeros(1, dtype=torch.bool), success=torch.tensor([hit]))
    record = tracker.update(torch.ones(1), torch.ones(1, dtype=torch.bool), success=torch.tensor([False]))[0]
    assert record.longest_success_streak == 3
    with pytest.raises(ValueError):
        tracker.update(torch.ones(2), torch.ones(1, dtype=torch.bool))


def test_control_diagnostics_do_not_assume_action_units():
    a = torch.tensor([[0., 2.], [-1., 1.]])
    stats = action_statistics(a, action_limit=2.)
    assert stats["max_abs_action"] == pytest.approx(2.)
    assert stats["fraction_at_action_limit"] == pytest.approx(.25)
    joints = torch.tensor([[0.1, 0.5]])
    targets = torch.tensor([[0.0, 0.6]])
    limits = torch.tensor([[0., 1.], [0., 1.]])
    result = joint_tracking_statistics(joints, targets, limits=limits)
    assert result["mean_abs_joint_target_error_rad"] == pytest.approx(.1)
    assert result["minimum_joint_limit_margin_rad"] == pytest.approx(.1)
    with pytest.raises(ValueError):
        joint_tracking_statistics(joints, targets[:, :1])
