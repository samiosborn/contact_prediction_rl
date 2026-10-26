"""Unit tests for shared success predicates and fixed-horizon episode metrics."""

import math

import pytest
import torch

from contact_prediction_rl.tasks.reorientation.diagnostics import ReorientationEpisodeMetrics
from contact_prediction_rl.tasks.reorientation.metrics import (
    ReorientationSuccessThresholds,
    opposing_fingertip_contact,
    reorientation_step_metrics,
)
from contact_prediction_rl.tasks.reorientation.orientation import axis_angle_quat


def _batch(*, angle_deg: float = 15.0, displacement_m: float = 0.0, linear_speed: float = 0.0, angular_speed: float = 0.0, contact: bool = True):
    cube_pos = torch.tensor([[displacement_m, 0.0, 0.0]])
    goal = axis_angle_quat(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([math.radians(15.0)]))[0]
    cube_quat = axis_angle_quat(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([math.radians(angle_deg)]))
    labels = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]) if contact else torch.zeros(1, 5)
    return reorientation_step_metrics(
        cube_pos, cube_quat, torch.tensor([[linear_speed, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, angular_speed]]), labels,
        start_pos_w=torch.zeros(1, 3), goal_quat_w=goal, thresholds=ReorientationSuccessThresholds(),
    )


def test_success_predicates_reject_transient_and_lost_grasp():
    assert opposing_fingertip_contact(torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]])).tolist() == [True, False]
    assert _batch()["stable_goal"].tolist() == [True]
    for state in (
        _batch(displacement_m=0.03),
        _batch(linear_speed=0.2),
        _batch(angular_speed=1.0),
        _batch(contact=False),
        _batch(angle_deg=30.0),
    ):
        assert state["stable_goal"].tolist() == [False]
    assert _batch(displacement_m=0.05)["object_dropped"].tolist() == [True]


def test_config_rejects_inverted_or_invalid_thresholds():
    with pytest.raises(ValueError):
        ReorientationSuccessThresholds(max_hold_displacement_m=0.05)
    with pytest.raises(ValueError):
        ReorientationSuccessThresholds(sustained_steps=0)


def test_streak_reset_and_partial_parallel_episode():
    tracker = ReorientationEpisodeMetrics(2, sustained_steps=3)
    yes = _batch()
    no = _batch(angular_speed=1.0)

    def emit(second_hit: bool):
        frame = {name: torch.cat((value, (yes if second_hit else no)[name]), dim=0) for name, value in yes.items()}
        tracker.observe(frame)

    emit(True)
    emit(True)
    emit(False)
    a = tracker.summarise_completed(torch.tensor([0, 1]))
    assert a["Metrics/sustained_success_rate"] == pytest.approx(0.5)
    assert a["Metrics/stable_goal_fraction"] == pytest.approx(5.0 / 6.0)
    assert a["Metrics/final_stable_goal_rate"] == pytest.approx(0.5)

    tracker.reset(torch.tensor([0]))
    assert tracker.summarise_completed(torch.tensor([0])) == {}
    assert tracker.steps.tolist() == [0, 3]
    emit(True)
    assert tracker.steps.tolist() == [1, 4]
    assert tracker.summarise_completed(torch.tensor([0]))["Metrics/sustained_success_rate"] == 0.0
    assert tracker.summarise_completed(torch.tensor([1]))["Metrics/sustained_success_rate"] == 0.0
    emit(True)
    emit(True)
    assert tracker.summarise_completed(torch.tensor([1]))["Metrics/sustained_success_rate"] == 1.0


def test_no_success_for_passing_goal_once_then_losing_contact():
    tracker = ReorientationEpisodeMetrics(1, sustained_steps=10)
    tracker.observe(_batch())
    for _ in range(11):
        tracker.observe(_batch(contact=False))
    result = tracker.summarise_completed(torch.tensor([0]))
    assert result["Metrics/transient_goal_rate"] == 1.0
    assert result["Metrics/sustained_success_rate"] == 0.0
    assert result["Metrics/stable_goal_fraction"] == pytest.approx(1.0 / 12.0)



def test_training_task_only_terminates_on_fixed_horizon():
    from contact_prediction_rl.tasks.reorientation.config.hx5_d20_right.train_env_cfg import ReorientationTrainEnvCfg

    cfg = ReorientationTrainEnvCfg()
    assert cfg.episode_length_s == 6.0
    assert cfg.is_finite_horizon
    assert cfg.terminations.time_out.time_out is True
    assert not hasattr(cfg.terminations, "success")
    assert not hasattr(cfg.terminations, "dropped")
    assert cfg.scene.num_envs >= 1

def test_inference_mode_observe_then_external_partial_reset():
    """Metrics stay mutable across inference-mode steps and ordinary partial resets."""
    tracker = ReorientationEpisodeMetrics(2, sustained_steps=3)
    names = (
        "steps", "near_goal_steps", "stable_goal_steps", "grasp_steps",
        "dropped_steps", "streak", "best_streak", "minimum_goal_error_rad",
        "final_goal_error_rad", "max_displacement_m", "final_stable_goal",
    )
    buffers = {name: getattr(tracker, name) for name in names}
    state = _batch()

    with torch.inference_mode():
        batched = {name: torch.cat((value, value), dim=0) for name, value in state.items()}
        tracker.observe(batched)
        tracker.observe(batched)

    assert tracker.steps.tolist() == [2, 2]
    assert tracker.streak.tolist() == [2, 2]
    assert all(getattr(tracker, name) is buffer for name, buffer in buffers.items())
    assert all(not buffer.is_inference() for buffer in buffers.values())

    # This reset happens outside inference mode, exactly as in validate_training_reset.py.
    tracker.reset(torch.tensor([0]))
    assert tracker.steps.tolist() == [0, 2]
    assert tracker.streak.tolist() == [0, 2]
    assert tracker.best_streak.tolist() == [0, 2]
    assert torch.isinf(tracker.minimum_goal_error_rad[0])
    assert torch.isfinite(tracker.minimum_goal_error_rad[1])
    assert tracker.final_stable_goal.tolist() == [False, True]

    with torch.inference_mode():
        tracker.observe(batched)
    assert tracker.steps.tolist() == [1, 3]
    assert tracker.best_streak.tolist() == [1, 3]
    tracker.reset(torch.tensor([0, 1]))
    assert tracker.steps.tolist() == [0, 0]
    assert tracker.best_streak.tolist() == [0, 0]
    assert all(getattr(tracker, name) is buffer for name, buffer in buffers.items())
