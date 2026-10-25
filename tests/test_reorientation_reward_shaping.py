"""CPU checks for minimal reward design and metric-only observer wiring."""

import math
from types import SimpleNamespace

import pytest
import torch

from contact_prediction_rl.tasks.reorientation.config.hx5_d20_right.train_env_cfg import ReorientationTrainEnvCfg
from contact_prediction_rl.tasks.reorientation.diagnostics import ReorientationEpisodeMetrics
from contact_prediction_rl.tasks.reorientation.mdp import metrics as mdp_metrics, rewards
from contact_prediction_rl.tasks.reorientation.orientation import axis_angle_quat


def _goal(angle_deg: float) -> torch.Tensor:
    return axis_angle_quat(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([math.radians(angle_deg)]),
    )


class _Scene(dict):
    def __init__(self, current_quat: torch.Tensor):
        super().__init__()
        self.env_origins = torch.zeros(1, 3)
        self["object"] = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=torch.zeros(1, 3)),
                root_quat_w=SimpleNamespace(torch=current_quat),
                root_lin_vel_w=SimpleNamespace(torch=torch.zeros(1, 3)),
                root_ang_vel_w=SimpleNamespace(torch=torch.zeros(1, 3)),
            )
        )


def _env(current_quat: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(scene=_Scene(current_quat))


def test_orientation_reward_is_negative_absolute_3d_error():
    from contact_prediction_rl.tasks.reorientation.orientation import orientation_error_rad

    goal = _goal(15.0)
    env = SimpleNamespace(scene={"object": SimpleNamespace(data=SimpleNamespace(root_quat_w=SimpleNamespace(torch=_goal(0.0))))})
    goal_tuple = tuple(goal[0].tolist())

    at_start = rewards.orientation_reward(env, goal_tuple)
    assert at_start.item() == pytest.approx(-math.radians(15.0), abs=1e-4)

    env.scene["object"].data.root_quat_w.torch = goal
    assert rewards.orientation_reward(env, goal_tuple).item() == pytest.approx(0.0, abs=1e-4)

    env.scene["object"].data.root_quat_w.torch = _goal(30.0)
    assert rewards.orientation_reward(env, goal_tuple).item() == pytest.approx(-math.radians(15.0), abs=1e-4)

    env.scene["object"].data.root_quat_w.torch = _goal(60.0)
    assert rewards.orientation_reward(env, goal_tuple).item() == pytest.approx(-math.radians(45.0), abs=1e-4)
    assert torch.allclose(orientation_error_rad(goal, goal), torch.zeros(1), atol=1e-4)


def test_fixed_horizon_has_no_stable_hold_reward():
    cfg = ReorientationTrainEnvCfg()
    assert cfg.rewards.orientation.func is rewards.orientation_reward
    assert cfg.rewards.orientation.weight == pytest.approx(1.0)
    assert cfg.rewards.episode_metrics.func is mdp_metrics.ReorientationMetricsTerm
    assert cfg.rewards.episode_metrics.weight == pytest.approx(1.0)
    assert cfg.rewards.dropped.weight == pytest.approx(-2.0)
    assert cfg.rewards.episode_metrics.params["sustained_steps"] == 10
    assert cfg.rewards.episode_metrics.params["max_hold_displacement_m"] == pytest.approx(0.02)
    assert not hasattr(cfg.rewards, "stable_hold")
    assert not hasattr(cfg.terminations, "success")
    assert not hasattr(cfg.terminations, "dropped")


def test_metrics_observer_cannot_shape_reward(monkeypatch):
    env = _env(_goal(15.0))
    monkeypatch.setattr(mdp_metrics, "fingertip_patch_contact", lambda _: torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]))
    recorder = object.__new__(mdp_metrics.ReorientationMetricsTerm)
    recorder.tracker = ReorientationEpisodeMetrics(1)
    reward = recorder(
        env,
        tuple(_goal(15.0)[0].tolist()),
        (0.0, 0.0, 0.0),
    )
    assert reward.shape == (1,)
    assert reward.item() == 0.0
    assert recorder.tracker.steps.item() == 1
