"""CPU regression checks for new Reorientation reward design and task wiring."""

import math
from types import SimpleNamespace

import pytest
import torch

from contact_prediction_rl.tasks.reorientation.config.hx5_d20_right.train_env_cfg import ReorientationTrainEnvCfg
from contact_prediction_rl.tasks.reorientation.mdp import rewards
from contact_prediction_rl.tasks.reorientation.orientation import axis_angle_quat


class _Scene(dict):
    def __init__(self, cube_quat):
        super().__init__()
        self.env_origins = torch.zeros(1, 3)
        self["object"] = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=torch.zeros(1, 3)),
                root_quat_w=SimpleNamespace(torch=cube_quat),
            )
        )


def test_dense_progress_only_when_grasp_is_held(monkeypatch):
    initial = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    goal = axis_angle_quat(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([math.radians(15.0)]))
    env = SimpleNamespace(scene=_Scene(initial))
    monkeypatch.setattr(rewards, "fingertip_patch_contact", lambda _: torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]))
    args = (tuple(goal[0].tolist()), (0.0, 0.0, 0.0))

    at_start = rewards.orientation_progress_reward(env, *args)
    env.scene["object"].data.root_quat_w.torch = goal
    at_goal = rewards.orientation_progress_reward(env, *args)
    assert at_start.item() == pytest.approx(0.0, abs=1e-3)
    assert at_goal.item() == pytest.approx(1.0, abs=1e-3)

    monkeypatch.setattr(rewards, "fingertip_patch_contact", lambda _: torch.zeros(1, 5))
    assert rewards.orientation_progress_reward(env, *args).item() == 0.0
    env.scene["object"].data.root_pos_w.torch[0, 0] = 0.05
    assert rewards.orientation_progress_reward(env, *args).item() == 0.0


def test_fixed_horizon_rewards_and_metric_settings():
    cfg = ReorientationTrainEnvCfg()
    assert cfg.rewards.orientation.func is rewards.orientation_progress_reward
    assert cfg.rewards.stable_hold.func is rewards.StableGoalReward
    assert cfg.rewards.orientation.weight == pytest.approx(1.0)
    assert cfg.rewards.stable_hold.weight == pytest.approx(4.0)
    assert cfg.rewards.dropped.weight == pytest.approx(-2.0)
    assert cfg.rewards.stable_hold.params["sustained_steps"] == 10
    assert cfg.rewards.stable_hold.params["max_hold_displacement_m"] == pytest.approx(0.02)
    assert not hasattr(cfg.terminations, "success") and not hasattr(cfg.terminations, "dropped")
