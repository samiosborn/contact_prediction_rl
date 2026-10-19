# tests/test_contact_prediction.py

"""CPU tests of the auxiliary model, gradient flow and matched PPO variants."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from contact_prediction_rl.learning.algorithms.ppo import PPO
from contact_prediction_rl.learning.models.actor import Actor
from contact_prediction_rl.learning.models.contact_prediction import ContactPredictionHead


def _obs():
    return TensorDict(
        {"policy": torch.randn(8, 12), "contact": torch.randint(0, 2, (8, 5)).float()},
        batch_size=[8],
    )


def _cfg(weight):
    return {
        "num_steps_per_env": 2,
        "actor": {"hidden_dims": [32], "latent_dim": 16, "activation": "elu", "observation_normalisation": False},
        "critic": {"hidden_dims": [32, 16], "activation": "elu", "observation_normalisation": False},
        "algorithm": {},
        "contact_prediction": {
            "hidden_dims": [16], "activation": "elu", "loss_weight": weight, "pos_weight": 2.0,
        },
    }


def test_head_bce_and_shared_encoder_gradients():
    actor = Actor(12, 4, [32], 16, "elu", observation_normalisation=False)
    head = ContactPredictionHead(16, 5, [16], "elu", pos_weight=2)
    obs = _obs()
    loss, metrics = head.compute_loss(actor.encode(obs["policy"]), obs["contact"])
    assert head(actor.encode(obs["policy"])).shape == (8, 5)
    assert torch.isfinite(loss) and loss > 0
    assert 0 <= metrics["contact_f1"] <= 1
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.observation_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert actor.action_head.weight.grad is None
    with pytest.raises(ValueError, match="binary labels"):
        head.compute_loss(actor.encode(obs["policy"]), torch.zeros(8, 4))


def test_fixed_architecture_only_aux_coefficient_changes():
    obs = _obs()
    env = SimpleNamespace(num_envs=8, num_actions=4)
    torch.manual_seed(42)
    baseline = PPO.construct_algorithm(obs, env, _cfg(0.0), device="cpu")
    torch.manual_seed(42)
    supervised = PPO.construct_algorithm(obs, env, _cfg(0.1), device="cpu")
    assert baseline.contact_head is not None
    assert supervised.contact_head is not None
    assert baseline.contact_loss_weight == 0.0
    assert supervised.contact_loss_weight == pytest.approx(.1)
    for a, b in zip(baseline.actor.parameters(), supervised.actor.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    for a, b in zip(baseline.contact_head.parameters(), supervised.contact_head.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    # No contact information is passed into the deployed actor.
    assert supervised.get_policy()(obs["policy"]).shape == (8, 4)


def test_aux_head_checkpoint_and_older_tasks_without_contact():
    obs = _obs()
    env = SimpleNamespace(num_envs=8, num_actions=4)
    first = PPO.construct_algorithm(obs, env, _cfg(.1), "cpu")
    saved = first.save()
    assert saved["contact_head_state_dict"] is not None
    second = PPO.construct_algorithm(obs, env, _cfg(.1), "cpu")
    second.load(saved, load_cfg=None, strict=True)
    for a, b in zip(first.contact_head.parameters(), second.contact_head.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="privileged"):
        PPO.construct_algorithm(TensorDict({"policy": obs["policy"]}, batch_size=[8]), env, _cfg(.1), "cpu")
    plain_cfg = _cfg(0.0)
    plain_cfg.pop("contact_prediction")
    ordinary = PPO.construct_algorithm(TensorDict({"policy": obs["policy"]}, batch_size=[8]), env, plain_cfg, "cpu")
    assert ordinary.contact_head is None
