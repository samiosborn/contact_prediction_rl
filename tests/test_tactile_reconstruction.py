# tests/test_tactile_reconstruction.py

"""Tests for the tactile reconstruction (TR) auxiliary loss and the auxiliary optimiser of the PPO implementation."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from contact_prediction_rl.learning.algorithms.ppo import PPO
from contact_prediction_rl.learning.models.auxiliary import TactileReconstruction

PROPRIO_DIM = 20
TACTILE_DIM = 12  # for example 3 steps of 4 fingertips
OBSERVATION_DIM = PROPRIO_DIM + TACTILE_DIM
ACTION_DIM = 5
NUM_ENVS = 16
LATENT_DIM = 8


def make_objective(pos_weight: float = 10.0) -> TactileReconstruction:
    return TactileReconstruction(LATENT_DIM, TACTILE_DIM, hidden_dims=[16, 16], pos_weight=pos_weight)


def test_the_target_is_the_tactile_tail_of_the_observation():
    objective = make_objective()
    latent = torch.randn(10, LATENT_DIM)
    tactile = (torch.rand(10, TACTILE_DIM) > 0.5).float()
    observations = torch.cat([torch.randn(10, PROPRIO_DIM), tactile], dim=1)

    loss, _ = objective.loss(observations, latent)

    expected = F.binary_cross_entropy_with_logits(
        objective.decoder(latent), tactile, pos_weight=torch.full((TACTILE_DIM,), 10.0)
    )
    assert loss.item() == pytest.approx(expected.item())

    # the proprioceptive part of the observation does not matter
    changed = observations.clone()
    changed[:, :PROPRIO_DIM] += 100.0
    assert objective.loss(changed, latent)[0].item() == pytest.approx(loss.item())


def test_a_missed_contact_costs_more_than_a_false_alarm():
    heavy, light = make_objective(10.0), make_objective(1.0)
    light.decoder.network.load_state_dict(heavy.decoder.network.state_dict())  # same network, other weight
    latent = torch.randn(32, LATENT_DIM)
    contacts = torch.cat([torch.randn(32, PROPRIO_DIM), torch.ones(32, TACTILE_DIM)], dim=1)  # every fingertip touching

    assert heavy.loss(contacts, latent)[0].item() > light.loss(contacts, latent)[0].item()


def make_cfg(with_aux: bool = True, architecture: str = "shared") -> dict:
    cfg = {
        "num_steps_per_env": 4,
        "architecture": architecture,
        "encoder": {"hidden_dims": [32, 16], "latent_dim": LATENT_DIM, "activation": "elu"},
        "actor": {"hidden_dims": [8], "activation": "elu", "initial_std": 0.2},
        "critic": {"hidden_dims": [8], "activation": "elu"},
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.002,
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "learning_rate": 1e-3,
            "optimizer": "adam",
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": False,
        },
    }
    if with_aux:
        cfg["auxiliary"] = {
            "objective": "tactile_reconstruction",
            "tactile_dim": TACTILE_DIM,
            "hidden_dims": [32, 32],
            "activation": "elu",
            "pos_weight": 10.0,
            "loss_weight": 1.0,
            "learning_rate": 3e-3,
        }
    return cfg


def observe() -> TensorDict:
    """Random proprioception, and a tactile tail that is a learnable function of it (contact when joint 0 is positive)"""
    proprio = torch.randn(NUM_ENVS, PROPRIO_DIM)
    tactile = (proprio[:, :1] > 0).float().repeat(1, TACTILE_DIM)
    return TensorDict({"policy": torch.cat([proprio, tactile], dim=1)}, batch_size=[NUM_ENVS])


def run_iteration(alg: PPO, obs: TensorDict) -> tuple[dict, TensorDict]:
    for _ in range(4):
        alg.act(obs)
        obs = observe()
        alg.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS, dtype=torch.bool), {})
    alg.compute_returns(obs)
    return alg.update(), obs


def build(cfg: dict) -> PPO:
    env = SimpleNamespace(num_envs=NUM_ENVS, num_actions=ACTION_DIM)
    return PPO.construct_algorithm(observe(), env, cfg, "cpu")


def test_the_auxiliary_optimiser_is_separate_and_not_clipped():
    alg = build(make_cfg())

    ppo_ids = {id(p) for g in alg.optimizer.param_groups for p in g["params"]}
    aux_ids = {id(p) for g in alg.aux_optimizer.param_groups for p in g["params"]}
    decoder_ids = {id(p) for p in alg.auxiliary.parameters()}
    encoder_ids = {id(p) for p in alg.actor.encoder.parameters()}
    clipped_ids = {id(p) for group in alg._clip_groups for p in group}

    assert decoder_ids == aux_ids - encoder_ids          # the decoder is trained by the auxiliary optimiser only
    assert encoder_ids <= aux_ids and encoder_ids <= ppo_ids  # the encoder by both
    assert not (decoder_ids & ppo_ids)
    assert not (decoder_ids & clipped_ids)               # and its gradients are not clipped
    assert alg.aux_optimizer.param_groups[0]["lr"] == 3e-3


def test_the_ppo_learning_rate_schedule_does_not_change_the_auxiliary_rate():
    torch.manual_seed(0)
    alg = build(make_cfg())
    obs = observe()

    for _ in range(3):
        _, obs = run_iteration(alg, obs)

    assert alg.aux_optimizer.param_groups[0]["lr"] == 3e-3


def test_the_reconstruction_loss_falls_on_a_learnable_signal():
    torch.manual_seed(0)
    alg = build(make_cfg())
    obs = observe()
    before = [p.detach().clone() for p in alg.auxiliary.parameters()]
    losses = []

    for _ in range(40):
        result, obs = run_iteration(alg, obs)
        losses.append(result["aux_loss"])

    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, alg.auxiliary.parameters()))
    assert sum(losses[-5:]) / 5 < 0.8 * sum(losses[:5]) / 5
    assert {"aux_loss", "aux_f1", "aux_positive_rate", "aux_predicted_rate"} <= set(result)


def test_without_an_auxiliary_objective_nothing_changes():
    alg = build(make_cfg(with_aux=False))

    assert alg.auxiliary is None and alg.aux_optimizer is None

    result, _ = run_iteration(alg, observe())
    assert not any(name.startswith("aux_") for name in result)


def test_an_auxiliary_objective_needs_the_shared_architecture():
    cfg = make_cfg(architecture="separate")
    cfg["actor"] = {"hidden_dims": [8], "latent_dim": LATENT_DIM, "activation": "elu"}
    cfg["critic"] = {"hidden_dims": [8, 8], "activation": "elu"}
    del cfg["encoder"]

    with pytest.raises(ValueError, match="shared"):
        build(cfg)


def test_the_auxiliary_state_is_saved_and_loaded():
    torch.manual_seed(0)
    trained = build(make_cfg())
    obs = observe()
    for _ in range(3):
        _, obs = run_iteration(trained, obs)

    state = trained.save()
    fresh = build(make_cfg())
    fresh.load(state, None, strict=True)

    for a, b in zip(trained.auxiliary.parameters(), fresh.auxiliary.parameters()):
        assert torch.equal(a, b)

    # a checkpoint without the objective cannot be loaded strictly into an agent that has one
    without = build(make_cfg(with_aux=False)).save()
    with pytest.raises(RuntimeError, match="no auxiliary"):
        fresh.load(without, None, strict=True)
