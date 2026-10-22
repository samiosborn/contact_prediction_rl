# tests/test_shared_agent.py

"""Tests for the shared-encoder agent of the tactile-RL paper and for the PPO changes that support it."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from contact_prediction_rl.learning.algorithms.ppo import PPO, _unique_parameters
from contact_prediction_rl.learning.models.actor import Actor
from contact_prediction_rl.learning.models.critic import Critic
from contact_prediction_rl.learning.models.shared import SharedActor, SharedCritic, SharedEncoder

OBSERVATION_DIM = 24
ACTION_DIM = 5


def make_agent(**actor_kwargs):
    encoder = SharedEncoder(OBSERVATION_DIM, hidden_dims=[64, 32], latent_dim=16)
    actor = SharedActor(encoder, ACTION_DIM, hidden_dims=[12, 8], **actor_kwargs)
    critic = SharedCritic(encoder, hidden_dims=[12, 8])
    return encoder, actor, critic


def test_encoder_follows_the_paper_layout():
    encoder = SharedEncoder(OBSERVATION_DIM, hidden_dims=[1024, 512], latent_dim=256)

    linear = [m for m in encoder.network if isinstance(m, nn.Linear)]
    norms = [m for m in encoder.network if isinstance(m, nn.LayerNorm)]
    activations = [m for m in encoder.network if isinstance(m, nn.ELU)]

    assert [(m.in_features, m.out_features) for m in linear] == [(OBSERVATION_DIM, 1024), (1024, 512), (512, 256)]
    assert len(norms) == len(activations) == 3
    assert encoder(torch.randn(7, OBSERVATION_DIM)).shape == (7, 256)


def test_heads_follow_the_paper_layout_and_the_mean_is_bounded_by_tanh():
    _, actor, critic = make_agent()
    observations = torch.randn(32, OBSERVATION_DIM) * 10.0

    mean = actor(observations)  # deterministic action
    values = critic(observations)

    assert mean.shape == (32, ACTION_DIM)
    assert mean.abs().max() <= 1.0
    assert isinstance(actor.action_head[-1], nn.Tanh)
    assert values.shape == (32, 1)


def test_the_actor_keeps_the_interface_that_ppo_uses():
    _, actor, _ = make_agent()
    observations = torch.randn(16, OBSERVATION_DIM)

    actions = actor(observations, stochastic_output=True)
    log_prob = actor.get_output_log_prob(actions)
    mean, std = actor.output_distribution_params
    kl = actor.get_kl_divergence((mean.detach(), std.detach()), (mean, std))

    assert actions.shape == (16, ACTION_DIM)
    assert log_prob.shape == actor.output_entropy.shape == kl.shape == (16,)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)
    assert actor.encode(observations).shape == (16, 16)


def test_small_initial_actions_option():
    _, default, _ = make_agent()
    _, small, _ = make_agent(action_head_init_scale=0.01)
    observations = torch.randn(64, OBSERVATION_DIM)

    assert default(observations).abs().mean() > 5 * small(observations).abs().mean()


def test_actor_and_critic_share_the_encoder_parameters():
    encoder, actor, critic = make_agent()

    shared = {id(p) for p in encoder.parameters()}

    assert shared <= {id(p) for p in actor.parameters()}
    assert shared <= {id(p) for p in critic.parameters()}
    assert len(_unique_parameters(actor.parameters(), critic.parameters())) == len(
        list(encoder.parameters())
    ) + len(list(actor.action_head.parameters())) + 1 + len(list(critic.value_head.parameters()))


@pytest.mark.parametrize("loss_of", ["actor", "critic"])
def test_both_heads_send_gradients_into_the_encoder(loss_of):
    encoder, actor, critic = make_agent()
    observations = torch.randn(16, OBSERVATION_DIM)

    loss = actor(observations).pow(2).sum() if loss_of == "actor" else critic(observations).pow(2).sum()
    loss.backward()

    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.network.parameters())


def test_the_shared_normaliser_is_updated_once_per_step():
    encoder, actor, critic = make_agent()
    observations = torch.randn(10, OBSERVATION_DIM)

    actor.update_normalisation(observations)
    critic.update_normalisation(observations)

    assert encoder.normaliser.count.item() == 10  # not 20


def dummy_ppo(actor, critic):
    return PPO(actor=actor, critic=critic, storage=None, num_mini_batches=1, num_learning_epochs=1)


def test_ppo_lists_shared_parameters_once_and_clips_them_together():
    encoder, actor, critic = make_agent()

    ppo = dummy_ppo(actor, critic)

    listed = [p for group in ppo.optimizer.param_groups for p in group["params"]]
    assert len(listed) == len({id(p) for p in listed})
    assert len(ppo._clip_groups) == 1
    assert {id(p) for p in encoder.parameters()} <= {id(p) for p in ppo._clip_groups[0]}


def test_ppo_with_separate_networks_is_unchanged():
    actor = Actor(OBSERVATION_DIM, ACTION_DIM, hidden_dims=[32], latent_dim=16, activation="elu")
    critic = Critic(OBSERVATION_DIM, hidden_dims=[32, 16], activation="elu")

    ppo = dummy_ppo(actor, critic)

    # same parameters, in the same order as before (actor, then critic), clipped as two separate groups
    expected = [*actor.parameters(), *critic.parameters()]
    listed = [p for group in ppo.optimizer.param_groups for p in group["params"]]
    assert [id(p) for p in listed] == [id(p) for p in expected]
    assert [len(g) for g in ppo._clip_groups] == [len(list(actor.parameters())), len(list(critic.parameters()))]


def small_cfg(architecture: str) -> dict:
    cfg = {
        "num_steps_per_env": 4,
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
    if architecture == "shared":
        cfg.update(
            architecture="shared",
            encoder={"hidden_dims": [32, 16], "latent_dim": 8, "activation": "elu"},
            actor={"hidden_dims": [8], "activation": "elu", "initial_std": 0.2},
            critic={"hidden_dims": [8], "activation": "elu"},
        )
    else:
        cfg.update(
            actor={"hidden_dims": [16], "latent_dim": 8, "activation": "elu", "observation_normalisation": True},
            critic={"hidden_dims": [16, 8], "activation": "elu", "observation_normalisation": True},
        )
    return cfg


@pytest.mark.parametrize("architecture", ["shared", "separate"])
def test_a_ppo_iteration_runs_and_changes_the_networks(architecture):
    torch.manual_seed(0)
    num_envs = 8
    env = SimpleNamespace(num_envs=num_envs, num_actions=ACTION_DIM)

    def observe():
        return TensorDict({"policy": torch.randn(num_envs, OBSERVATION_DIM)}, batch_size=[num_envs])

    obs = observe()
    alg = PPO.construct_algorithm(obs, env, small_cfg(architecture), "cpu")
    before = [p.detach().clone() for p in alg.actor.parameters()]

    for _ in range(4):
        alg.act(obs)
        obs = observe()
        alg.process_env_step(obs, torch.randn(num_envs), torch.zeros(num_envs, dtype=torch.bool), {})

    alg.compute_returns(obs)
    losses = alg.update()

    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, alg.actor.parameters()))
    assert isinstance(alg.get_policy(), SharedActor if architecture == "shared" else Actor)


def test_an_unknown_architecture_is_rejected():
    env = SimpleNamespace(num_envs=4, num_actions=ACTION_DIM)
    obs = TensorDict({"policy": torch.randn(4, OBSERVATION_DIM)}, batch_size=[4])
    cfg = small_cfg("separate") | {"architecture": "other"}

    with pytest.raises(ValueError, match="Unknown architecture"):
        PPO.construct_algorithm(obs, env, cfg, "cpu")
