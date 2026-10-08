# tests/test_actor.py

import torch
from tensordict import TensorDict

import contact_prediction_rl.learning.configs.rsl_rl as CONFIG
from contact_prediction_rl.learning.models.actor import Actor


def test_actor_shapes():
    observation_dim = 12
    action_dim = 4
    batch_size = 8

    observations = TensorDict(
        {
            "policy": torch.randn(batch_size, observation_dim),
        },
        batch_size=[batch_size],
    )

    actor = Actor(
        obs=observations,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=action_dim,
        hidden_dims=[
            *CONFIG.ACTOR_ENCODER_HIDDEN_DIMS,
            CONFIG.ACTOR_LATENT_DIM,
        ],
        activation=CONFIG.ACTOR_ACTIVATION,
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": CONFIG.ACTOR_INIT_STD,
        },
    )

    latent = actor.encode(observations)
    actions = actor(observations)

    assert latent.shape == (batch_size, CONFIG.ACTOR_LATENT_DIM)
    assert actions.shape == (batch_size, action_dim)


def test_actor_loss_reaches_encoder():
    observation_dim = 12
    action_dim = 4
    batch_size = 8

    observations = TensorDict(
        {
            "policy": torch.randn(batch_size, observation_dim),
        },
        batch_size=[batch_size],
    )

    actor = Actor(
        obs=observations,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=action_dim,
        hidden_dims=[
            *CONFIG.ACTOR_ENCODER_HIDDEN_DIMS,
            CONFIG.ACTOR_LATENT_DIM,
        ],
        activation=CONFIG.ACTOR_ACTIVATION,
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": CONFIG.ACTOR_INIT_STD,
        },
    )

    actions = actor(observations)
    loss = actions.square().mean()

    loss.backward()

    for parameter in actor.observation_encoder.parameters():
        assert parameter.grad is not None