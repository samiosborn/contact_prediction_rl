# tests/test_critic.py

import torch

from contact_prediction_rl.learning.models.critic import Critic


def test_critic_shapes():
    observation_dim = 12
    batch_size = 8

    critic = Critic(
        observation_dim=observation_dim,
        hidden_dims=[256, 128, 64],
        activation="elu",
        observation_normalisation=True,
    )

    # Create a batch of observations
    observations = torch.randn(
        batch_size,
        observation_dim,
    )

    # Update the running observation statistics
    critic.update_normalisation(observations)

    # Forward pass through the critic
    values = critic(observations)

    # Critic should output one scalar value per observation
    assert values.shape == (batch_size, 1)


def test_critic_loss_reaches_encoder():
    observation_dim = 12
    batch_size = 8

    critic = Critic(
        observation_dim=observation_dim,
        hidden_dims=[256, 128, 64],
        activation="elu",
        observation_normalisation=False,
    )

    # Create a batch of observations
    observations = torch.randn(
        batch_size,
        observation_dim,
    )

    # Forward pass through the critic
    values = critic(observations)

    # Use a simple scalar loss to test the gradient path
    loss = values.square().mean()
    loss.backward()

    # Every encoder parameter should receive a non-zero gradient
    for parameter in critic.observation_encoder.parameters():
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0

    # The final value head should also receive a non-zero gradient
    assert critic.value_head.weight.grad is not None
    assert critic.value_head.weight.grad.abs().sum() > 0


def test_critic_normalisation_updates():
    observation_dim = 12

    critic = Critic(
        observation_dim=observation_dim,
        hidden_dims=[256, 128, 64],
        activation="elu",
        observation_normalisation=True,
    )

    # Create observations used to update the running statistics
    observations = torch.randn(
        32,
        observation_dim,
    )

    # No observations should have been seen yet
    assert critic.normaliser.count.item() == 0

    # Update the running mean and variance
    critic.update_normalisation(observations)

    # The normaliser should now have processed the full batch
    assert critic.normaliser.count.item() == 32