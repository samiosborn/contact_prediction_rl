# tests/test_actor.py

import torch

from contact_prediction_rl.learning.models.actor import Actor


def test_actor_shapes():
    observation_dim = 12
    action_dim = 4
    batch_size = 8

    actor = Actor(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=[256, 128],
        latent_dim=64,
        activation="elu",
        initial_std=0.2,
        observation_normalisation=False,
    )

    # Create a batch of observations
    observations = torch.randn(
        batch_size,
        observation_dim,
    )

    # Encode observations into the latent representation
    latent = actor.encode(observations)

    # Deterministic forward pass returns the Gaussian mean
    actions = actor(observations)

    assert latent.shape == (
        batch_size,
        64,
    )

    assert actions.shape == (
        batch_size,
        action_dim,
    )


def test_actor_loss_reaches_encoder():
    observation_dim = 12
    action_dim = 4
    batch_size = 8

    actor = Actor(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=[256, 128],
        latent_dim=64,
        activation="elu",
        initial_std=0.2,
        observation_normalisation=False,
    )

    # Create a batch of observations
    observations = torch.randn(
        batch_size,
        observation_dim,
    )

    # Use the deterministic mean action so gradients pass through the policy network
    actions = actor(observations)

    # Simple scalar loss for testing the gradient path
    loss = actions.square().mean()
    loss.backward()

    # Every encoder parameter should receive a non-zero gradient
    for parameter in actor.observation_encoder.parameters():
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0

    # The action head should also receive gradients
    assert actor.action_head.weight.grad is not None
    assert actor.action_head.weight.grad.abs().sum() > 0


def test_actor_stochastic_shapes():
    observation_dim = 12
    action_dim = 4
    batch_size = 8

    actor = Actor(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=[256, 128],
        latent_dim=64,
        activation="elu",
        initial_std=0.2,
        observation_normalisation=False,
    )

    # Create a batch of observations
    observations = torch.randn(
        batch_size,
        observation_dim,
    )

    # Sample actions from the Gaussian policy
    actions = actor(
        observations,
        stochastic_output=True,
    )

    assert actions.shape == (
        batch_size,
        action_dim,
    )

    # Log probability should give one scalar per action vector
    log_prob = actor.get_output_log_prob(actions)

    assert log_prob.shape == (batch_size,)

    # Entropy should also give one scalar per action vector
    assert actor.output_entropy.shape == (batch_size,)


def test_actor_distribution_parameters():
    observation_dim = 12
    action_dim = 4
    batch_size = 8
    initial_std = 0.2

    actor = Actor(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=[256, 128],
        latent_dim=64,
        activation="elu",
        initial_std=initial_std,
        observation_normalisation=False,
    )

    # Create the current action distribution
    observations = torch.randn(
        batch_size,
        observation_dim,
    )

    actor(observations)

    mean, std = actor.output_distribution_params

    assert mean.shape == (
        batch_size,
        action_dim,
    )

    assert std.shape == (
        batch_size,
        action_dim,
    )

    assert torch.allclose(
        std,
        torch.full_like(
            std,
            initial_std,
        ),
    )


def test_actor_normalisation_updates():
    observation_dim = 12
    action_dim = 4

    actor = Actor(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=[256, 128],
        latent_dim=64,
        activation="elu",
        initial_std=0.2,
        observation_normalisation=True,
    )

    # Create observations used to update the running statistics
    observations = torch.randn(
        32,
        observation_dim,
    )

    # No observations should have been seen yet
    assert actor.normaliser.count.item() == 0

    # Update the running mean and variance
    actor.update_normalisation(observations)

    # The normaliser should now have processed the full batch
    assert actor.normaliser.count.item() == 32
