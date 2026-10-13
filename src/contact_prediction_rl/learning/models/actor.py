# src/contact_prediction_rl/learning/models/actor.py

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from contact_prediction_rl.learning.models.normalisation import ObservationNormaliser
from contact_prediction_rl.learning.models.observation_encoder import ObservationEncoder


class Actor(nn.Module):
    """Gaussian policy with an explicit observation encoder and action head"""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        latent_dim: int,
        activation: str,
        initial_std: float = 0.2,
        observation_normalisation: bool = False,
    ) -> None:
        super().__init__()

        if initial_std <= 0:
            raise ValueError("Initial action standard deviation must be positive")

        # Observation normalisation
        self.normaliser = ObservationNormaliser(observation_dim) if observation_normalisation else nn.Identity()

        # Encode observations into the shared latent representation z
        self.observation_encoder = ObservationEncoder(
            observation_dim=observation_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
        )

        # Map the latent representation to the mean action
        self.action_head = nn.Linear(
            latent_dim,
            action_dim,
        )

        # State-independent learned standard deviation
        self.std = nn.Parameter(
            torch.full(
                (action_dim,),
                initial_std,
            )
        )

        # Current action distribution
        self._distribution: Normal | None = None

    def encode(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observations into the latent representation"""

        observations = self.normaliser(observations)

        return self.observation_encoder(observations)

    def update_distribution(
        self,
        observations: torch.Tensor,
    ) -> None:
        """Construct the Gaussian action distribution"""

        latent = self.encode(observations)

        mean = self.action_head(latent)

        std = self.std.expand_as(mean)

        self._distribution = Normal(
            mean,
            std,
        )

    def forward(
        self,
        observations: torch.Tensor,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Return sampled or deterministic actions"""

        self.update_distribution(observations)

        if stochastic_output:
            return self.distribution.sample()

        return self.distribution.mean

    def get_output_log_prob(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return the log probability of each action"""

        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_kl_divergence(
        self,
        old_distribution_params: tuple[torch.Tensor, torch.Tensor],
        new_distribution_params: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Return KL divergence from the old policy to the new policy"""

        old_mean, old_std = old_distribution_params
        new_mean, new_std = new_distribution_params

        old_distribution = Normal(
            old_mean,
            old_std,
        )

        new_distribution = Normal(
            new_mean,
            new_std,
        )

        return kl_divergence(
            old_distribution,
            new_distribution,
        ).sum(dim=-1)

    def update_normalisation(
        self,
        observations: torch.Tensor,
    ) -> None:
        """Update observation normalisation statistics"""

        if isinstance(
            self.normaliser,
            ObservationNormaliser,
        ):
            self.normaliser.update(observations)

    @property
    def distribution(
        self,
    ) -> Normal:
        """Return the current action distribution"""

        if self._distribution is None:
            raise RuntimeError("Action distribution has not been initialised")

        return self._distribution

    @property
    def output_distribution_params(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and standard deviation of the current policy"""

        return (
            self.distribution.mean,
            self.distribution.stddev,
        )

    @property
    def output_entropy(
        self,
    ) -> torch.Tensor:
        """Return policy entropy summed across action dimensions"""

        return self.distribution.entropy().sum(dim=-1)

    @property
    def output_std(
        self,
    ) -> torch.Tensor:
        """Return the learned action standard deviation"""

        return self.std
