# src/contact_prediction_rl/learning/models/critic.py

import torch
import torch.nn as nn

from contact_prediction_rl.learning.models.normalisation import ObservationNormaliser
from contact_prediction_rl.learning.models.observation_encoder import ObservationEncoder


class Critic(nn.Module):
    """Critic network for estimating state values"""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        observation_normalisation: bool = False,
    ) -> None:
        super().__init__()

        if len(hidden_dims) < 1:
            raise ValueError("Critic requires at least one hidden dimension")

        self.normaliser = ObservationNormaliser(observation_dim) if observation_normalisation else nn.Identity()

        encoder_hidden_dims = list(hidden_dims[:-1])
        latent_dim = hidden_dims[-1]

        self.observation_encoder = ObservationEncoder(
            observation_dim=observation_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
        )

        self.value_head = nn.Linear(
            latent_dim,
            1,
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        observations = self.normaliser(observations)
        latent = self.observation_encoder(observations)

        return self.value_head(latent)

    def update_normalisation(
        self,
        observations: torch.Tensor,
    ) -> None:
        """Update observation normalisation statistics"""

        if isinstance(self.normaliser, ObservationNormaliser):
            self.normaliser.update(observations)
