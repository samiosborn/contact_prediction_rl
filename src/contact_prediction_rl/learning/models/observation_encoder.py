# src/contact_prediction_rl/learning/models/observation_encoder.py

import torch
import torch.nn as nn

from contact_prediction_rl.learning.models.activations import get_activation


class ObservationEncoder(nn.Module):
    """Network for encoding observations into a latent representation"""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        latent_dim: int,
        activation: str,
    ) -> None:
        super().__init__()

        layers = []
        last_dim = observation_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(get_activation(activation))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, latent_dim))
        layers.append(get_activation(activation))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(observations)
