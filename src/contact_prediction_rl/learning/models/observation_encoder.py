# src/contact_prediction_rl/learning/models/observation_encoder.py

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """Network for encoding observations into a latent representation"""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()

        # Build a list of layers
        layers = []
        last_dim = observation_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, latent_dim))
        layers.append(nn.ELU())

        # Unpack layers
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)
