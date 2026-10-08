# src/contact_prediction_rl/learning/models/observation_encoder.py

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


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

        # Build a list of layers
        layers = []
        last_dim = observation_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(resolve_nn_activation(activation))
            last_dim = hidden_dim

        # Map the final hidden layer into the latent representation
        layers.append(nn.Linear(last_dim, latent_dim))
        layers.append(resolve_nn_activation(activation))

        # Unpack layers
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)
    