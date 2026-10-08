# src/contact_prediction_rl/learning/models/actor.py

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel

from contact_prediction_rl.learning.models.observation_encoder import ObservationEncoder


class Actor(MLPModel):
    """Actor with an explicit observation encoder and action head"""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        if len(hidden_dims) < 1:
            raise ValueError("Actor requires at least one hidden dimension")

        # Final hidden dimension is the latent dimension
        encoder_hidden_dims = list(hidden_dims[:-1])
        latent_dim = hidden_dims[-1]

        # Set up RSL-RL observation handling, normalisation and action distribution
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        # Determine the output dimension required by the action distribution
        action_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim

        # Replace the default RSL-RL MLP with an explicit encoder and action head
        self.mlp = nn.Sequential(
            ObservationEncoder(
                observation_dim=self.obs_dim,
                hidden_dims=encoder_hidden_dims,
                latent_dim=latent_dim,
                activation=activation,
            ),
            nn.Linear(latent_dim, action_output_dim),
        )

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    @property
    def observation_encoder(self) -> ObservationEncoder:
        return self.mlp[0]

    @property
    def action_head(self) -> nn.Linear:
        return self.mlp[1]

    def encode(self, obs: TensorDict) -> torch.Tensor:
        # Concatenate and normalise the task observations
        observations = super().get_latent(obs)

        # Encode observations into z
        return self.observation_encoder(observations)
    