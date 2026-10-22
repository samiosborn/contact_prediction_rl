# src/contact_prediction_rl/learning/models/shared.py

"""Actor and critic that share one observation encoder (the agent of the tactile-RL paper).

The encoder maps the observation to a latent ``z``. The policy and the value function are small heads that both read the
same ``z``, so the policy loss, the value loss and, later, an auxiliary loss all train the encoder.

Paper architecture (Appendix F): encoder ``o -> 1024 -> 512 -> 256 = z`` with layer normalisation and ELU after every
layer; policy ``z -> 128 -> 64 -> actions`` with ELU after the first two layers and a tanh output; value
``z -> 128 -> 64 -> 1``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from contact_prediction_rl.learning.models.activations import get_activation
from contact_prediction_rl.learning.models.actor import Actor
from contact_prediction_rl.learning.models.normalisation import ObservationNormaliser


class SharedEncoder(nn.Module):
    """Observation normalisation followed by an MLP encoder with layer normalisation"""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        latent_dim: int,
        activation: str = "elu",
        layer_norm: bool = True,
        observation_normalisation: bool = True,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        self.normaliser = (
            ObservationNormaliser(observation_dim)
            if observation_normalisation
            else nn.Identity()
        )

        layers: list[nn.Module] = []
        width_in = observation_dim

        for width in (*hidden_dims, latent_dim):
            layers.append(nn.Linear(width_in, width))

            if layer_norm:
                layers.append(nn.LayerNorm(width))

            layers.append(get_activation(activation))
            width_in = width

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(self.normaliser(observations))


def _head(
    latent_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    output_dim: int,
    activation: str,
    output_activation: str | None = None,
) -> nn.Sequential:
    """MLP head from the latent to ``output_dim`` values, with an optional activation on the output"""

    layers: list[nn.Module] = []
    width_in = latent_dim

    for width in hidden_dims:
        layers.extend((nn.Linear(width_in, width), get_activation(activation)))
        width_in = width

    layers.append(nn.Linear(width_in, output_dim))

    if output_activation is not None:
        layers.append(get_activation(output_activation))

    return nn.Sequential(*layers)


class SharedActor(Actor):
    """Gaussian policy whose mean is a head on a shared encoder

    It reuses everything of :class:`Actor` except the construction: the sampling, the log-probability, the entropy and
    the KL divergence are the same, so the PPO code does not change. The mean is ``head(z)``, and the learned standard
    deviation is state independent.
    """

    def __init__(
        self,
        encoder: SharedEncoder,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        initial_std: float = 0.2,
        output_activation: str | None = "tanh",
        action_head_init_scale: float | None = None,
    ) -> None:
        # The parent builds its own encoder, which this class replaces, so only the module base is initialised
        nn.Module.__init__(self)

        if initial_std <= 0:
            raise ValueError("Initial action standard deviation must be positive")

        self.encoder = encoder

        self.action_head = _head(
            encoder.latent_dim,
            hidden_dims,
            action_dim,
            activation,
            output_activation,
        )

        # Optionally start with small mean actions (see the note on the all-joints task: an untrained policy should sit
        # close to zero action)
        if action_head_init_scale is not None:
            last_linear = [m for m in self.action_head if isinstance(m, nn.Linear)][-1]

            with torch.no_grad():
                last_linear.weight.mul_(action_head_init_scale)
                last_linear.bias.zero_()

        self.std = nn.Parameter(
            torch.full(
                (action_dim,),
                initial_std,
            )
        )

        self._distribution = None

    def encode(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Latent representation z of the observations (normalisation included)"""

        return self.encoder(observations)

    def update_normalisation(
        self,
        observations: torch.Tensor,
    ) -> None:
        """Update the running observation statistics, which live in the shared encoder"""

        if isinstance(self.encoder.normaliser, ObservationNormaliser):
            self.encoder.normaliser.update(observations)


class SharedCritic(nn.Module):
    """Value function: a head on the same shared encoder as the actor"""

    def __init__(
        self,
        encoder: SharedEncoder,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.value_head = _head(encoder.latent_dim, hidden_dims, 1, activation)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.value_head(self.encoder(observations))

    def update_normalisation(
        self,
        observations: torch.Tensor,
    ) -> None:
        """Nothing to do: the shared encoder's statistics are updated once, by the actor"""
