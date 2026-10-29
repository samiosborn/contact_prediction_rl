# src/contact_prediction_rl/learning/models/auxiliary.py

"""Auxiliary (self-supervised) objectives of the tactile-RL paper, applied to the shared latent ``z``.

An objective owns the small networks it needs, which are discarded after training, and computes a loss from a batch of
observations and their latents. The shared encoder is trained by it through ``z``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from contact_prediction_rl.learning.models.activations import get_activation
from contact_prediction_rl.learning.models.contact_prediction import ContactPredictionHead


class TactileReconstruction(nn.Module):
    """Tactile reconstruction (TR): decode the tactile readings of the observation from the latent z.

    The policy observation ends with the stacked binary contact readings (the last ``tactile_dim`` entries: the last k
    steps of every fingertip). The decoder ``z -> 512 -> 512 -> tactile_dim`` has to reproduce them, as a binary
    classification with a positive weight, so that the latent keeps the tactile information instead of favouring the
    smoother proprioceptive signals.
    """

    def __init__(
        self,
        latent_dim: int,
        tactile_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        activation: str = "elu",
        pos_weight: float = 10.0,
    ) -> None:
        super().__init__()

        self.tactile_dim = tactile_dim

        # The contact head is an MLP with a binary cross-entropy on logits, a positive weight and detection metrics
        self.decoder = ContactPredictionHead(
            latent_dim=latent_dim,
            num_contacts=tactile_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            pos_weight=pos_weight,
        )

    def loss(
        self,
        observations: torch.Tensor,
        latent: torch.Tensor,
        normalised_observations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Binary cross-entropy between the decoded and the real tactile readings, and detection metrics

        ``normalised_observations`` is not used: the target is the raw binary readings.
        """

        targets = observations[:, -self.tactile_dim :]

        loss, metrics = self.decoder.compute_loss(latent, targets)

        return loss, {
            "aux_positive_rate": metrics["contact_positive_rate"],
            "aux_predicted_rate": metrics["contact_predicted_rate"],
            "aux_f1": metrics["contact_f1"],
        }


class FullReconstruction(nn.Module):
    """Full reconstruction (FR): decode the whole observation from the latent z.

    The decoder ``z -> 512 -> 512 -> observation size`` rebuilds the tactile readings (the last ``tactile_dim`` entries)
    as in TR, and the proprioceptive part (everything before them) as numbers. The loss is the sum of the two::

        L_FR = BCE(tactile) + MSE(proprioception)

    The proprioceptive target is the *normalised* observation, the version the encoder reads, so that quantities of
    very different sizes (positions, velocities, the goal) count about equally in the squared error.
    """

    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        tactile_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        activation: str = "elu",
        pos_weight: float = 10.0,
    ) -> None:
        super().__init__()

        if not 0 < tactile_dim < observation_dim:
            raise ValueError("The tactile part must be a non-empty part of the observation")

        self.tactile_dim = tactile_dim

        layers: list[nn.Module] = []
        width_in = latent_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(width_in, width), get_activation(activation)))
            width_in = width
        layers.append(nn.Linear(width_in, observation_dim))
        self.decoder = nn.Sequential(*layers)

        self.register_buffer("positive_weight", torch.full((tactile_dim,), float(pos_weight)))

    def loss(
        self,
        observations: torch.Tensor,
        latent: torch.Tensor,
        normalised_observations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Sum of the tactile binary cross-entropy and the proprioceptive squared error, with metrics"""

        if normalised_observations is None:
            raise ValueError("Full reconstruction needs the normalised observations as the proprioceptive target")

        decoded = self.decoder(latent)

        tactile_logits = decoded[:, -self.tactile_dim :]
        proprio_prediction = decoded[:, : -self.tactile_dim]

        tactile_targets = observations[:, -self.tactile_dim :].detach()
        proprio_targets = normalised_observations[:, : -self.tactile_dim].detach()

        tactile_loss = F.binary_cross_entropy_with_logits(
            tactile_logits, tactile_targets, pos_weight=self.positive_weight
        )
        proprio_loss = F.mse_loss(proprio_prediction, proprio_targets)

        with torch.no_grad():
            actual = tactile_targets.bool()
            predicted = tactile_logits >= 0
            true_positives = (predicted & actual).sum().float()
            precision = true_positives / predicted.sum().clamp(min=1)
            recall = true_positives / actual.sum().clamp(min=1)
            f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

        return tactile_loss + proprio_loss, {
            "aux_tactile_bce": tactile_loss.item(),
            "aux_proprio_mse": proprio_loss.item(),
            "aux_positive_rate": tactile_targets.mean().item(),
            "aux_predicted_rate": predicted.float().mean().item(),
            "aux_f1": f1.item(),
        }
