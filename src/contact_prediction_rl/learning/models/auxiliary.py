# src/contact_prediction_rl/learning/models/auxiliary.py

"""Auxiliary (self-supervised) objectives of the tactile-RL paper, applied to the shared latent ``z``.

An objective owns the small networks it needs, which are discarded after training, and computes a loss from a batch of
observations and their latents. The shared encoder is trained by it through ``z``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Binary cross-entropy between the decoded and the real tactile readings, and detection metrics"""

        targets = observations[:, -self.tactile_dim :]

        loss, metrics = self.decoder.compute_loss(latent, targets)

        return loss, {
            "aux_positive_rate": metrics["contact_positive_rate"],
            "aux_predicted_rate": metrics["contact_predicted_rate"],
            "aux_f1": metrics["contact_f1"],
        }
