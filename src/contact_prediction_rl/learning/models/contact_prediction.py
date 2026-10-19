# src/contact_prediction_rl/learning/models/contact_prediction.py

"""Training-only binary fingertip contact head sharing the actor's latent."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from contact_prediction_rl.learning.models.activations import get_activation


class ContactPredictionHead(nn.Module):
    """Predict one independent object-contact logit per fingertip."""

    def __init__(
        self,
        latent_dim: int,
        num_contacts: int,
        hidden_dims: list[int] | tuple[int, ...],
        activation: str = "elu",
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or num_contacts <= 0 or pos_weight <= 0:
            raise ValueError("Latent size, contact count and BCE positive weight must be positive")
        layers: list[nn.Module] = []
        width_in = latent_dim
        for width in hidden_dims:
            if width <= 0:
                raise ValueError("Contact head hidden dimensions must be positive")
            layers.extend((nn.Linear(width_in, width), get_activation(activation)))
            width_in = width
        layers.append(nn.Linear(width_in, num_contacts))
        self.network = nn.Sequential(*layers)
        self.num_contacts = num_contacts
        self.register_buffer("positive_weight", torch.full((num_contacts,), float(pos_weight)))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)  # raw logits, never sigmoid before BCE

    def compute_loss(self, latent: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """BCE on c_t from the same rollout o_t used to compute latent z_t."""
        if labels.ndim != 2 or labels.shape != (latent.shape[0], self.num_contacts):
            raise ValueError(f"Expected binary labels [batch, {self.num_contacts}], got {tuple(labels.shape)}")
        targets = labels.detach().float()
        logits = self(latent)
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.positive_weight)
        with torch.no_grad():
            actual = targets.bool()
            predicted = logits >= 0  # sigmoid(logits) >= 0.5
            true_pos = (predicted & actual).sum().float()
            precision = true_pos / predicted.sum().clamp(min=1)
            recall = true_pos / actual.sum().clamp(min=1)
            f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
            metrics = {
                "contact_positive_rate": targets.mean().item(),
                "contact_predicted_rate": predicted.float().mean().item(),
                "contact_f1": f1.item(),
            }
        return loss, metrics
