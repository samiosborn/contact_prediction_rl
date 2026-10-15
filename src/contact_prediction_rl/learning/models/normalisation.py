# src/contact_prediction_rl/learning/models/normalisation.py

import torch
import torch.nn as nn


class ObservationNormaliser(nn.Module):
    """Normalise observations using running mean and variance"""

    def __init__(
        self,
        observation_dim: int,
        eps: float = 1e-2,
    ) -> None:
        super().__init__()

        self.eps = eps

        self.register_buffer(
            "mean",
            torch.zeros(observation_dim),
        )

        self.register_buffer(
            "variance",
            torch.ones(observation_dim),
        )

        self.register_buffer(
            "count",
            torch.tensor(0.0),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.sqrt(self.variance)

        return (observations - self.mean) / (std + self.eps)

    @torch.no_grad()
    def update(
        self,
        observations: torch.Tensor,
    ) -> None:
        """Update running statistics from a batch of observations"""

        if not self.training:
            return

        batch_count = observations.shape[0]

        batch_mean = observations.mean(dim=0)

        batch_variance = observations.var(
            dim=0,
            unbiased=False,
        )

        if self.count == 0:
            self.mean.copy_(batch_mean)
            self.variance.copy_(batch_variance)
            self.count.fill_(batch_count)
            return

        total_count = self.count + batch_count

        delta = batch_mean - self.mean

        new_mean = self.mean + delta * batch_count / total_count

        previous_sum_squares = self.variance * self.count

        batch_sum_squares = batch_variance * batch_count

        correction = delta.square() * self.count * batch_count / total_count

        new_variance = (previous_sum_squares + batch_sum_squares + correction) / total_count

        self.mean.copy_(new_mean)
        self.variance.copy_(new_variance)
        self.count.copy_(total_count)
