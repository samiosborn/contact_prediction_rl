"""Optional contact-label diagnostics for tactile and dexterous manipulation tasks."""

from __future__ import annotations

import torch


def _f1(true_positives: torch.Tensor, predicted_positives: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """F1 from counts; define all-negative perfect predictions as 1, not NaN."""
    denominator = predicted_positives + positives
    return torch.where(denominator > 0, 2 * true_positives / denominator.clamp(min=1), torch.ones_like(denominator))


def binary_contact_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | list[float]]:
    """Compute micro and per-channel contact scores; logits are pre-sigmoid.

    These measure predictive classification only. Always compare against a
    constant-mask predictor on the same trajectories to expose trivial labels.
    """
    if logits.ndim != 2 or labels.shape != logits.shape or not torch.isfinite(logits).all():
        raise ValueError("Finite logits and matching [samples, contacts] labels required")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("Contact labels must be binary")
    actual, predicted = labels.bool(), logits >= 0
    true_pos = (actual & predicted).sum(dim=0).float()
    pos = actual.sum(dim=0).float()
    pred = predicted.sum(dim=0).float()
    per_contact = _f1(true_pos, pred, pos)
    micro = _f1(true_pos.sum(), pred.sum(), pos.sum())
    return {
        "contact_f1": micro.item(),
        "contact_positive_rate": actual.float().mean().item(),
        "contact_predicted_rate": predicted.float().mean().item(),
        "per_contact_f1": per_contact.tolist(),
        "per_contact_positive_rate": actual.float().mean(dim=0).tolist(),
    }


class ContactTracker:
    """Accumulate contact prevalence, flips and constant-predictor baselines.

    observe() receives the *current* binary contact labels. done masks prevent
    counting a transition between different episodes as a real contact flip.
    """

    def __init__(self, num_envs: int, num_contacts: int, device: str | torch.device = "cpu") -> None:
        if num_envs <= 0 or not 1 <= num_contacts <= 20:
            raise ValueError("Expected positive environments and 1-20 contact channels")
        self.num_envs, self.num_contacts = num_envs, num_contacts
        self.device = torch.device(device)
        self.positives = torch.zeros(num_contacts, dtype=torch.long, device=device)
        self.flips = torch.zeros_like(self.positives)
        self.flip_opportunities = torch.zeros((), dtype=torch.long, device=device)
        self.samples = 0
        self.mask_counts: dict[int, int] = {}
        self._previous = torch.zeros((num_envs, num_contacts), dtype=torch.bool, device=device)
        self._valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def observe(self, labels: torch.Tensor, dones: torch.Tensor | None = None) -> None:
        if labels.shape != (self.num_envs, self.num_contacts):
            raise ValueError("Contact labels must be [num_envs, num_contacts]")
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("Contact labels must be binary")
        if dones is not None and dones.shape != (self.num_envs,):
            raise ValueError("Done flags must be [num_envs]")
        actual = labels.to(self.device).bool()
        valid = self._valid
        self.flips += ((actual ^ self._previous) & valid[:, None]).sum(dim=0)
        self.flip_opportunities += valid.sum()
        self.positives += actual.sum(dim=0)
        self.samples += self.num_envs

        # Bitmask counts are for diagnostics, not the hot training loop.
        weights = (1 << torch.arange(self.num_contacts, device=self.device, dtype=torch.long))
        masks = (actual.long() * weights).sum(dim=1)
        codes, counts = torch.unique(masks, return_counts=True)
        for code, count in zip(codes.tolist(), counts.tolist(), strict=True):
            self.mask_counts[code] = self.mask_counts.get(code, 0) + count

        self._previous.copy_(actual)
        self._valid.fill_(True)
        if dones is not None:
            self._valid[dones.to(self.device).bool()] = False

    def summary(self, *, fixed_pattern: tuple[int, ...] | None = None) -> dict[str, object]:
        if fixed_pattern is not None and (
            len(fixed_pattern) != self.num_contacts or any(bit not in (0, 1) for bit in fixed_pattern)
        ):
            raise ValueError("fixed_pattern must have one binary label per contact")
        if self.samples == 0:
            return {}
        rates = self.positives.float() / self.samples
        majority = rates >= 0.5
        positives = self.positives.sum().float()

        def constant_f1(pattern: torch.Tensor) -> float:
            predicted = self.samples * pattern.sum().float()
            true_pos = (self.positives * pattern.long()).sum().float()
            return _f1(true_pos, predicted, positives).item()

        if fixed_pattern is not None:
            fixed = torch.tensor(fixed_pattern, device=self.device, dtype=torch.bool)
        else:
            fixed = majority

        opportunities = int(self.flip_opportunities.item())
        return {
            "samples": self.samples,
            "per_contact_positive_rate": rates.tolist(),
            "per_contact_transition_rate": (self.flips.float() / max(opportunities, 1)).tolist(),
            "contact_transition_rate": self.flips.sum().item() / max(opportunities * self.num_contacts, 1),
            "distinct_contact_masks": len(self.mask_counts),
            "most_common_contact_mask": max(self.mask_counts, key=self.mask_counts.get),
            "most_common_mask_rate": max(self.mask_counts.values()) / self.samples,
            "majority_constant_pattern": majority.long().tolist(),
            "majority_constant_f1_in_sample": constant_f1(majority),
            "fixed_constant_pattern": fixed.long().tolist(),
            "fixed_constant_f1": constant_f1(fixed),
        }
