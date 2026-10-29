# src/contact_prediction_rl/pretraining/data.py

"""Demonstration data for fine-tuning: a synthetic dataset for benchmarking and a loader for lerobot datasets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


@dataclass
class BatchSpec:
    """Shapes of one demonstration sample. The defaults follow the Franka + HX5-D20 (7 arm + 20 hand joints)."""

    image_keys: tuple[str, ...] = ("observation.images.wrist", "observation.images.front")
    image_shape: tuple[int, int, int] = (3, 224, 224)
    state_dim: int = 27
    tactile_dim: int = 5  # one contact per fingertip; 0 for policies without a tactile input
    action_dim: int = 27
    chunk_size: int = 50
    language_length: int = 48
    vocab_size: int = 1000  # random token ids stay below this, which is valid for every supported tokenizer


def spec_from_lerobot_policy(policy) -> BatchSpec:
    """Shapes expected by a lerobot policy, read from its configuration."""
    config = policy.config
    return BatchSpec(
        image_keys=tuple(config.image_features),
        image_shape=tuple(next(iter(config.image_features.values())).shape),
        state_dim=config.robot_state_feature.shape[0],
        tactile_dim=0,
        action_dim=config.action_feature.shape[0],
        chunk_size=config.chunk_size,
        language_length=config.tokenizer_max_length,
    )


class SyntheticDemoDataset(Dataset):
    """Random samples with the shapes of real demonstrations, for throughput and memory benchmarks."""

    def __init__(self, spec: BatchSpec, num_samples: int = 10_000, seed: int = 0):
        self.spec = spec
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        spec = self.spec
        generator = torch.Generator().manual_seed(self.seed + index)
        sample = {key: torch.rand(spec.image_shape, generator=generator) for key in spec.image_keys}
        sample["observation.state"] = torch.randn(spec.state_dim, generator=generator)
        if spec.tactile_dim > 0:
            sample["observation.tactile"] = (torch.rand(spec.tactile_dim, generator=generator) > 0.5).float()
        sample["action"] = torch.randn(spec.chunk_size, spec.action_dim, generator=generator)
        sample["observation.language.tokens"] = torch.randint(spec.vocab_size, (spec.language_length,), generator=generator)
        sample["observation.language.attention_mask"] = torch.ones(spec.language_length, dtype=torch.bool)
        return sample


def load_lerobot_dataset(repo_id: str, root: str | None = None):
    """A demonstration dataset in the lerobot format, from the Hub or a local directory."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise ImportError("lerobot datasets need lerobot: run through scripts/pretraining/with_lerobot.sh") from error

    return LeRobotDataset(repo_id, root=root)


def make_loader(dataset: Dataset, batch_size: int, num_workers: int = 4, distributed: bool = False, seed: int = 0) -> DataLoader:
    """Shuffled loader with pinned memory and prefetching; each rank reads its own shard when distributed."""
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=True) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
