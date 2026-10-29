# src/contact_prediction_rl/pretraining/models.py

"""Loaders for the pretrained policies and a small toy policy with the same training interface.

Every policy takes a batch dictionary (images, ``observation.state``, ``action`` and the language tokens) and returns
``(loss, info)`` from ``forward``, as the lerobot policies do.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# lerobot policy type and Hugging Face checkpoint of each pretrained policy
LEROBOT_POLICIES = {
    "smolvla": ("smolvla", "lerobot/smolvla_base"),
    "pi0": ("pi0", "lerobot/pi0_base"),
}

# linear layers that LoRA adapts by default
LORA_TARGET_MODULES = {
    "toy": ["linear1", "linear2"],
    "smolvla": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "pi0": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

MODEL_NAMES = ["toy", *LEROBOT_POLICIES]

LEROBOT_INSTALL_HINT = "run through scripts/pretraining/with_lerobot.sh (see the Pretraining section of the README)"
OPENPI_INSTALL_HINT = "install openpi in its own environment: https://github.com/Physical-Intelligence/openpi"


def load_lerobot_policy(name: str, repo_id: str | None = None) -> nn.Module:
    """Load SmolVLA or pi0 from the Hugging Face Hub (weights are cached locally, not stored in the repository)."""
    try:
        from lerobot.policies.factory import get_policy_class
    except ImportError as error:
        raise ImportError(f"{name} needs lerobot: {LEROBOT_INSTALL_HINT}") from error

    policy_type, default_repo_id = LEROBOT_POLICIES[name]
    return get_policy_class(policy_type).from_pretrained(repo_id or default_repo_id)


def load_openpi_policy(config_name: str = "pi05_droid", checkpoint: str = "gs://openpi-assets/checkpoints/pi05_droid"):
    """Load an openpi policy for inference.

    openpi pins its own JAX and PyTorch versions, so it is not installed with this project.
    """
    try:
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as openpi_config
    except ImportError as error:
        raise ImportError(f"openpi is not installed: {OPENPI_INSTALL_HINT}") from error

    checkpoint_dir = download.maybe_download(checkpoint)
    return policy_config.create_trained_policy(openpi_config.get_config(config_name), checkpoint_dir)


class ToyPolicy(nn.Module):
    """Small image + state + tactile + language transformer that regresses an action chunk.

    It exercises the benchmark (data loading, DDP / FSDP, LoRA, mixed precision) without downloading a VLA.
    """

    def __init__(
        self,
        image_keys: tuple[str, ...],
        state_dim: int,
        tactile_dim: int,
        action_dim: int,
        chunk_size: int,
        vocab_size: int,
        width: int = 256,
        depth: int = 4,
        heads: int = 8,
    ):
        super().__init__()
        self.image_keys = image_keys
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.GELU(),
            nn.Conv2d(64, width, 3, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.state_encoder = nn.Linear(state_dim + tactile_dim, width)
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.action_queries = nn.Parameter(torch.zeros(chunk_size, width))
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(width, heads, 4 * width, batch_first=True, norm_first=True) for _ in range(depth))
        self.action_head = nn.Linear(width, action_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        batch_size = batch["action"].shape[0]
        proprio = batch["observation.state"]
        if "observation.tactile" in batch:
            proprio = torch.cat([proprio, batch["observation.tactile"]], dim=-1)

        tokens = [self.image_encoder(batch[key]).unsqueeze(1) for key in self.image_keys]
        tokens.append(self.state_encoder(proprio).unsqueeze(1))
        tokens.append(self.token_embedding(batch["observation.language.tokens"]))
        tokens.append(self.action_queries.expand(batch_size, -1, -1))
        x = torch.cat(tokens, dim=1)
        for layer in self.layers:
            x = layer(x)

        actions = self.action_head(x[:, -self.chunk_size :])
        loss = F.mse_loss(actions.float(), batch["action"].float())
        return loss, {"loss": loss.item()}


def build_policy(name: str, repo_id: str | None = None, **toy_kwargs) -> nn.Module:
    if name == "toy":
        return ToyPolicy(**toy_kwargs)
    if name in LEROBOT_POLICIES:
        return load_lerobot_policy(name, repo_id)
    raise ValueError(f"unknown model {name!r}, expected one of {MODEL_NAMES}")
