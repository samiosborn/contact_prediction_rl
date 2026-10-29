# src/contact_prediction_rl/pretraining/lora.py

"""Parameter-efficient fine-tuning with LoRA adapters (peft)."""

from __future__ import annotations

import torch.nn as nn


def apply_lora(model: nn.Module, target_modules: list[str], rank: int = 16, alpha: int = 32, dropout: float = 0.0) -> nn.Module:
    """Add LoRA adapters to the target linear layers in place and freeze every other parameter."""
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as error:
        raise ImportError("LoRA needs peft: uv sync --extra pretraining") from error

    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules)
    model = inject_adapter_in_model(config, model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) number of parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
