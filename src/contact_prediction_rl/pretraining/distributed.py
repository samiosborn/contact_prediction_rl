# src/contact_prediction_rl/pretraining/distributed.py

"""Multi-GPU training: process-group setup from torchrun, DDP / FSDP wrapping and activation checkpointing."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

STRATEGIES = ["none", "ddp", "fsdp"]


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def init_distributed() -> DistInfo:
    """Join the process group described by the torchrun environment variables (a single process without them)."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return DistInfo(rank, local_rank, world_size, device)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _layer_types(model: nn.Module) -> set[type[nn.Module]]:
    """Types of the repeated blocks (the members of every ModuleList), the units to shard and checkpoint."""
    return {type(layer) for module in model.modules() if isinstance(module, nn.ModuleList) for layer in module}


def enable_activation_checkpointing(model: nn.Module) -> None:
    """Recompute the activations of every repeated block in the backward pass instead of storing them."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

    layer_types = tuple(_layer_types(model))
    apply_activation_checkpointing(model, check_fn=lambda module: isinstance(module, layer_types))


def wrap_model(model: nn.Module, strategy: str, info: DistInfo, find_unused_parameters: bool = False) -> nn.Module:
    """Move the model to its device and wrap it for data-parallel (DDP) or fully sharded (FSDP2) training."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")
    model = model.to(info.device)
    if strategy == "none" or not info.distributed:
        return model

    if strategy == "ddp":
        device_ids = [info.local_rank] if info.device.type == "cuda" else None
        return nn.parallel.DistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=find_unused_parameters)

    from torch.distributed.fsdp import fully_shard

    # shard each repeated block on its own so that only one block is gathered at a time, then the remaining parameters
    layer_types = tuple(_layer_types(model))
    for module in list(model.modules()):
        if isinstance(module, layer_types):
            fully_shard(module)
    fully_shard(model)
    return model


def max_across_ranks(value: float, info: DistInfo) -> float:
    if not info.distributed:
        return value
    tensor = torch.tensor(value, device=info.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()
