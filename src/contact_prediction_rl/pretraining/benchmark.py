# src/contact_prediction_rl/pretraining/benchmark.py

"""Fine-tuning benchmark: training throughput, step time, data-loading share and peak GPU memory.

Launched once per process by torchrun; rank 0 appends one JSON line per run, and ``summarise`` turns a sweep over
1, 2, 4 and 8 GPUs into scaling efficiencies.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from contact_prediction_rl.pretraining.data import BatchSpec, SyntheticDemoDataset, make_loader, spec_from_lerobot_policy
from contact_prediction_rl.pretraining.distributed import (
    DistInfo,
    cleanup_distributed,
    enable_activation_checkpointing,
    init_distributed,
    max_across_ranks,
    wrap_model,
)
from contact_prediction_rl.pretraining.lora import apply_lora, count_parameters
from contact_prediction_rl.pretraining.models import LORA_TARGET_MODULES, build_policy

MODES = ["full", "lora"]


@dataclass
class BenchmarkConfig:
    model: str = "toy"
    repo_id: str | None = None  # overrides the default checkpoint of a pretrained model
    mode: str = "full"
    strategy: str = "ddp"
    batch_size: int = 8  # per GPU
    steps: int = 50  # timed steps, after the warm-up
    warmup_steps: int = 10
    learning_rate: float = 1e-4
    num_workers: int = 4
    bf16: bool = True
    activation_checkpointing: bool = False
    lora_rank: int = 16
    num_samples: int = 10_000
    seed: int = 0
    output: str | None = None  # JSON-lines file the results are appended to


def _build(cfg: BenchmarkConfig) -> tuple[torch.nn.Module, BatchSpec]:
    if cfg.model == "toy":
        spec = BatchSpec()
        policy = build_policy(
            "toy",
            image_keys=spec.image_keys,
            state_dim=spec.state_dim,
            tactile_dim=spec.tactile_dim,
            action_dim=spec.action_dim,
            chunk_size=spec.chunk_size,
            vocab_size=spec.vocab_size,
        )
    else:
        policy = build_policy(cfg.model, cfg.repo_id)
        spec = spec_from_lerobot_policy(policy)

    if cfg.mode == "lora":
        policy = apply_lora(policy, LORA_TARGET_MODULES[cfg.model], rank=cfg.lora_rank)
    elif cfg.mode == "full":
        # the lerobot policies freeze their vision-language backbone by default
        for parameter in policy.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}, expected one of {MODES}")
    return policy, spec


def _synchronize(info: DistInfo) -> None:
    if info.device.type == "cuda":
        torch.cuda.synchronize(info.device)


def run_benchmark(cfg: BenchmarkConfig) -> dict:
    info = init_distributed()
    torch.manual_seed(cfg.seed)

    policy, spec = _build(cfg)
    trainable_params, total_params = count_parameters(policy)
    if cfg.activation_checkpointing:
        enable_activation_checkpointing(policy)
    # the pretrained policies leave some parameters out of the loss (for example unused cameras)
    model = wrap_model(policy, cfg.strategy, info, find_unused_parameters=cfg.model != "toy")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)

    loader = make_loader(SyntheticDemoDataset(spec, cfg.num_samples, cfg.seed), cfg.batch_size, cfg.num_workers, info.distributed, cfg.seed)

    def batches():
        epoch = 0
        while True:
            if info.distributed:
                loader.sampler.set_epoch(epoch)
            yield from loader
            epoch += 1

    iterator = batches()
    model.train()
    step_times, data_times = [], []
    loss = torch.zeros(())
    for step in range(cfg.warmup_steps + cfg.steps):
        if step == cfg.warmup_steps and info.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(info.device)

        start = time.perf_counter()
        batch = {key: value.to(info.device, non_blocking=True) for key, value in next(iterator).items()}
        data_end = time.perf_counter()

        with torch.autocast(device_type=info.device.type, dtype=torch.bfloat16, enabled=cfg.bf16):
            loss, _ = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _synchronize(info)

        if step >= cfg.warmup_steps:
            step_times.append(time.perf_counter() - start)
            data_times.append(data_end - start)

    # the slowest rank sets the pace of synchronous data-parallel training
    step_time = max_across_ranks(statistics.mean(step_times), info)
    data_time = max_across_ranks(statistics.mean(data_times), info)
    peak_memory = torch.cuda.max_memory_allocated(info.device) / 1e9 if info.device.type == "cuda" else 0.0
    global_batch = cfg.batch_size * info.world_size

    result = {
        **asdict(cfg),
        "world_size": info.world_size,
        "device": torch.cuda.get_device_name(info.device) if info.device.type == "cuda" else "cpu",
        "trainable_params": trainable_params,
        "total_params": total_params,
        "global_batch_size": global_batch,
        "step_time_s": step_time,
        "samples_per_s": global_batch / step_time,
        "data_time_fraction": data_time / step_time,
        "peak_memory_gb": max_across_ranks(peak_memory, info),
        "final_loss": loss.item(),
    }
    if info.is_main and cfg.output:
        Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output, "a") as file:
            file.write(json.dumps(result) + "\n")

    cleanup_distributed()
    return result


def summarise(path: str) -> list[dict]:
    """Weak-scaling efficiency of each configuration: throughput on N GPUs over N times the throughput on one GPU."""
    groups = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        result = json.loads(line)
        key = (result["model"], result["mode"], result["strategy"], result["batch_size"], result["activation_checkpointing"])
        groups[key][result["world_size"]] = result

    rows = []
    for (model, mode, strategy, batch_size, checkpointing), runs in sorted(groups.items()):
        baseline = runs.get(1)
        for world_size, result in sorted(runs.items()):
            efficiency = result["samples_per_s"] / (world_size * baseline["samples_per_s"]) if baseline else None
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "strategy": strategy,
                    "batch_size": batch_size,
                    "activation_checkpointing": checkpointing,
                    "world_size": world_size,
                    "samples_per_s": result["samples_per_s"],
                    "peak_memory_gb": result["peak_memory_gb"],
                    "data_time_fraction": result["data_time_fraction"],
                    "scaling_efficiency": efficiency,
                }
            )
    return rows
