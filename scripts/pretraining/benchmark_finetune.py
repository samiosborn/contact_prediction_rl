# scripts/pretraining/benchmark_finetune.py

"""Fine-tuning benchmark entrypoint, launched with torchrun (one process per GPU).

Examples:
    uv run torchrun --nproc_per_node 1 scripts/pretraining/benchmark_finetune.py --model toy
    scripts/pretraining/with_lerobot.sh torchrun --nproc_per_node 4 \\
        scripts/pretraining/benchmark_finetune.py --model smolvla --mode lora --strategy ddp
    uv run python scripts/pretraining/benchmark_finetune.py --summarise logs/pretraining/benchmark.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

from contact_prediction_rl.pretraining.benchmark import MODES, BenchmarkConfig, run_benchmark, summarise
from contact_prediction_rl.pretraining.distributed import STRATEGIES
from contact_prediction_rl.pretraining.models import MODEL_NAMES


def parse_args() -> argparse.Namespace:
    defaults = BenchmarkConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODEL_NAMES, default=defaults.model)
    parser.add_argument("--repo_id", default=None, help="Hugging Face checkpoint replacing the model's default.")
    parser.add_argument("--mode", choices=MODES, default=defaults.mode)
    parser.add_argument("--strategy", choices=STRATEGIES, default=defaults.strategy)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size, help="Samples per GPU.")
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--warmup_steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--no_bf16", action="store_true", help="Train in float32.")
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output", default="logs/pretraining/benchmark.jsonl")
    parser.add_argument("--summarise", metavar="JSONL", default=None, help="Print the scaling table of a results file and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarise:
        for row in summarise(args.summarise):
            print(json.dumps(row))
        return

    cfg = BenchmarkConfig(
        model=args.model,
        repo_id=args.repo_id,
        mode=args.mode,
        strategy=args.strategy,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        bf16=not args.no_bf16,
        activation_checkpointing=args.activation_checkpointing,
        lora_rank=args.lora_rank,
        seed=args.seed,
        output=args.output,
    )
    result = run_benchmark(cfg)
    if int(os.environ.get("RANK", 0)) != 0:
        return
    print(
        f"[{cfg.model} {cfg.mode} {cfg.strategy} x{result['world_size']}] "
        f"{result['samples_per_s']:.1f} samples/s, step {result['step_time_s'] * 1e3:.1f} ms, "
        f"data {100 * result['data_time_fraction']:.1f} %, peak memory {result['peak_memory_gb']:.2f} GB, "
        f"trainable {result['trainable_params'] / 1e6:.1f} M of {result['total_params'] / 1e6:.1f} M parameters"
    )


if __name__ == "__main__":
    main()
