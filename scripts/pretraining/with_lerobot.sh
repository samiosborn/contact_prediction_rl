#!/bin/bash
# scripts/pretraining/with_lerobot.sh
#
# Run a command with lerobot (SmolVLA, pi0) and peft layered on top of the locked environment, without changing
# uv.lock. Adding lerobot to pyproject.toml would also change base packages (protobuf, einops), so it stays out of the
# lock. torch and torchvision are pinned to the project's CUDA 12.8 builds, otherwise the overlay pulls newer ones.
#
#   scripts/pretraining/with_lerobot.sh torchrun --nproc_per_node 1 scripts/pretraining/benchmark_finetune.py --model smolvla

set -euo pipefail

exec uv run \
    --index https://download.pytorch.org/whl/cu128 \
    --with "torch==2.11.0+cu128" \
    --with "torchvision==0.26.0+cu128" \
    --with "lerobot[smolvla]>=0.6,<0.7" \
    --with "peft>=0.17" \
    "$@"
