#!/bin/bash
# scripts/pretraining/slurm_benchmark.sh
#
# Single-node scaling sweep of the fine-tuning benchmark over 1, 2, 4 and 8 GPUs.
# Template: set the account, partition and module / container lines for the target system.
#
#   sbatch scripts/pretraining/slurm_benchmark.sh
#   MODEL=smolvla MODE=lora STRATEGY=fsdp sbatch scripts/pretraining/slurm_benchmark.sh

#SBATCH --job-name=cprl-benchmark
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=logs/pretraining/slurm-%j.out

set -euo pipefail

MODEL=${MODEL:-toy}
MODE=${MODE:-full}
STRATEGY=${STRATEGY:-ddp}
BATCH_SIZE=${BATCH_SIZE:-8}
OUTPUT=${OUTPUT:-logs/pretraining/benchmark_${SLURM_JOB_ID}.jsonl}
# lerobot and peft are layered on top of the locked environment without changing uv.lock
RUN=${RUN:-scripts/pretraining/with_lerobot.sh}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}  # compute nodes often have no internet: download the weights first

for GPUS in 1 2 4 8; do
    $RUN torchrun --standalone --nproc_per_node "$GPUS" scripts/pretraining/benchmark_finetune.py \
        --model "$MODEL" --mode "$MODE" --strategy "$STRATEGY" --batch_size "$BATCH_SIZE" --output "$OUTPUT"
done

$RUN python scripts/pretraining/benchmark_finetune.py --summarise "$OUTPUT"
