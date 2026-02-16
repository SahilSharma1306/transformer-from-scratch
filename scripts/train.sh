#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Distributed Training Launch Script
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/train.sh                    # Auto-detect GPUs
#   bash scripts/train.sh 2                  # Use 2 GPUs
#   bash scripts/train.sh 4 --num_epochs 50  # 4 GPUs + config override
#
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}
shift 2>/dev/null || true

echo "╔══════════════════════════════════════════════════╗"
echo "║  Transformer from Scratch — Training Launch      ║"
echo "║  GPUs: ${NUM_GPUS}                                        ║"
echo "╚══════════════════════════════════════════════════╝"

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    -m transformer.train \
    --config configs/default.yaml \
    "$@"
