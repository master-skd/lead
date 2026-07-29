#!/bin/bash
# P5b open-loop eval: measure the ACTUAL p5b_S / p5b_M imitation quality (ADE/FDE)
# on the training-cache VLM features, reproducing the canonical p5b forward.
#
# Purpose: closed-loop p5b_S=72.9 / p5b_M=66.2 DS collapsed vs LEAD/P2 ~93. This
# separates "1/10-data underfit" from "closed-loop-specific break (live IPC VLM
# features != training cache)":
#   wp_ade ~= P2's 0.185 here  -> imitation fine; the break is closed-loop-specific.
#   wp_ade much worse here     -> 1/10-data underfit is real.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/p4/eval_p5b_openloop.sh outputs/local_training/p5b_S
#   CUDA_VISIBLE_DEVICES=0 bash scripts/p4/eval_p5b_openloop.sh outputs/local_training/p5b_M
#   # multi-GPU:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/p4/eval_p5b_openloop.sh outputs/local_training/p5b_M
#   # smoke test (32 frames):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/p4/eval_p5b_openloop.sh outputs/local_training/p5b_S --limit 32
set -euo pipefail

CKPT_DIR="${1:?usage: eval_p5b_openloop.sh <ckpt_dir> [extra args...]}"
shift || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")

if [ "$nproc_per_node" -gt 1 ]; then
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=$((10000 + RANDOM % 50000))
  torchrun --standalone --nnodes=1 --nproc_per_node="$nproc_per_node" --max_restarts=0 \
    scripts/p4/eval_p5b_openloop.py --ckpt-dir "$CKPT_DIR" --with-zero "$@"
else
  python scripts/p4/eval_p5b_openloop.py --ckpt-dir "$CKPT_DIR" --with-zero "$@"
fi
