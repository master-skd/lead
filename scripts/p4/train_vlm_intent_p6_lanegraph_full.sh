#!/usr/bin/env bash
# Train a fresh 3-camera lane-graph intent decoder on exact-frame full-data caches.
set -euo pipefail

cd "$(dirname "$0")/../.."

LEAD_ENV=/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead
SPLIT_DIR=outputs/local_training/p5_stepB3a_v2_dense_data/route_split
LOG_DIR=outputs/local_training/vlm_intent_p6_lanegraph_full
LANEGRAPH_SHARDS="${LANEGRAPH_SHARDS:-4}"

"$LEAD_ENV/bin/python" scripts/p4/check_vlm_intent_full_ready.py \
  --lanegraph-shards "$LANEGRAPH_SHARDS"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export LEAD_PROJECT_ROOT="$PWD"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS=$("$LEAD_ENV/bin/python" -c 'import torch; print(torch.cuda.device_count())')
if [[ "$GPUS" -lt 1 ]]; then
  echo "No CUDA GPUs visible for intent training" >&2
  exit 1
fi

"$LEAD_ENV/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="$GPUS" \
  --max_restarts=0 scripts/p4/train_vlm_intent.py \
  --vlm-cache data/p6/vlm_cache_3cam \
  --manifest "$SPLIT_DIR/train.jsonl" \
  --val-manifest "$SPLIT_DIR/heldout.jsonl" \
  --split-metadata "$SPLIT_DIR/metadata.json" \
  --logdir "$LOG_DIR" \
  --multimodal-intent --tversky-weight 0.75 \
  --lanegraph-label-dir data/p6/lanegraph_label \
  --batch-size 256 --lr 6e-4 --epochs 15 --num-workers 1 \
  "$@"
