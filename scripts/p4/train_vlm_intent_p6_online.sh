#!/usr/bin/env bash
# Full-frame P6 intent training with online frozen Qwen; no VLM feature cache.
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="${VLM_ONLINE_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/qwenvl/bin/python}"
if [[ -n "${VLM_ONLINE_ASSET_ROOT:-}" ]]; then
  # All four data archives can be extracted beneath one directory outside this repo.
  MODEL_DEFAULT="$VLM_ONLINE_ASSET_ROOT/models/Qwen3-VL-4B-Instruct"
  SPLIT_DEFAULT="$VLM_ONLINE_ASSET_ROOT/outputs/local_training/p5_stepB3a_v2_dense_data/route_split"
  LABEL_DEFAULT="$VLM_ONLINE_ASSET_ROOT/data/p6/lanegraph_label"
  RGB_DEFAULT="$VLM_ONLINE_ASSET_ROOT/data/carla_leaderboard2/data"
else
  MODEL_DEFAULT=/mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct
  SPLIT_DEFAULT=outputs/local_training/p5_stepB3a_v2_dense_data/route_split
  LABEL_DEFAULT=data/p6/lanegraph_label
  RGB_DEFAULT=data/carla_leaderboard2/data
fi
MODEL="${VLM_ONLINE_MODEL:-$MODEL_DEFAULT}"
SPLIT_DIR="${VLM_ONLINE_SPLIT_DIR:-$SPLIT_DEFAULT}"
LABEL_DIR="${VLM_ONLINE_LABEL_DIR:-$LABEL_DEFAULT}"
RGB_ROOT="${VLM_ONLINE_RGB_ROOT:-$RGB_DEFAULT}"
LOGDIR="${VLM_ONLINE_LOGDIR:-outputs/local_training/vlm_intent_p6_lanegraph_online}"

[[ -x "$PY" && -d "$MODEL" && -f "$SPLIT_DIR/train.jsonl" \
    && -f "$SPLIT_DIR/heldout.jsonl" && -f "$SPLIT_DIR/metadata.json" \
    && -d "$LABEL_DIR" && -d "$RGB_ROOT" ]] || {
  echo "Missing online Python, Qwen model, route split, RGB, or lane-graph labels" >&2
  exit 1
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

GPUS=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
if [[ "$GPUS" -lt 1 ]]; then
  echo "No CUDA GPUs visible" >&2
  exit 1
fi

"$(dirname "$PY")/torchrun" --standalone --nnodes=1 --nproc_per_node="$GPUS" \
  --max_restarts=0 scripts/p4/train_vlm_intent_online.py \
  --qwen-model "$MODEL" \
  --train-manifest "$SPLIT_DIR/train.jsonl" \
  --val-manifest "$SPLIT_DIR/heldout.jsonl" \
  --split-metadata "$SPLIT_DIR/metadata.json" \
  --lanegraph-label-dir "$LABEL_DIR" \
  --rgb-root "$RGB_ROOT" \
  --logdir "$LOGDIR" \
  "$@"
