#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

exec >> outputs/local_training/p7_sampled_train.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
P7_LEAD_PYTHON="$PWD/.venv/bin/python" \
P7_QWEN_PYTHON="$PWD/.qwenvenv/bin/python" \
P7_QWEN_MODEL="$PWD/models/Qwen-Drive-1.0-4B" \
P7_POSTTRAIN="$PWD/outputs/local_training/posttrain/model_0030.pth" \
P7_MANIFEST="$PWD/data/p7/intent_manifest_tenth.jsonl" \
P7_INTENT_DIR="$PWD/data/p7/intent_cache_tenth" \
P7_GENERATION_BATCH_SIZE=64 P7_PREPROCESS_WORKERS=8 \
LD_PRELOAD="$PWD/.qwenvenv/lib/glibc_single_threaded_shim_unversioned.so" \
bash scripts/p7/generate_p7_intents.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
P7_LEAD_PYTHON="$PWD/.venv/bin/python" \
P7_POSTTRAIN="$PWD/outputs/local_training/posttrain/model_0030.pth" \
P7_MANIFEST="$PWD/data/p7/intent_manifest_tenth.jsonl" \
P7_INTENT_CACHE="$PWD/data/p7/intent_cache_tenth/intents.sqlite" \
P7_LOGDIR="$PWD/outputs/local_training/p7_cached_tenth" \
P7_BATCH_SIZE=32 P7_WORKERS=8 P7_EPOCHS=20 \
WANDB_MODE=disabled \
LD_PRELOAD="$PWD/.qwenvenv/lib/glibc_single_threaded_shim_unversioned.so" \
bash scripts/p7/train_p7_cached.sh
