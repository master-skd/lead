#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

exec >> outputs/local_training/p7_wandb_resume.log 2>&1

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY before running resume_train_wandb.sh}"
unset WANDB_MODE

CUDA_VISIBLE_DEVICES=0,1,2,3 \
P7_LEAD_PYTHON="$PWD/.venv/bin/python" \
P7_POSTTRAIN="$PWD/outputs/local_training/p7_cached_tenth/model_0009.pth" \
P7_MANIFEST="$PWD/data/p7/intent_manifest_tenth.jsonl" \
P7_INTENT_CACHE="$PWD/data/p7/intent_cache_tenth/intents.sqlite" \
P7_LOGDIR="$PWD/outputs/local_training/p7_cached_tenth" \
P7_BATCH_SIZE=32 P7_WORKERS=8 P7_EPOCHS=20 \
P7_CONTINUE_FAILED_TRAINING=true \
LD_PRELOAD="$PWD/.qwenvenv/lib/glibc_single_threaded_shim_unversioned.so" \
bash scripts/p7/train_p7_cached.sh
