#!/usr/bin/env bash
# Planner/adapter only: no Qwen environment, model or GPU service is loaded.
set -euo pipefail
cd "$(dirname "$0")/../.."
export P8_HIDDEN_CACHE=${P8_HIDDEN_CACHE:-data/p8/hidden_cache}
export P8_LOGDIR=${P8_LOGDIR:-outputs/local_training/p8_hidden_cached_tenth}
export P8_HIDDEN_PREFETCH=${P8_HIDDEN_PREFETCH:-1}
exec bash scripts/p8/train_p8_hidden.sh
