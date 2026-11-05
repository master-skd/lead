#!/usr/bin/env bash
# Original LEAD single route/speed heads and losses + one hidden-state condition.
set -euo pipefail
cd "$(dirname "$0")/../.."
export P8_SINGLE_PATH=true
export P8_LOGDIR=${P8_LOGDIR:-outputs/local_training/p8_hidden_single_cached_tenth}
exec bash scripts/p8/train_p8_cached.sh
