#!/usr/bin/env bash
# Same fixed frames/settings as P8 single-route, original planner without Qwen.
set -euo pipefail
cd "$(dirname "$0")/../.."
export P8_PLANNER_BASELINE=true
export P8_FRAME_MANIFEST=${P8_FRAME_MANIFEST:-outputs/local_training/p8_hidden_single_cached_tenth/p8_train_frames.txt}
export P8_LOGDIR=${P8_LOGDIR:-outputs/local_training/p8_planner_only_tenth}
exec bash scripts/p8/train_p8_hidden.sh
