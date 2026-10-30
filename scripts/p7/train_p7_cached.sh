#!/usr/bin/env bash
# Train the semantic encoder and LEAD planner without loading Qwen-Drive.
set -euo pipefail
export P7_INTENT_CACHE=${P7_INTENT_CACHE:-data/p7/intent_cache/intents.sqlite}
export P7_MANIFEST=${P7_MANIFEST:-data/p7/intent_manifest.jsonl}
export P7_LOGDIR=${P7_LOGDIR:-outputs/local_training/p7_cached}
exec bash "$(dirname "$0")/train_p7_hf_online.sh"
