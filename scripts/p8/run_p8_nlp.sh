#!/usr/bin/env bash
# P8 convenience entry for the nlp_hdd checkout and its local environments.
set -euo pipefail
cd /mmu_nlp_hdd/liuzihan08/lead
export P8_LEAD_PYTHON=${P8_LEAD_PYTHON:-/mmu_nlp_hdd/liuzihan08/miniconda3/envs/lead/bin/python}
export P8_QWEN_PYTHON=${P8_QWEN_PYTHON:-/mmu_nlp_hdd/liuzihan08/miniconda3/envs/qwendrive/bin/python}
export P8_QWEN_MODEL=${P8_QWEN_MODEL:-/mmu_nlp_hdd/liuzihan08/models/Qwen-Drive-1.0-4B}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
case "${1:-}" in
    extract) exec bash scripts/p8/extract_p8_hidden_cache.sh ;;
    train) exec bash scripts/p8/train_p8_cached.sh ;;
    *) echo 'Usage: bash scripts/p8/run_p8_nlp.sh {extract|train}' >&2; exit 2 ;;
esac
