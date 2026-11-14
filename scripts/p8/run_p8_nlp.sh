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
    train-baseline) exec bash scripts/p8/train_p8_baseline.sh ;;
    train-single) exec bash scripts/p8/train_p8_single_cached.sh ;;
    extract-full) P8_FRAME_FRACTION=1.0 P8_HIDDEN_CACHE=${P8_HIDDEN_CACHE:-data/p8/hidden_cache_full} \
        P8_HIDDEN_MANIFEST=${P8_HIDDEN_MANIFEST:-data/p8/hidden_manifest_full.jsonl} \
        exec bash scripts/p8/extract_p8_hidden_cache.sh ;;
    extract-full-perturbated) P8_FRAME_FRACTION=1.0 P8_PERTURBATED=true \
        P8_HIDDEN_CACHE=${P8_HIDDEN_CACHE:-data/p8/hidden_cache_full_perturbated} \
        P8_HIDDEN_MANIFEST=${P8_HIDDEN_MANIFEST:-data/p8/hidden_manifest_full_perturbated.jsonl} \
        exec bash scripts/p8/extract_p8_hidden_cache.sh ;;
    train-full) exec bash scripts/p8/train_p8_full.sh ;;
    *) echo 'Usage: bash scripts/p8/run_p8_nlp.sh {extract|extract-full|extract-full-perturbated|train|train-single|train-baseline|train-full}' >&2; exit 2 ;;
esac
