#!/usr/bin/env bash
# Eight independent GPU workers; rerun the same command to resume JSONL shards.
set -euo pipefail
cd "$(dirname "$0")/../.."
LEAD_PYTHON=${P7_LEAD_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python}
QWEN_PYTHON=${P7_QWEN_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/qwendrive_official/bin/python}
QWEN_MODEL=${P7_QWEN_MODEL:-$PWD/models/Qwen-Drive-1.0-4B}
MANIFEST=${P7_MANIFEST:-data/p7/intent_manifest.jsonl}
OUT=${P7_INTENT_DIR:-data/p7/intent_cache}
POSTTRAIN=${P7_POSTTRAIN:-outputs/local_training/posttrain/model_0030.pth}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a DEVICES <<< "$GPUS"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
if [[ ! -f "$MANIFEST" ]]; then
    "$LEAD_PYTHON" scripts/p7/export_p7_manifest.py \
        --posttrain-config "$(dirname "$POSTTRAIN")/config.json" \
        --output "$MANIFEST" --max-frames "${P7_GENERATE_FRAMES:-0}"
fi
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
retry=()
[[ ${P7_RETRY_FAILED:-0} == 1 ]] && retry+=(--retry-failed)
for ((i=0; i<${#DEVICES[@]}; i++)); do
    CUDA_VISIBLE_DEVICES="${DEVICES[$i]}" "$QWEN_PYTHON" -u scripts/p7/generate_p7_intents.py generate \
        --manifest "$MANIFEST" --output-dir "$OUT" --model "$QWEN_MODEL" \
        --data-root "$PWD/data/carla_leaderboard2/data" \
        --num-shards "${#DEVICES[@]}" --shard "$i" \
        --analysis-tokens "${P7_ANALYSIS_TOKENS:-512}" \
        --answer-tokens "${P7_ANSWER_TOKENS:-384}" "${retry[@]}" \
        > "$OUT/generate_rank$i.log" 2>&1 &
    pids+=("$!")
done
# Collect worker exit codes; a failed worker prevents publishing an incomplete index.
for pid in "${pids[@]}"; do wait "$pid"; done
pids=()
"$LEAD_PYTHON" scripts/p7/generate_p7_intents.py index \
    --manifest "$MANIFEST" --shard-dir "$OUT" --output "$OUT/intents.sqlite"
