#!/usr/bin/env bash
# Fixed 1/10 manifest -> one resumable packed FP16 shard per GPU -> complete index.
set -euo pipefail
cd "$(dirname "$0")/../.."
LEAD_PYTHON=${P8_LEAD_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python}
QWEN_PYTHON=${P8_QWEN_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/qwendrive_official/bin/python}
QWEN_MODEL=${P8_QWEN_MODEL:-/mmu_mllm_hdd_3/liuzihan08/models/Qwen-Drive-1.0-4B}
POSTTRAIN=${P8_POSTTRAIN:-outputs/local_training/posttrain/model_0030.pth}
CACHE_DIR=${P8_HIDDEN_CACHE:-data/p8/hidden_cache}
MANIFEST=${P8_HIDDEN_MANIFEST:-data/p8/hidden_manifest.jsonl}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a DEVICES <<< "$GPUS"
[[ -x "$LEAD_PYTHON" && -x "$QWEN_PYTHON" ]] || { echo 'Set P8 Python environment paths' >&2; exit 1; }
export LEAD_PROJECT_ROOT="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
"$LEAD_PYTHON" scripts/p8/export_hidden_manifest.py \
    --posttrain-config "$(dirname "$POSTTRAIN")/config.json" --output "$MANIFEST" \
    --fraction "${P8_FRAME_FRACTION:-0.1}" --seed "${P8_SAMPLE_SEED:-2026}"
"$QWEN_PYTHON" scripts/p8/cache_qwendrive_hidden.py prepare \
    --cache-dir "$CACHE_DIR" --manifest "$MANIFEST" --model "$QWEN_MODEL" --shards "${#DEVICES[@]}"
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
for ((rank=0; rank<${#DEVICES[@]}; rank++)); do
    CUDA_VISIBLE_DEVICES="${DEVICES[$rank]}" "$QWEN_PYTHON" -u scripts/p8/cache_qwendrive_hidden.py generate \
        --cache-dir "$CACHE_DIR" --model "$QWEN_MODEL" \
        --data-root "${P8_DATA_ROOT:-data/carla_leaderboard2/data}" --shard "$rank" \
        --batch-size "${P8_QWEN_MICROBATCH:-32}" --workers "${P8_CACHE_WORKERS:-16}" \
        > "$CACHE_DIR/generate_rank$rank.log" 2>&1 &
    pids+=("$!")
done
# Reap every PID explicitly, including generators that finished simultaneously.
# A failed shard stops its peers within one monitor interval.
active=("${pids[@]}")
while (( ${#active[@]} > 0 )); do
    next=()
    for pid in "${active[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            next+=("$pid")
        elif ! wait "$pid"; then
            echo "Hidden extraction failed; inspect $CACHE_DIR/generate_rank*.log. Rerun to resume." >&2
            exit 1
        fi
    done
    active=("${next[@]}")
    pids=("${active[@]}")
    (( ${#active[@]} == 0 )) || sleep 1
done
"$LEAD_PYTHON" scripts/p8/cache_qwendrive_hidden.py index --cache-dir "$CACHE_DIR"
echo "Complete hidden cache: $CACHE_DIR; planner training: bash scripts/p8/train_p8_cached.sh"
