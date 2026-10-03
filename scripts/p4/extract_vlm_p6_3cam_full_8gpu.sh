#!/usr/bin/env bash
# Extract exact-frame 3-camera VLM features for the full dense manifest, one shard/GPU.
set -euo pipefail

cd "$(dirname "$0")/../.."

PY=/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/qwenvl/bin/python
MANIFEST=outputs/local_training/p5_stepB3a_v2_dense_data/manifest_stride1.jsonl
MODEL=/mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct
LOG_DIR=outputs/local_training/vlm_cache_3cam_full_logs
BATCH_SIZE="${VLM_BATCH_SIZE:-1}"
NUM_WORKERS="${VLM_NUM_WORKERS:-4}"

[[ -x "$PY" && -f "$MANIFEST" && -d "$MODEL" ]] || {
  echo "Missing Qwen Python, full manifest, or model directory" >&2
  exit 1
}
mkdir -p "$LOG_DIR"

pids=()
for gpu in {0..7}; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/p4/extract_vlm_p6_3cam.py \
    --manifest "$MANIFEST" \
    --out data/p6/vlm_cache_3cam \
    --model "$MODEL" \
    --num-shards 8 --shard "$gpu" \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
  "$@" > "$LOG_DIR/shard_${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "GPU $gpu: pid=${pids[$gpu]} log=$LOG_DIR/shard_${gpu}.log"
done

failed=0
for gpu in {0..7}; do
  if ! wait "${pids[$gpu]}"; then
    echo "GPU $gpu failed; see $LOG_DIR/shard_${gpu}.log" >&2
    failed=1
  fi
done
echo "Full VLM cache extraction finished: failed=$failed"
exit "$failed"
