#!/usr/bin/env bash
# Generate exact-frame lane-graph intent labels and anchors for the dense manifest.
set -euo pipefail

cd "$(dirname "$0")/../.."

PY=/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python
MANIFEST=outputs/local_training/p5_stepB3a_v2_dense_data/manifest_stride1.jsonl
LOG_DIR=outputs/local_training/lanegraph_intent_full_logs
SHARDS="${LANEGRAPH_SHARDS:-4}"

[[ -x "$PY" && -f "$MANIFEST" ]] || {
  echo "Missing LEAD Python or dense manifest" >&2
  exit 1
}
[[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "LANEGRAPH_SHARDS must be a positive integer" >&2
  exit 1
}
mkdir -p "$LOG_DIR"

pids=()
for ((shard=0; shard<SHARDS; shard++)); do
  CUDA_VISIBLE_DEVICES="" "$PY" -u scripts/p4/precompute_lanegraph_intent.py \
    --manifest "$MANIFEST" \
    --out-label data/p6/lanegraph_label \
    --out-anchor data/p6/lanegraph_anchor \
    --num-shards "$SHARDS" --shard "$shard" \
    > "$LOG_DIR/shard_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "lane-graph shard $shard: pid=${pids[$shard]} log=$LOG_DIR/shard_${shard}.log"
done

failed=0
for ((shard=0; shard<SHARDS; shard++)); do
  if ! wait "${pids[$shard]}"; then
    echo "lane-graph shard $shard failed; see $LOG_DIR/shard_${shard}.log" >&2
    failed=1
  fi
done
echo "Full lane-graph labels finished: failed=$failed"
exit "$failed"
