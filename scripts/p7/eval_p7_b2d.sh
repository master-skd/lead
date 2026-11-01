#!/usr/bin/env bash
# Evaluate one P7 semantic-intent checkpoint with online Qwen-Drive JSON.
# Usage: bash scripts/p7/eval_p7_b2d.sh <model_NNNN.pth> "0 1 2 3 4 5 6 7" <tag>
set -euo pipefail

CKPT_FILE="${1:?provide one completed P7 model_NNNN.pth checkpoint}"
GPU_LIST="${2:-0}"
TAG="${3:?provide a unique output tag}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "tag must use letters, digits, _ or -" >&2; exit 1; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
LEAD_PYTHON="${P7_B2D_LEAD_PYTHON:-/mmu_nlp_hdd/liuzihan08/miniconda3/envs/lead/bin/python}"
QWEN_PYTHON="${P7_B2D_QWEN_PYTHON:-/mmu_nlp_hdd/liuzihan08/miniconda3/envs/qwendrive_official/bin/python}"
QWEN_MODEL="${P7_B2D_QWEN_MODEL:-/mmu_nlp_hdd/liuzihan08/models/Qwen-Drive-1.0-4B}"
[[ -x "$LEAD_PYTHON" && -x "$QWEN_PYTHON" && -d "$QWEN_MODEL" ]] || {
    echo "Set P7_B2D_LEAD_PYTHON, P7_B2D_QWEN_PYTHON and P7_B2D_QWEN_MODEL to existing paths" >&2
    exit 1
}
[[ -n "${CARLA_ROOT:-}" && -x "$CARLA_ROOT/CarlaUE4.sh" ]] || {
    echo "Set CARLA_ROOT to the local CARLA installation" >&2; exit 1
}
[[ -d data/benchmark_routes/bench2drive ]] || {
    echo "Bench2Drive routes are missing" >&2; exit 1
}
CKPT_FILE="$(realpath "$CKPT_FILE")"
CONFIG_FILE="$(dirname "$CKPT_FILE")/config.json"
[[ -f "$CKPT_FILE" && -f "$CONFIG_FILE" ]] || {
    echo "checkpoint and sibling config.json are both required" >&2; exit 1
}

"$LEAD_PYTHON" - "$CONFIG_FILE" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1], encoding="utf-8") as stream:
    cfg = json.load(stream)
required = {
    "use_semantic_intent_planner": True, "multimodal_planner": True,
    "multimodal_target_speed": True, "joint_scorer_replaces_conf": False,
    "use_vlm_intent": False, "use_intent_decoder": False, "use_control_conditioning": False,
}
bad = {key: (cfg.get(key, False), value) for key, value in required.items()
       if cfg.get(key, False) is not value}
if bad:
    raise SystemExit(f"This evaluator requires a P7 semantic K-route/K-speed checkpoint: {bad}")
print("P7 semantic checkpoint verified; selection=masked route confidence, paired speed")
PY

read -r -a GPUS <<< "${GPU_LIST//,/ }"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "empty GPU list" >&2; exit 1; }
declare -A GPU_SEEN=()
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ && -z "${GPU_SEEN[$gpu]:-}" ]] || {
        echo "GPU list must contain unique non-negative numeric IDs" >&2; exit 1
    }
    GPU_SEEN[$gpu]=1
done

# The agent scans model*.pth. Expose exactly one checkpoint to avoid accidentally
# ensembling multiple epochs. Keep completed model files in the training directory.
EVAL_CKPT_DIR="${P7_B2D_EVAL_CHECKPOINT_ROOT:-$REPO_ROOT/outputs/p7_b2d_eval_checkpoints}/$TAG"
mkdir -p "$EVAL_CKPT_DIR"
for pair in "model.pth:$CKPT_FILE" "config.json:$CONFIG_FILE"; do
    name="${pair%%:*}"
    target="${pair#*:}"
    if [[ -e "$EVAL_CKPT_DIR/$name" || -L "$EVAL_CKPT_DIR/$name" ]]; then
        [[ "$(realpath "$EVAL_CKPT_DIR/$name")" == "$target" ]] || {
            echo "$EVAL_CKPT_DIR/$name points to a different file; use a new tag" >&2
            exit 1
        }
    else
        ln -s "$target" "$EVAL_CKPT_DIR/$name"
    fi
done
if [[ "${P7_B2D_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    echo "Preflight OK: $EVAL_CKPT_DIR"
    exit 0
fi

export LEAD_PROJECT_ROOT="$REPO_ROOT"
export B2D_LEAD_PYTHON="$LEAD_PYTHON"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Do not inherit training-time overrides such as load_file/logdir/batch_size.
export LEAD_TRAINING_CONFIG=""
export ROUTE_TIMEOUT="${P7_B2D_ROUTE_TIMEOUT:-86400}"
export B2D_AGENT_TIMEOUT="${P7_B2D_AGENT_TIMEOUT:-600}"
export LEAD_CLOSED_LOOP_CONFIG="${LEAD_CLOSED_LOOP_CONFIG:-} p7_intent_timeout=$B2D_AGENT_TIMEOUT"
unset B2D_QWEN_SOCKET_DIR P6_EVAL_FORCE_SLOT0
LEAD_NVRTC_DIR="${P7_B2D_LEAD_LIBNVRTC_DIR:-$("$LEAD_PYTHON" -c 'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_nvrtc/lib")')}"
if [[ -f "$LEAD_NVRTC_DIR/libnvrtc.so.12" ]]; then
    export B2D_LEAD_LIBRARY_PATH="$LEAD_NVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    export B2D_LEAD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
fi
SOCKET_DIR="$(mktemp -d "${TMPDIR:-/tmp}/p7_b2d_${TAG}.XXXXXX")"
export B2D_P7_SOCKET_DIR="$SOCKET_DIR"
QWEN_PIDS=()
cleanup() {
    for pid in "${QWEN_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${QWEN_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    for gpu in "${GPUS[@]}"; do
        [[ ! -S "$SOCKET_DIR/gpu${gpu}.sock" ]] || rm -- "$SOCKET_DIR/gpu${gpu}.sock"
    done
    rmdir "$SOCKET_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

LOG_DIR="$REPO_ROOT/outputs/p7_b2d_eval_logs/$TAG"
mkdir -p "$LOG_DIR"
for gpu in "${GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$QWEN_PYTHON" -u -m lead.inference.qwendrive_intent_service \
        --model "$QWEN_MODEL" --socket "$SOCKET_DIR/gpu${gpu}.sock" \
        --device cuda:0 \
        --analysis-tokens "${P7_B2D_ANALYSIS_TOKENS:-512}" \
        --answer-tokens "${P7_B2D_ANSWER_TOKENS:-384}" \
        > "$LOG_DIR/qwen_gpu${gpu}.log" 2>&1 &
    QWEN_PIDS+=("$!")
done
for index in "${!GPUS[@]}"; do
    gpu="${GPUS[$index]}"
    ready=0
    for ((attempt=0; attempt<300; attempt++)); do
        if [[ -S "$SOCKET_DIR/gpu${gpu}.sock" ]]; then ready=1; break; fi
        kill -0 "${QWEN_PIDS[$index]}" 2>/dev/null || break
        sleep 1
    done
    [[ "$ready" == 1 ]] || {
        echo "Qwen GPU $gpu did not become ready; see $LOG_DIR/qwen_gpu${gpu}.log" >&2
        exit 1
    }
done

echo "Qwen-Drive ready on GPUs: $GPU_LIST; evaluating $CKPT_FILE"
bash scripts/eval_bench2drive_local.sh "$EVAL_CKPT_DIR" "$GPU_LIST" "$TAG"

expected=220
if [[ -n "${B2D_ROUTE_IDS:-}" ]]; then
    read -r -a selected <<< "${B2D_ROUTE_IDS//,/ }"
    expected="${#selected[@]}"
fi
actual="$(find "outputs/local_evaluation_$TAG" -mindepth 2 -maxdepth 2 -name checkpoint_endpoint.json -type f | wc -l)"
echo "Bench2Drive route result files: $actual/$expected"
[[ "$actual" -eq "$expected" ]] || {
    echo "Incomplete evaluation; rerun with the SAME tag after inspecting route logs" >&2
    exit 1
}
"$LEAD_PYTHON" - "outputs/local_evaluation_$TAG" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
bad = []
for path in root.glob("*/checkpoint_endpoint.json"):
    try:
        with path.open(encoding="utf-8") as stream:
            records = json.load(stream)["_checkpoint"]["records"]
        if not records:
            raise ValueError("no route records")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        bad.append(f"{path}: {exc}")
if bad:
    raise SystemExit("Invalid route results; inspect before merging:\n" + "\n".join(bad))
PY
if [[ -z "${B2D_ROUTE_IDS:-}" ]]; then
    MERGE_DIR="outputs/b2d_$TAG"
    mkdir -p "$MERGE_DIR"
    for file in "outputs/local_evaluation_$TAG"/*/checkpoint_endpoint.json; do
        route="$(basename "$(dirname "$file")")"
        cp -- "$file" "$MERGE_DIR/$route.json"
    done
    EVALUATION_DATASET=bench2drive "$LEAD_PYTHON" slurm/evaluation/merge_route_json.py -f "$MERGE_DIR"
    echo "Merged results: $MERGE_DIR"
fi
