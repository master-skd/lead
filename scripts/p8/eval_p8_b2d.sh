#!/usr/bin/env bash
# Evaluate one P8 hidden-intent checkpoint with same-frame single-forward Qwen-Drive features.
# Usage: bash scripts/p8/eval_p8_b2d.sh <model_NNNN.pth> "0 1 2 3 4 5 6 7" <tag>
set -euo pipefail

CKPT_FILE="${1:?provide one completed P8 model_NNNN.pth checkpoint}"
GPU_LIST="${2:-0}"
TAG="${3:?provide a unique output tag}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "tag must use letters, digits, _ or -" >&2; exit 1; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LEAD_PYTHON="${P8_B2D_LEAD_PYTHON:-/mmu_nlp_hdd/liuzihan08/miniconda3/envs/lead/bin/python}"
QWEN_PYTHON="${P8_B2D_QWEN_PYTHON:-/mmu_nlp_hdd/liuzihan08/miniconda3/envs/qwendrive/bin/python}"
QWEN_MODEL="${P8_B2D_QWEN_MODEL:-/mmu_nlp_hdd/liuzihan08/models/Qwen-Drive-1.0-4B}"
[[ -x "$LEAD_PYTHON" && -x "$QWEN_PYTHON" && -d "$QWEN_MODEL" ]] || {
    echo "Set P8_B2D_LEAD_PYTHON, P8_B2D_QWEN_PYTHON and P8_B2D_QWEN_MODEL to existing paths" >&2
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
HIDDEN_DIM=$("$LEAD_PYTHON" - "$CONFIG_FILE" <<'PYCFG'
import json,sys
from lead.inference.p8_evaluation import validate_config
print(validate_config(json.load(open(sys.argv[1]))))
PYCFG
)
echo "P8 checkpoint verified: K paired routes/speeds; selection=confidence; hidden_dim=$HIDDEN_DIM"
TRAIN_SIGNATURE="${P8_B2D_SIGNATURE:-$(dirname "$CKPT_FILE")/p8_hidden_cache_signature.json}"
if [[ -n "${P8_B2D_SIGNATURE:-}" && ! -f "$TRAIN_SIGNATURE" ]]; then
    echo "P8_B2D_SIGNATURE does not exist: $TRAIN_SIGNATURE" >&2; exit 1
fi

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
EVAL_CKPT_DIR="${P8_B2D_EVAL_CHECKPOINT_ROOT:-$REPO_ROOT/outputs/p8_b2d_eval_checkpoints}/$TAG"
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
if [[ -f "$TRAIN_SIGNATURE" ]]; then
    cp -- "$TRAIN_SIGNATURE" "$EVAL_CKPT_DIR/p8_hidden_cache_signature.json"
else
    echo "No training feature signature provided; model/processor provenance will not be verified" >&2
fi
if [[ "${P8_B2D_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    echo "Preflight OK: $EVAL_CKPT_DIR"
    exit 0
fi

export LEAD_PROJECT_ROOT="$REPO_ROOT"
export B2D_LEAD_PYTHON="$LEAD_PYTHON"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Do not inherit training-time overrides such as load_file/logdir/batch_size.
export LEAD_TRAINING_CONFIG=""
export ROUTE_TIMEOUT="${P8_B2D_ROUTE_TIMEOUT:-86400}"
export B2D_AGENT_TIMEOUT="${P8_B2D_AGENT_TIMEOUT:-600}"
export LEAD_CLOSED_LOOP_CONFIG="${LEAD_CLOSED_LOOP_CONFIG:-} p8_hidden_timeout=$B2D_AGENT_TIMEOUT strict_weight_load=true"
unset B2D_QWEN_SOCKET_DIR B2D_P7_SOCKET_DIR P6_EVAL_FORCE_SLOT0
LEAD_NVRTC_DIR="${P8_B2D_LEAD_LIBNVRTC_DIR:-$("$LEAD_PYTHON" -c 'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_nvrtc/lib")')}"
if [[ -f "$LEAD_NVRTC_DIR/libnvrtc.so.12" ]]; then
    export B2D_LEAD_LIBRARY_PATH="$LEAD_NVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    export B2D_LEAD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
fi
SOCKET_DIR="$(mktemp -d "${TMPDIR:-/tmp}/p8_b2d_${TAG}.XXXXXX")"
export B2D_P8_SOCKET_DIR="$SOCKET_DIR"
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

LOG_DIR="$REPO_ROOT/outputs/p8_b2d_eval_logs/$TAG"
mkdir -p "$LOG_DIR"
signature_args=()
if [[ -f "$TRAIN_SIGNATURE" ]]; then
    signature_args=(--training-signature "$TRAIN_SIGNATURE")
fi
for gpu in "${GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$QWEN_PYTHON" -u -m lead.inference.qwendrive_hidden_service \
        --model "$QWEN_MODEL" --socket "$SOCKET_DIR/gpu${gpu}.sock" \
        --device cuda:0 \
        --microbatch 1 --expected-hidden-dim "$HIDDEN_DIM" \
        "${signature_args[@]}" \
        > "$LOG_DIR/qwen_gpu${gpu}.log" 2>&1 &
    QWEN_PIDS+=("$!")
done
for index in "${!GPUS[@]}"; do
    gpu="${GPUS[$index]}"
    ready=0
    for ((attempt=0; attempt<${P8_B2D_SERVICE_WAIT:-600}; attempt++)); do
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
"$LEAD_PYTHON" - "outputs/local_evaluation_$TAG" <<'PYRESUME'
import json,sys
from lead.inference.p8_evaluation import archive_invalid_results
for record in archive_invalid_results(sys.argv[1]):
    print('Retrying unfinished route: '+json.dumps(record),flush=True)
PYRESUME
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
"$LEAD_PYTHON" - "outputs/local_evaluation_$TAG" <<'PYRESULT'
import sys
from pathlib import Path
from lead.inference.p8_evaluation import result_error
bad=[]
for path in Path(sys.argv[1]).glob('*/checkpoint_endpoint.json'):
    reason=result_error(path)
    if reason: bad.append(f'{path}: {reason}')
if bad:
    raise SystemExit('Invalid route results; rerun with the same tag to retry:\n'+'\n'.join(bad))
PYRESULT
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
