#!/usr/bin/env bash
# Fine-tune a multi-arm planner and scene-interacting scorer from LEAD posttrain.
# Qwen runs live in one qwenvl service per training GPU; no vlm_cache is used.
set -euo pipefail
cd "$(dirname "$0")/../.."

LEAD_PYTHON="${P6_JOINT_LEAD_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python}"
QWEN_PYTHON="${P6_JOINT_QWEN_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/qwenvl/bin/python}"
QWEN_MODEL="${P6_JOINT_QWEN_MODEL:-/mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct}"
PRETRAIN="${P6_JOINT_PRETRAIN:-outputs/local_training/posttrain/model_0030.pth}"
RESUME="${P6_JOINT_RESUME:-}"
LOAD_FILE="${RESUME:-$PRETRAIN}"
RESUME_CONFIG=""
if [[ -n "$RESUME" ]]; then
    state_suffix="${RESUME##*/}"
    [[ "$state_suffix" =~ ^model_[0-9]+\.pth$ ]] || {
        echo "P6_JOINT_RESUME must point to a completed model_NNNN.pth checkpoint" >&2
        exit 1
    }
    state_suffix="${state_suffix#model_}"
    for state_kind in optimizer scheduler scaler; do
        [[ -f "$(dirname "$RESUME")/${state_kind}_${state_suffix}" ]] || {
            echo "Missing resume state: $(dirname "$RESUME")/${state_kind}_${state_suffix}" >&2
            exit 1
        }
    done
    RESUME_CONFIG="continue_failed_training=true"
fi
INTENT="${P6_JOINT_INTENT:-outputs/local_training/vlm_intent_p6_online_full_b32_w4/model_best.pth}"
SPLIT="${P6_JOINT_SPLIT:-outputs/local_training/p5_stepB3a_v2_dense_data/route_split}"
LOGDIR="${P6_JOINT_LOGDIR:-outputs/local_training/p6_joint_online}"
FULL_DATA="${P6_JOINT_FULL_DATA:-0}"
BATCH_SIZE="${P6_JOINT_BATCH_SIZE:-64}"
WORKERS="${P6_JOINT_WORKERS:-2}"
PREFETCH="${P6_JOINT_PREFETCH_FACTOR:-2}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a DEVICES <<< "$GPUS"

[[ "$FULL_DATA" == 0 || "$FULL_DATA" == 1 ]] || {
    echo "P6_JOINT_FULL_DATA must be 0 or 1" >&2
    exit 1
}
for value in "$BATCH_SIZE" "$WORKERS" "$PREFETCH"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "Batch size, workers, and prefetch factor must be positive integers" >&2
        exit 1
    }
done
(( BATCH_SIZE % ${#DEVICES[@]} == 0 )) || {
    echo "P6_JOINT_BATCH_SIZE must be divisible by the GPU count" >&2
    exit 1
}
MULTIMODAL_SPEED="${P6_JOINT_MULTIMODAL_SPEED:-false}"
[[ "$MULTIMODAL_SPEED" == true || "$MULTIMODAL_SPEED" == false ]] || {
    echo "P6_JOINT_MULTIMODAL_SPEED must be true or false" >&2
    exit 1
}

ROUTE_POINTS="${P6_JOINT_ROUTE_POINTS:-30}"
if [[ "$ROUTE_POINTS" == 10 ]]; then
    SCORER_STEPS="${P6_JOINT_SCORER_STEPS:-4}"
else
    SCORER_STEPS="${P6_JOINT_SCORER_STEPS:-8}"
fi
[[ "$ROUTE_POINTS" =~ ^[1-9][0-9]*$ && "$SCORER_STEPS" =~ ^[1-9][0-9]*$ ]] || {
    echo "P6_JOINT_ROUTE_POINTS and P6_JOINT_SCORER_STEPS must be positive integers" >&2
    exit 1
}
if (( ROUTE_POINTS < 2 || SCORER_STEPS > 8 || (ROUTE_POINTS <= 10 && SCORER_STEPS > 4) )); then
    echo "10 m routes require at most four 4 Hz scorer states; route needs at least two points" >&2
    exit 1
fi
[[ -x "$LEAD_PYTHON" && -x "$QWEN_PYTHON" && -d "$QWEN_MODEL" \
   && -f "$LOAD_FILE" && -f "$(dirname "$LOAD_FILE")/config.json" \
   && -f "$INTENT" && -f "$SPLIT/train.jsonl" \
   && -d data/p6/lanegraph_label && -d data/carla_leaderboard2/data \
   && -d data/carla_leaderboard2/buckets ]] || {
    echo "Missing joint training Python, model, pretrain/config, intent, split, or CARLA data" >&2
    exit 1
}

if [[ -n "$RESUME" ]]; then
    "$LEAD_PYTHON" - "$(dirname "$RESUME")/config.json" "$MULTIMODAL_SPEED" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    saved = json.load(stream)
if bool(saved.get("multimodal_target_speed", False)) != (sys.argv[2] == "true"):
    raise SystemExit("Cannot change multimodal_target_speed during resume; start a new run from pretrain with a new logdir")
PY
fi

"$LEAD_PYTHON" - "$SPLIT/train.jsonl" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1]) as stream:
    entry = json.loads(next(stream))
rgb = Path(entry["src"])
route = rgb.parent.parent
frame = rgb.stem
required = [rgb]
required += [route / subdir / (frame + suffix) for subdir, suffix in (
    ("metas", ".pkl"), ("bboxes", ".pkl"), ("lidar", ".laz"),
    ("radar", ".npz"), ("hdmap", ".png"),
    ("semantics", ".png"), ("depth", ".png"),
)]
required.append(Path("data/p6/lanegraph_label") / entry["scenario"] / entry["route"] / (frame + ".npy"))
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("Joint training data is incomplete; sample files missing:\n" + "\n".join(missing))
print("joint data preflight passed:", route)
PY

mkdir -p "$LOGDIR"
if [[ "$FULL_DATA" == 1 ]]; then
    [[ -f "$SPLIT/heldout.jsonl" && -f "$SPLIT/metadata.json" ]] || {
        echo "Full-data mode requires train, heldout, and metadata files in $SPLIT" >&2
        exit 1
    }
    TRAIN_MANIFEST="$LOGDIR/all_frames.jsonl"
    "$LEAD_PYTHON" scripts/p4/build_p6_joint_all_manifest.py \
        --split "$SPLIT" --out "$TRAIN_MANIFEST"
    VAL_CONFIG=""
    echo "P6 full-data training: all manifest frames; no overlapping held-out validation"
else
    [[ -f "$SPLIT/heldout.jsonl" ]] || {
        echo "Missing heldout manifest: $SPLIT/heldout.jsonl" >&2
        exit 1
    }
    TRAIN_MANIFEST="$SPLIT/train.jsonl"
    VAL_CONFIG="joint_val_manifest=$SPLIT/heldout.jsonl joint_val_max_frames=${P6_JOINT_VAL_FRAMES:-5000}"
fi
socket_dir=$(mktemp -d "${TMPDIR:-/tmp}/lead_joint_vlm.XXXXXX")
service_pids=()
cleanup() {
    for pid in "${service_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${service_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    for ((i=0; i<${#DEVICES[@]}; i++)); do
        if [[ -S "$socket_dir/rank$i.sock" ]]; then rm "$socket_dir/rank$i.sock"; fi
    done
    rmdir "$socket_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export LEAD_TRAIN_TIMING="${P6_JOINT_TIMING:-0}"
export LEAD_TIMING_EVERY="${P6_JOINT_TIMING_EVERY:-20}"
export LEAD_TIMING_SYNC="${P6_JOINT_TIMING_SYNC:-1}"
export LEAD_TIMING_DIR="$LOGDIR/timing"
export LEAD_TIMING_RUN_ID="$(date -u +%Y%m%dT%H%M%S)_$$"
export NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for ((i=0; i<${#DEVICES[@]}; i++)); do
    CUDA_VISIBLE_DEVICES="${DEVICES[$i]}" LEAD_TIMING_RANK="$i" "$QWEN_PYTHON" -u -m lead.inference.vlm_service \
        --model "$QWEN_MODEL" --socket "$socket_dir/rank$i.sock" \
        --prompt-mode 3cam_drivable --device cuda:0 \
        > "$LOGDIR/qwen_rank$i.log" 2>&1 &
    service_pids+=("$!")
done
for ((attempt=0; attempt<600; attempt++)); do
    ready=0
    for ((i=0; i<${#DEVICES[@]}; i++)); do
        [[ -S "$socket_dir/rank$i.sock" ]] && ((ready+=1)) || true
        kill -0 "${service_pids[$i]}" 2>/dev/null || {
            echo "Qwen rank $i exited; inspect $LOGDIR/qwen_rank$i.log" >&2
            exit 1
        }
    done
    [[ "$ready" -eq "${#DEVICES[@]}" ]] && break
    sleep 1
done
[[ "$ready" -eq "${#DEVICES[@]}" ]] || { echo "Qwen service startup timed out" >&2; exit 1; }

export LEAD_PROJECT_ROOT="$PWD"
export LEAD_QWEN_PREFETCH="${P6_JOINT_QWEN_PREFETCH:-0}"
[[ "$LEAD_QWEN_PREFETCH" == 0 || "$LEAD_QWEN_PREFETCH" == 1 ]] || {
    echo "P6_JOINT_QWEN_PREFETCH must be 0 or 1" >&2; exit 1
}
export LEAD_TRAINING_CONFIG="logdir=$LOGDIR \
load_file=$LOAD_FILE $RESUME_CONFIG image_encoder_pretrained=false \
use_planning_decoder=true use_vlm_intent=true vlm_intent_ckpt=$INTENT \
online_joint_training=true online_vlm_socket_dir=$socket_dir \
vlm_manifest=$TRAIN_MANIFEST $VAL_CONFIG \
lanegraph_label_dir=data/p6/lanegraph_label \
use_multimodal_intent=true multimodal_planner=true multimodal_planner_k=6 \
joint_scorer_replaces_conf=true route_select_by_conf=false \
multimodal_target_speed=$MULTIMODAL_SPEED \
num_route_points_smoothing=40 num_route_points_prediction=$ROUTE_POINTS \
route_near_points=10 route_far_weight=0.3 \
joint_scorer_steps=$SCORER_STEPS \
use_control_conditioning=true use_route_corridor_loss=true \
use_collision_cost=true use_sensor_perburtation=false vlm_3cam=true \
use_persistent_cache=false use_training_session_cache=false \
prefetch_factor=$PREFETCH joint_dataloader_workers=$WORKERS \
carla_num_samples=${P6_JOINT_TRAIN_FRAMES:-0} \
batch_size=$BATCH_SIZE lr=${P6_JOINT_LR:-0.0003} \
epochs=${P6_JOINT_EPOCHS:-20}"

LEAD_LIBNVRTC_DIR="${P6_JOINT_LEAD_LIBNVRTC_DIR:-$("$LEAD_PYTHON" -c 'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_nvrtc/lib")')}"
if [[ -f "$LEAD_LIBNVRTC_DIR/libnvrtc.so.12" ]]; then
    LEAD_LIBRARY_PATH="$LEAD_LIBNVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "LEAD libnvrtc: $LEAD_LIBNVRTC_DIR/libnvrtc.so.12"
else
    LEAD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    echo "Warning: LEAD libnvrtc.so.12 not found in $LEAD_LIBNVRTC_DIR" >&2
fi
CUDA_VISIBLE_DEVICES="$GPUS" LD_LIBRARY_PATH="$LEAD_LIBRARY_PATH" "$(dirname "$LEAD_PYTHON")/torchrun" \
    --standalone --nnodes=1 --nproc_per_node="${#DEVICES[@]}" \
    --max_restarts=0 --no-python "$LEAD_PYTHON" lead/training/train.py
