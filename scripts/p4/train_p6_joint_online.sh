#!/usr/bin/env bash
# Train a fresh multi-arm planner and scene-interacting scorer from LEAD pretrain.
# Qwen runs live in one qwenvl service per training GPU; no vlm_cache is used.
set -euo pipefail
cd "$(dirname "$0")/../.."

LEAD_PYTHON="${P6_JOINT_LEAD_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python}"
QWEN_PYTHON="${P6_JOINT_QWEN_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/qwenvl/bin/python}"
QWEN_MODEL="${P6_JOINT_QWEN_MODEL:-/mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct}"
PRETRAIN="${P6_JOINT_PRETRAIN:-outputs/local_training/pretrain/model_0030.pth}"
INTENT="${P6_JOINT_INTENT:-outputs/local_training/vlm_intent_p6_online_full_b32_w4/model_best.pth}"
SPLIT="${P6_JOINT_SPLIT:-outputs/local_training/p5_stepB3a_v2_dense_data/route_split}"
LOGDIR="${P6_JOINT_LOGDIR:-outputs/local_training/p6_joint_online}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a DEVICES <<< "$GPUS"

[[ -x "$LEAD_PYTHON" && -x "$QWEN_PYTHON" && -d "$QWEN_MODEL" \
   && -f "$PRETRAIN" && -f "$(dirname "$PRETRAIN")/config.json" \
   && -f "$INTENT" && -f "$SPLIT/train.jsonl" \
   && -d data/p6/lanegraph_label && -d data/carla_leaderboard2/data \
   && -d data/carla_leaderboard2/buckets ]] || {
    echo "Missing joint training Python, model, pretrain/config, intent, split, or CARLA data" >&2
    exit 1
}

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
export NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for ((i=0; i<${#DEVICES[@]}; i++)); do
    CUDA_VISIBLE_DEVICES="${DEVICES[$i]}" "$QWEN_PYTHON" -u -m lead.inference.vlm_service \
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
export LEAD_TRAINING_CONFIG="logdir=$LOGDIR \
load_file=$PRETRAIN image_encoder_pretrained=false \
use_planning_decoder=true use_vlm_intent=true vlm_intent_ckpt=$INTENT \
online_joint_training=true online_vlm_socket_dir=$socket_dir \
vlm_manifest=$SPLIT/train.jsonl joint_val_manifest=$SPLIT/heldout.jsonl \
joint_val_max_frames=${P6_JOINT_VAL_FRAMES:-5000} \
lanegraph_label_dir=data/p6/lanegraph_label \
use_multimodal_intent=true multimodal_planner=true multimodal_planner_k=6 \
joint_scorer_replaces_conf=true route_select_by_conf=false \
num_route_points_smoothing=40 num_route_points_prediction=30 \
route_near_points=10 route_far_weight=0.3 \
use_control_conditioning=true use_route_corridor_loss=true \
use_collision_cost=true use_sensor_perburtation=false vlm_3cam=true \
use_persistent_cache=false use_training_session_cache=false \
prefetch_factor=2 joint_dataloader_workers=${P6_JOINT_WORKERS:-2} \
carla_num_samples=${P6_JOINT_TRAIN_FRAMES:-0} \
batch_size=${P6_JOINT_BATCH_SIZE:-64} lr=${P6_JOINT_LR:-0.0003} \
epochs=${P6_JOINT_EPOCHS:-20}"

CUDA_VISIBLE_DEVICES="$GPUS" "$(dirname "$LEAD_PYTHON")/torchrun" \
    --standalone --nnodes=1 --nproc_per_node="${#DEVICES[@]}" \
    --max_restarts=0 --no-python "$LEAD_PYTHON" lead/training/train.py
