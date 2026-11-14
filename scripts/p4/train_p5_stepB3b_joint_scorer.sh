#!/usr/bin/env bash
# Train only the B3b joint scorer; frozen B2 corridor model is never updated.
set -euo pipefail

ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
PYTHON="${B3B_PYTHON:-/mmu_mllm_hdd_3/liuzihan08/miniconda3/envs/lead/bin/python}"
SCENE_ROOT="${B3B_SCENE_ROOT:-${ROOT}/outputs/local_training/p5_stepB3b_joint_scorer/scene_cache}"
LABEL_ROOT="${B3B_LABEL_ROOT:-${ROOT}/outputs/local_training/p5_stepB3b_joint_scorer/labels}"
OUTPUT="${B3B_TRAIN_OUTPUT:-${ROOT}/outputs/local_training/p5_stepB3b_joint_scorer/training}"
GPUS="${B3B_TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
EPOCHS="${B3B_TRAIN_EPOCHS:-8}"
BATCH="${B3B_TRAIN_BATCH_SIZE:-16}"
IFS=',' read -r -a DEVICES <<< "${GPUS}"
cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node="${#DEVICES[@]}" \
    scripts/p4/train_b3b_joint_scorer.py \
    --scene-root "${SCENE_ROOT}" --label-root "${LABEL_ROOT}" \
    --output "${OUTPUT}" --epochs "${EPOCHS}" --batch-size "${BATCH}" "$@"
