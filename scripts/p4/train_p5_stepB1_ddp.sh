#!/bin/bash
# P5 Step B1: swap the intent SOURCE from single-mode (P4a) to multimodal (P5a drivable
# blob), planner STILL outputs a SINGLE route. Isolates one variable -- "can the planner
# ingest a multimodal blob intent without degrading?" -- before we make it emit K arms (B2).
#
# What changes vs Step A (train_p5_stepA_ddp.sh):
#   - vlm_intent_ckpt: p4a (single-mode) -> p5a_tversky (multimodal drivable blob)
#   - vlm_cache_dir:   data/p4/vlm_cache -> data/p5/vlm_cache (drivable-prompt features)
#   - use_multimodal_intent=true (GT label = drivable blob, for consistency)
#   - load_file:       control_p2_full -> p5_stepA/model_0019.pth (inherit the 30m planner)
# Everything else identical: 30m route head, single-route output, near-weighted loss,
# frozen backbone, small lr. NO WTA / anchors / confidence -- those are B2.
#
# P5a head is single-channel (blob), so intent_adapter (Conv2d 1->D) is UNCHANGED. The
# frozen intent head is loaded from config.vlm_intent_ckpt and RE-loaded after load_file
# (training_utils overrides whatever vlm_intent_decoder weights the Step A ckpt carried).
#
# NB closed-loop: this uses P5a drivable-prompt features -> the Qwen service must run with
# --prompt-mode drivable (NOT command like Step A / p4a_p2).
#
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_p5_stepB1_ddp.sh

source /mmu_mllm_hdd_3/liuzihan08/miniconda3/etc/profile.d/conda.sh
conda activate lead

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export LEAD_PROJECT_ROOT=/mmu_mllm_hdd_3/liuzihan08/vla/lead

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/p5_stepB1 \
load_file=outputs/local_training/p5_stepA/model_0019.pth \
image_encoder_pretrained=false \
freeze_backbone=true \
num_route_points_smoothing=40 \
num_route_points_prediction=30 \
route_near_points=10 \
route_far_weight=0.3 \
use_vlm_intent=true \
vlm_intent_ckpt=outputs/local_training/vlm_intent_p5a_tversky/model_0014.pth \
vlm_cache_dir=data/p5/vlm_cache \
vlm_manifest=data/p4/manifest.jsonl \
use_multimodal_intent=true \
use_intent_decoder=false \
use_planning_decoder=true \
use_control_conditioning=true \
use_collision_cost=true \
batch_size=128 \
lr=3e-5 \
epochs=20"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --no-python \
    python3 lead/training/train.py
