#!/bin/bash
# P5 B2-final: the multimodal (K-arm) planner on top of the LANE-GRAPH intent, replacing
# every mushy-blob ingredient the earlier B2 attempts (p5_stepB2, p5_stepB2_v2) used.
#     Step A -> B1 -> B1' (3-cam blob) -> B1''' (3-cam lane-graph) -> B2-final  <- here
#
# vs train_p5_stepB2_ddp.sh (B2_v2), four things change -- all of them "stop using the blob":
#   load_file:         p5_stepB1_3cam        -> p5_stepB1_lanegraph  (B1''' beat B1' on every
#                                                split: NEAR ADE 0.0375 vs 0.0452)
#   vlm_intent_ckpt:   vlm_intent_p6_3cam    -> vlm_intent_p6_lanegraph (pred-vs-GT corridor
#                                                IoU 0.741 on junction frames)
#   lanegraph_label_dir: (unset, blob)       -> data/p6/lanegraph_label
#   anchor_cache_dir:  data/p6/anchor_cache  -> data/p6/lanegraph_anchor
#
# That last one is why B2_v2's arms cut across lanes: the blob-skeleton anchors put 3-6 arms
# on 42% of frames (many of them infeasible -- oncoming / cross-lane), while the lane-graph
# anchors give 1 arm on 80%, 2 on 13%, 3 on 7% -- i.e. arms only where the road really forks.
#
# Also new here (uncommitted -> see planning_decoder._multimodal_route_loss):
#   - anchor consistency is now a LOOSE pair of hinges (angular cone route_anchor_tol_deg,
#     one-sided reach route_anchor_min_reach_frac) instead of an endpoint L1. An endpoint L1
#     turns the anchor into a trajectory label and binds control to intent; intent must stay
#     a fuzzy "which branch" and let the planner resolve the path late.
#   - loss_route_collision applies the collision cost to ALL valid arms, not just the winner,
#     so closed-loop late resolution (controller picks the cheapest arm) is meaningful and
#     an arm the angular hinge aims at an obstacle has to route around it.
#
# NB closed-loop: 3-cam drivable features -> the Qwen service must extract the FULL strip
# with the 3-cam drivable prompt (extract_vlm_p6_3cam.py), not the front crop.
#
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_p5_stepB2_final_ddp.sh

source /mmu_mllm_hdd_3/liuzihan08/miniconda3/etc/profile.d/conda.sh
conda activate lead

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export LEAD_PROJECT_ROOT=/mmu_mllm_hdd_3/liuzihan08/vla/lead

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/p5_stepB2_final \
load_file=outputs/local_training/p5_stepB1_lanegraph/model_0019.pth \
image_encoder_pretrained=false \
freeze_backbone=true \
num_route_points_smoothing=40 \
num_route_points_prediction=30 \
route_near_points=10 \
route_far_weight=0.3 \
use_vlm_intent=true \
vlm_intent_ckpt=outputs/local_training/vlm_intent_p6_lanegraph/model_0014.pth \
lanegraph_label_dir=data/p6/lanegraph_label \
vlm_cache_dir=data/p6/vlm_cache_3cam \
vlm_manifest=data/p4/manifest.jsonl \
use_multimodal_intent=true \
multimodal_planner=true \
multimodal_planner_k=6 \
anchor_cache_dir=data/p6/lanegraph_anchor \
route_anchor_loss_weight=1.0 \
route_anchor_tol_deg=25.0 \
route_anchor_min_reach_frac=0.5 \
route_collision_loss_weight=1.0 \
vlm_3cam=true \
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
