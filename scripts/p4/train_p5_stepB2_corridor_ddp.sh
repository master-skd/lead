#!/bin/bash
# P5 B2-corridor: same K-arm planner as B2-final, plus the two loss terms that fix what
# B2-final's own arms turned out to be doing.
#     Step A -> B1 -> B1' -> B1''' (3-cam lane-graph) -> B2-final -> B2-corridor  <- here
#
# Same load_file as B2-final ON PURPOSE: both start from the identical frozen B1''' planner,
# so term5/term6 are the ONLY variable and any change in arm quality is attributable.
# StepA and B1''' are NOT retrained -- both have multimodal_planner false/absent, so the new
# terms are weight-gated to exactly 0 there and a rerun would be bit-identical.
#
# WHY THIS RUN EXISTS. B2-final scored well (full 96873-frame open loop: ALL 0.1274, NEAR
# 0.0370, beating B1''''s 0.1433 / 0.0375) and did NOT collapse (30.2 m endpoint spread over
# 17698 forking frames). But visualising the arms showed every NON-WINNER arm driving off the
# road, and three measurements explain why the losses never noticed:
#   1. loss_route_anchor hit exactly 0.0000 at step ~4561 and stayed there for 70% of
#      training -- its hinges only constrain BEARING and REACH, never "stay on the road".
#   2. loss_route_collision is INVERTED on these arms (non-winner 0.014 vs winner 0.067):
#      its danger field is built from OBSTACLE_CLASSES (vehicles/walkers), so grass and
#      oncoming lanes are free, and an off-road arm scores as the SAFER one.
#   3. ADE only ever measures the winner, so an off-road non-winner arm costs nothing.
#   4. Zeroing the VLM intent feature moved the arms by 0.065 m (winner: 0.010 m) -- the
#      planner had learned to IGNORE the lane-graph corridor entirely, because the anchor
#      embedding (sin/cos/reach/valid) hands it the same "which way" more cheaply. So the
#      corridor has been in the config since B1''' yet never touched the route gradient.
#
# term 5 (use_route_corridor_loss) is what closes that: it puts the lane-graph corridor
# DIRECTLY on the route output, constraining arm SHAPE -- the part the anchor's 4 numbers
# cannot express. Measured on B2-final's own arms with the shipped field: GT expert route
# 0.018, winner 0.020, non-winner 0.365 (18x GT, 100% of them). It agrees with the expert and
# charges only the off-road arms. Deliberately weak (union of L/S/R arms at lane width, not a
# centreline): an arm may curve, pick any branch, overshoot -- it just cannot leave the road,
# so intent stays a fuzzy "which way" and the planner still resolves the path late.
#
# term 6 (route_pad_conf_loss_weight) drives PADDING arms' confidence to 0. B2-final left them
# at 0.989 -- higher than the real arms' 0.414 -- because loss_route_conf masks padding out of
# its own mean. Harmless in open loop (eval masks by valid) but a closed-loop hazard.
#
# NB closed-loop: 3-cam drivable features -> the Qwen service must extract the FULL strip
# with the 3-cam drivable prompt (extract_vlm_p6_3cam.py), not the front crop.
# NB pred_route is still arm 0 unconditionally (planning_decoder.py:189) and arm 0 is not
# necessarily the expert-supervised winner. Independent of this run; must be fixed before
# closed loop (B2_v2's arm-0 ADE was 6.09 m, so the assumption can fail catastrophically).
#
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_p5_stepB2_corridor_ddp.sh

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

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/p5_stepB2_corridor \
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
use_route_corridor_loss=true \
route_corridor_loss_weight=1.0 \
route_corridor_reach_m=20.0 \
route_pad_conf_loss_weight=1.0 \
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
