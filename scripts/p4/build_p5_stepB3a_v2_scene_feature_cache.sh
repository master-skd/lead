#!/usr/bin/env bash
# Build B3a-v2 scene tokens, full trajectory states and five oracle labels.
set -euo pipefail

REPO_ROOT="/mmu_mllm_hdd_3/liuzihan08/vla/lead"
export B3A_VELOCITY_FEATURE_ROOT="${B3A_V2_FEATURE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_scene_scorer/features}"
export B3A_NEAREST_VLM_MANIFEST="${B3A_NEAREST_VLM_MANIFEST:-${REPO_ROOT}/data/p4/manifest.jsonl}"
export B3A_FULL_PROFILE_REACHABILITY=1
export B3A_SCENE_V2=1
export B3A_EXTRACT_BATCH_SIZE="${B3A_EXTRACT_BATCH_SIZE:-64}"
export B3A_EXTRACT_NUM_WORKERS="${B3A_EXTRACT_NUM_WORKERS:-4}"
if [[ "${B3A_V2_DENSE:-0}" == "1" ]]; then
    DENSE_ROOT="${B3A_V2_DENSE_ROOT:-${REPO_ROOT}/outputs/local_training/p5_stepB3a_v2_dense_data}"
    export B3A_SPLIT_DIR="${B3A_SPLIT_DIR:-${DENSE_ROOT}/route_split}"
    export B3A_ACTOR_ROOT="${B3A_ACTOR_ROOT:-${DENSE_ROOT}/future_actor_cache}"
    export B3A_VELOCITY_VOCAB_DIR="${B3A_VELOCITY_VOCAB_DIR:-${DENSE_ROOT}/velocity_vocab}"
    export B3A_RELATIVE_VOCAB_DIR="${B3A_RELATIVE_VOCAB_DIR:-${DENSE_ROOT}/relative_velocity_vocab}"
fi

cd "${REPO_ROOT}"
bash scripts/p4/build_p5_stepB3a_velocity_feature_cache.sh "$@"
