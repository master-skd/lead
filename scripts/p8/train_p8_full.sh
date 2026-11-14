#!/usr/bin/env bash
# LEAD posttrain recipe with P8 K routes + K target speeds: pretrain init, all frames, 31 epochs,
# lr 3e-4, unfrozen backbone, route smoothing 20 and sensor perturbation (perturbated Qwen tokens).
set -euo pipefail
cd "$(dirname "$0")/../.."
export P8_POSTTRAIN=${P8_POSTTRAIN:-outputs/local_training/pretrain/model_0030.pth}
export P8_HIDDEN_CACHE=${P8_HIDDEN_CACHE:-data/p8/hidden_cache_full}
export P8_HIDDEN_CACHE_PERTURBATED=${P8_HIDDEN_CACHE_PERTURBATED:-data/p8/hidden_cache_full_perturbated}
export P8_SENSOR_PERTURBATION=true
export P8_ROUTE_SMOOTHING=${P8_ROUTE_SMOOTHING:-20}
export P8_EPOCHS=${P8_EPOCHS:-31}
export P8_LOGDIR=${P8_LOGDIR:-outputs/local_training/p8_hidden_full_posttrain}
export P8_HIDDEN_PREFETCH=${P8_HIDDEN_PREFETCH:-1}
[[ -f "$P8_HIDDEN_CACHE_PERTURBATED/frames.sqlite" ]] || { echo "Perturbated hidden cache incomplete: $P8_HIDDEN_CACHE_PERTURBATED" >&2; exit 1; }
exec bash scripts/p8/train_p8_hidden.sh
