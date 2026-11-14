#!/usr/bin/env bash
# Package the CARLA sensors needed by P6 online joint training in parallel,
# independently transferable scenario archives. Safe to rerun after interruption.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${P6_PACK_OUT:-"$ROOT/outputs/transfer/p6_joint_train_assets"}
JOBS=${P6_PACK_JOBS:-8}
SCENARIO=

while (($#)); do
    case "$1" in
        --out) OUT=$2; shift 2 ;;
        --jobs) JOBS=$2; shift 2 ;;
        --scenario) SCENARIO=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$ROOT"

package_archive() {
    local name=$1
    shift
    local archive="$OUT/$name.tar.gz"
    local checksum="$OUT/$name.tar.gz.sha256"
    local temp

    if [[ -e "$archive" ]]; then
        if [[ ! -f "$checksum" ]]; then
            gzip -t "$archive"
            (cd "$OUT" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
        fi
        (cd "$OUT" && sha256sum --status -c "$name.tar.gz.sha256") || {
            echo "Existing archive failed checksum: $archive" >&2
            exit 1
        }
        echo "skip verified: $archive"
        return
    fi

    temp=$(mktemp "$OUT/.$name.XXXXXXXX.part")
    trap 'rm -f -- "$temp"' EXIT
    tar -I 'gzip -1' -cf "$temp" -C "$ROOT" "$@"
    gzip -t "$temp"
    mv -- "$temp" "$archive"
    trap - EXIT
    (cd "$OUT" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
    echo "ready: $archive ($(du -h "$archive" | cut -f1))"
}

if [[ -n "$SCENARIO" ]]; then
    [[ "$SCENARIO" != *'/'* && "$SCENARIO" != '.' && "$SCENARIO" != '..' ]] || {
        echo "Invalid scenario: $SCENARIO" >&2
        exit 2
    }
    [[ -d "data/carla_leaderboard2/data/$SCENARIO" ]] || {
        echo "Missing scenario: $SCENARIO" >&2
        exit 1
    }
    package_archive "scenario_$SCENARIO" \
        --exclude='*/rgb' --exclude='*/rgb/*' \
        --exclude='*_perturbated' --exclude='*_perturbated/*' \
        "data/carla_leaderboard2/data/$SCENARIO"
    exit
fi

package_archive static \
    data/carla_leaderboard2/buckets \
    outputs/local_training/pretrain/config.json \
    outputs/local_training/pretrain/model_0030.pth \
    outputs/local_training/vlm_intent_p6_online_full_b32_w4/model_best.pth

echo "Packaging 43 scenario archives with $JOBS parallel workers into $OUT"
find data/carla_leaderboard2/data -mindepth 1 -maxdepth 1 -type d -printf '%f\0' |
    xargs -0 -r -P "$JOBS" -I '{}' bash "$ROOT/scripts/p4/package_p6_joint_train_assets.sh" \
        --out "$OUT" --jobs "$JOBS" --scenario '{}'
echo "All scenario archives are ready: $OUT"
