#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-atanas}"
gpu="${2:-0}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

case "$dataset" in
  atanas)
    data_root="$repo_root/Data/Atanas_SF_unified_000776/date_disjoint_v1/full"
    pairs_per_epoch=128
    ;;
  rld)
    data_root="$repo_root/Data/Dunn_001623/date_disjoint_full95_v1"
    pairs_per_epoch=256
    ;;
  *)
    echo "usage: bash scripts/run_seed42.sh {atanas|rld} [gpu]" >&2
    exit 2
    ;;
esac

output_dir="$repo_root/runs/mprt_v1_1/$dataset/seed42/full"
CUDA_VISIBLE_DEVICES="$gpu" python -m mprt_net.train \
  --dataset-root "$data_root" \
  --output-dir "$output_dir" \
  --variant full \
  --seed 42 \
  --epochs 80 \
  --pairs-per-epoch "$pairs_per_epoch"
