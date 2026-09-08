#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
data_root="$repo_root/Data/Atanas_SF_unified_000776/date_disjoint_v1/full"
run_root="$repo_root/runs/mprt_v1_1/atanas/seed42"
comparison_root="$run_root/comparisons"
mkdir -p "$comparison_root"

for variant in no_transport no_population node_only; do
  python -m mprt_net.compare \
    --dataset-root "$data_root" \
    --split val \
    --checkpoint-a "$run_root/full/best.pt" \
    --checkpoint-b "$run_root/$variant/best.pt" \
    --name-a full \
    --name-b "$variant" \
    --bootstrap-iterations 10000 \
    --bootstrap-seed 20260823 \
    --device cuda \
    --output "$comparison_root/full_vs_${variant}.json" \
    --query-output "$comparison_root/full_vs_${variant}_queries.csv"
done

echo "Comparisons saved under $comparison_root"
