#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
data_root="$repo_root/Data/Atanas_SF_unified_000776/date_disjoint_v1/full"
run_root="$repo_root/runs/mprt_v1_1/atanas/seed42"

run_variant() {
  local variant="$1"
  local gpu="$2"
  local output_dir="$run_root/$variant"
  if [[ -e "$output_dir/history.jsonl" || -e "$output_dir/best.pt" || -e "$output_dir/last.pt" ]]; then
    echo "Refusing to overwrite existing run: $output_dir" >&2
    return 1
  fi
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" python -m mprt_net.train \
    --dataset-root "$data_root" \
    --output-dir "$output_dir" \
    --variant "$variant" \
    --seed 42 \
    --epochs 80 \
    --pairs-per-epoch 128 \
    --activity-length 512 \
    --min-shared 2 \
    --synthetic-drop-probability 0.05 \
    --focal-gamma 2.0 \
    --learning-rate 2e-4 \
    --weight-decay 1e-4 \
    --gradient-clip 1.0 \
    --hidden-dim 96 \
    --edge-dim 48 \
    --relation-dim 8 \
    --activity-channels 32 \
    --num-heads 4 \
    --population-layers 2 \
    --dropout 0.10 \
    --sinkhorn-iterations 20 \
    --transport-steps 2 \
    --structural-weight 1.0 \
    --device cuda \
    >"$output_dir/train.log" 2>&1
  tail -n 1 "$output_dir/train.log"
}

(
  run_variant full 0
  run_variant no_population 0
) &
pid_gpu0=$!

(
  run_variant no_transport 1
  run_variant node_only 1
) &
pid_gpu1=$!

wait "$pid_gpu0"
wait "$pid_gpu1"

echo "All four Atanas seed-42 runs completed under $run_root"
