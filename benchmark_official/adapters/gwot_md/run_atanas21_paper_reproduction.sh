#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/nuclr310/bin/python}"
paper_dir="$root/gwot_md_atanas21_paper"
data_dir="${DATA_DIR:-$paper_dir/data_paper_snapshot_v1}"
labels="${LABELS:-$paper_dir/neuropal_label_v4.json.bz2}"
h5_dir="${H5_DIR:-$paper_dir/official_h5}"
output_template="${OUTPUT_TEMPLATE:-$paper_dir/atanas21_gwot_md_paper_h{h}}"
top5_dir="${TOP5_DIR:-$paper_dir/atanas21_paper_top5}"

action="${1:-}"
case "$action" in
  prepare)
    "$python_bin" "$root/benchmark_official/adapters/gwot_md/prepare_atanas21_paper_inputs.py" \
      --h5-dir "$h5_dir" \
      --labels "$labels" \
      --old-npz-root "$root/Data/Atanas_SF_unified_000776/full" \
      --output-dir "$data_dir" \
      --verify-h5-sha256
    ;;
  solve)
    "$python_bin" "$paper_dir/solve_gwot_md_atanas21.py" solve \
      --data-dir "$data_dir" \
      --device "${DEVICE:-cuda}" \
      --init-batch-size "${INIT_BATCH_SIZE:-50}" \
      --num-shards "${NUM_SHARDS:-64}" \
      --shard-index "${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}" \
      --output-template "$output_template"
    ;;
  audit)
    "$python_bin" "$paper_dir/solve_gwot_md_atanas21.py" audit \
      --output-template "$output_template"
    ;;
  evaluate)
    "$python_bin" "$paper_dir/evaluate_paper_top5_strict.py" \
      --run-template "$output_template" \
      --h-values 0,5,10,15,20,25,30,35,40,45,50 \
      --teacher-count 9 \
      --num-splits 1000 \
      --selection-v 5 \
      --selection-k 5 \
      --v-values 5 \
      --k-values 5 \
      --seed 42 \
      --tie-break first \
      --output-dir "$top5_dir"
    ;;
  self-check)
    "$python_bin" "$paper_dir/solve_gwot_md_atanas21.py" self-check
    ;;
  official-h0-audit)
    "$python_bin" "$root/benchmark_official/adapters/gwot_md/audit_gwtune_h0_equivalence.py" \
      --data-dir "$data_dir" \
      --output "$paper_dir/GWTUNE_H0_EQUIVALENCE.json"
    ;;
  *)
    echo "usage: $0 {prepare|solve|audit|evaluate|self-check|official-h0-audit}" >&2
    exit 2
    ;;
esac
