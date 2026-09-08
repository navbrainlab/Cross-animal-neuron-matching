#!/usr/bin/env bash
set -euo pipefail

ZM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ZM_DATA_ROOT="${ZM_DATA_ROOT:-/home/ubuntu/klb/zm9624}"
ZM_OUTPUT_ROOT="${ZM_OUTPUT_ROOT:-${ZM_REPO_ROOT}/outputs/zm9624_strict_loso_seed42}"
ZM_PYTHON_BIN="${ZM_PYTHON_BIN:-python}"
ZM_DEVICE="${ZM_DEVICE:-cuda}"
ZM_MODEL_ROOT="${ZM_REPO_ROOT}/mprt_net_v1_1"
ZM_SCRIPT_ROOT="${ZM_REPO_ROOT}/scripts/zm9624"

if ! command -v "${ZM_PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found. Activate the project environment first." >&2
  exit 1
fi
if [[ ! -d "${ZM_DATA_ROOT}/0214-02" || ! -d "${ZM_DATA_ROOT}/0321-01" ]]; then
  echo "Expected 0214-02 and 0321-01 under: ${ZM_DATA_ROOT}" >&2
  exit 1
fi
if [[ -e "${ZM_OUTPUT_ROOT}" ]]; then
  echo "Output already exists; choose a new ZM_OUTPUT_ROOT: ${ZM_OUTPUT_ROOT}" >&2
  exit 1
fi
if [[ "${ZM_DEVICE}" != "cuda" && "${ZM_DEVICE}" != "cpu" ]]; then
  echo "ZM_DEVICE must be cuda or cpu" >&2
  exit 1
fi

mkdir -p "${ZM_OUTPUT_ROOT}"
export PYTHONPATH="${ZM_MODEL_ROOT}:${ZM_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

prepare_fold() {
  local train_worm="$1"
  local fold_dir="$2"
  "${ZM_PYTHON_BIN}" "${ZM_SCRIPT_ROOT}/prepare_zm9624_position_loso.py" \
    --train-worm-dir "${ZM_DATA_ROOT}/${train_worm}" \
    --output-root "${fold_dir}/data" \
    --window-length 500 \
    --train-windows 9 \
    --val-windows 3 \
    --use-activity \
    --activity-normalization per_trace_zscore \
    --train-only
}

train_fold() {
  local fold_dir="$1"
  "${ZM_PYTHON_BIN}" -m mprt_net.train_transfer \
    --dataset-root "${fold_dir}/data" \
    --output-dir "${fold_dir}/scratch_full_seed42_e40" \
    --variant full \
    --seed 42 \
    --epochs 40 \
    --pairs-per-epoch 16 \
    --val-max-pairs 0 \
    --activity-length 256 \
    --min-shared 2 \
    --synthetic-drop-probability 0.05 \
    --focal-gamma 2.0 \
    --learning-rate 0.0002 \
    --weight-decay 0.0001 \
    --gradient-clip 1.0 \
    --cycle-weight 0.0 \
    --atlas-weight 0.0 \
    --atlas-blend-weight 0.0 \
    --early-stopping-patience 0 \
    --hidden-dim 96 \
    --edge-dim 48 \
    --relation-dim 8 \
    --activity-channels 32 \
    --num-heads 4 \
    --population-layers 2 \
    --dropout 0.1 \
    --sinkhorn-iterations 20 \
    --transport-steps 2 \
    --structural-weight 1.0 \
    --hard-knn-k 16 \
    --device "${ZM_DEVICE}"
}

evaluate_fold() {
  local held_out_worm="$1"
  local fold_dir="$2"
  local result_name="$3"
  "${ZM_PYTHON_BIN}" "${ZM_SCRIPT_ROOT}/match_zm9624_position_window_ensemble.py" \
    --worm-dir "${ZM_DATA_ROOT}/${held_out_worm}" \
    --checkpoint "${fold_dir}/scratch_full_seed42_e40/best.pt" \
    --output-dir "${fold_dir}/${result_name}" \
    --window-length 500 \
    --gap-windows 1 \
    --activity-normalization per_trace_zscore \
    --long-range-weight 0 \
    --device "${ZM_DEVICE}"
}

ZM_FOLD_0321="${ZM_OUTPUT_ROOT}/train_0214_test_0321"
ZM_FOLD_0214="${ZM_OUTPUT_ROOT}/train_0321_test_0214"

# Each training fold is prepared without reading the held-out worm.
prepare_fold "0214-02" "${ZM_FOLD_0321}"
prepare_fold "0321-01" "${ZM_FOLD_0214}"

"${ZM_PYTHON_BIN}" -m mprt_net.self_check --dataset-root "${ZM_FOLD_0321}/data" --device "${ZM_DEVICE}"
"${ZM_PYTHON_BIN}" -m mprt_net.self_check --dataset-root "${ZM_FOLD_0214}/data" --device "${ZM_DEVICE}"

# No initialization checkpoint: both directions train from scratch.
train_fold "${ZM_FOLD_0321}"
train_fold "${ZM_FOLD_0214}"

# Held-out worms are read only after both training runs have finished.
evaluate_fold "0321-01" "${ZM_FOLD_0321}" "heldout_0321_locked"
evaluate_fold "0214-02" "${ZM_FOLD_0214}" "heldout_0214_locked"

echo "Finished. Results: ${ZM_OUTPUT_ROOT}"
