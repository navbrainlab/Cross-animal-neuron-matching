#!/usr/bin/env bash
set -euo pipefail

action="${1:-all}"  # prepare | audit | train | lock | unlock | test | aggregate | all

repo_root="${REPO_ROOT:-/home/ubuntu/klb/nuclr/nuclr}"
python_bin="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/nuclr310/bin/python}"
mprt_root="${MPRT_ROOT:-$repo_root/neurid}"
source_root="${SOURCE_ROOT:-$repo_root/Data/Zebrafish_LOFO8_joint_from_scratch}"
data_root="${DATA_ROOT:-$repo_root/Data/Zebrafish_MPRT_LOFO8_60m}"
run_root="${RUN_ROOT:-$repo_root/runs/mprt_v1_1/zebrafish_lofo8_seed42}"
legacy_fold1_full="${LEGACY_FOLD1_FULL:-$repo_root/runs/mprt_v1_1/zebrafish_lofo/fold1/seed42/full}"
lock_manifest="$run_root/LOCKED_BEFORE_TEST.json"
epochs="${EPOCHS:-80}"
device="${DEVICE:-cuda}"

read -r -a fold_array <<< "${FOLDS:-1 2 3 4 5 6 7 8}"
read -r -a seed_array <<< "${SEEDS:-42}"
read -r -a variant_array <<< "${VARIANTS:-full no_transport geometry_only activity_only}"
read -r -a gpu_array <<< "${GPUS:-0 1}"

script_root="$repo_root/scripts/zebrafish"
prepare_script="$script_root/prepare_zebrafish_mprt_lofo8_seed42.py"
lock_script="$script_root/lock_zebrafish_mprt_seed42_pretest.py"
euclidean_script="$script_root/evaluate_zebrafish_euclidean.py"
permutation_script="$script_root/audit_zebrafish_mprt_permutation.py"
aggregate_script="$script_root/aggregate_zebrafish_mprt_lofo8_seed42.py"

die() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }

for script in "$prepare_script" "$lock_script" "$euclidean_script" \
              "$permutation_script" "$aggregate_script"; do
  require_file "$script"
done
require_dir "$mprt_root/mprt_net"
require_dir "$source_root"
(( ${#gpu_array[@]} > 0 )) || die "GPUS cannot be empty"
[[ "$device" == "cuda" || "$device" == "cpu" ]] || die "DEVICE must be cuda or cpu"

run_directory() {
  local fold="$1" seed="$2" variant="$3"
  if [[ "$fold" == "1" && "$seed" == "42" && "$variant" == "full" ]]; then
    echo "$legacy_fold1_full"
  else
    echo "$run_root/fold_${fold}/seed${seed}/${variant}"
  fi
}

complete_run() {
  local directory="$1"
  [[ -f "$directory/best.pt" && -f "$directory/history.jsonl" ]] || return 1
  "$python_bin" - "$directory/history.jsonl" "$epochs" <<'PY' >/dev/null
import json
import sys
from pathlib import Path
rows = [x for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
if not rows or int(json.loads(rows[-1])["epoch"]) < int(sys.argv[2]):
    raise SystemExit(1)
PY
}

prepare_stage() {
  cd "$repo_root"
  "$python_bin" "$prepare_script" prepare \
    --source-root "$source_root" \
    --output-root "$data_root" \
    --folds "1-8" \
    --min-shared 20
}

audit_stage() {
  cd "$repo_root"
  "$python_bin" "$prepare_script" audit \
    --source-root "$source_root" \
    --output-root "$data_root" \
    --folds "1-8" \
    --min-shared 20

  cd "$mprt_root"
  for fold in "${fold_array[@]}"; do
    for split in train val; do
      CUDA_VISIBLE_DEVICES="${gpu_array[0]}" "$python_bin" -m mprt_net.self_check \
        --dataset-root "$data_root/fold_${fold}" \
        --split "$split" \
        --device "$device"
    done
  done
}

train_one() {
  local fold="$1" seed="$2" variant="$3" gpu="$4"
  local directory
  directory="$(run_directory "$fold" "$seed" "$variant")"
  if complete_run "$directory"; then
    echo "[SKIP complete] fold=$fold seed=$seed variant=$variant path=$directory"
    return 0
  fi
  if [[ -e "$directory/history.jsonl" || -e "$directory/best.pt" || -e "$directory/last.pt" ]]; then
    echo "Incomplete run exists; preserving it and stopping: $directory" >&2
    return 2
  fi
  mkdir -p "$directory"
  echo "[TRAIN] gpu=$gpu fold=$fold seed=$seed variant=$variant"
  cd "$mprt_root"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m mprt_net.train \
    --dataset-root "$data_root/fold_${fold}" \
    --output-dir "$directory" \
    --variant "$variant" \
    --seed "$seed" \
    --epochs "$epochs" \
    --pairs-per-epoch 128 \
    --activity-length 128 \
    --min-shared 20 \
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
    --device "$device" \
    2>&1 | tee "$directory/train.log"
}

train_stage() {
  local jobs=()
  for fold in "${fold_array[@]}"; do
    for seed in "${seed_array[@]}"; do
      for variant in "${variant_array[@]}"; do
        jobs+=("$fold $seed $variant")
      done
    done
  done

  local worker_pids=()
  local workers="${#gpu_array[@]}"
  for ((worker=0; worker<workers; worker++)); do
    (
      for ((index=worker; index<${#jobs[@]}; index+=workers)); do
        read -r fold seed variant <<< "${jobs[$index]}"
        train_one "$fold" "$seed" "$variant" "${gpu_array[$worker]}"
      done
    ) &
    worker_pids+=("$!")
  done

  local failed=0
  for pid in "${worker_pids[@]}"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 )) || die "one or more training workers failed"
}

lock_stage() {
  mkdir -p "$run_root"
  cd "$repo_root"
  "$python_bin" "$lock_script" \
    --run-root "$run_root" \
    --legacy-fold1-full "$legacy_fold1_full" \
    --data-root "$data_root" \
    --output "$lock_manifest" \
    --epochs "$epochs"
}

unlock_stage() {
  require_file "$lock_manifest"
  cd "$repo_root"
  "$python_bin" "$prepare_script" unlock-test \
    --source-root "$source_root" \
    --output-root "$data_root" \
    --folds "1-8" \
    --min-shared 20 \
    --unlock-manifest "$lock_manifest"
}

test_one() {
  local fold="$1" seed="$2" variant="$3" gpu="$4"
  local directory checkpoint
  directory="$(run_directory "$fold" "$seed" "$variant")"
  checkpoint="$directory/best.pt"
  require_file "$checkpoint"
  if [[ -f "$directory/test_metrics.json" ]]; then
    echo "[SKIP tested] fold=$fold seed=$seed variant=$variant"
    return 0
  fi
  mkdir -p "$directory"
  echo "[TEST] gpu=$gpu fold=$fold seed=$seed variant=$variant"
  cd "$mprt_root"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m mprt_net.evaluate \
    --dataset-root "$data_root/fold_${fold}" \
    --split test \
    --checkpoint "$checkpoint" \
    --activity-length 128 \
    --min-shared 20 \
    --device "$device" \
    --output "$directory/test_metrics.json" \
    --pair-output "$directory/test_pair_metrics.jsonl" \
    2>&1 | tee "$directory/test_eval.log"
}

test_stage() {
  require_file "$lock_manifest"
  for fold in {1..8}; do
    require_dir "$data_root/fold_${fold}/test"
  done

  local jobs=()
  for fold in "${fold_array[@]}"; do
    for seed in "${seed_array[@]}"; do
      for variant in "${variant_array[@]}"; do
        jobs+=("$fold $seed $variant")
      done
    done
  done
  local worker_pids=()
  local workers="${#gpu_array[@]}"
  for ((worker=0; worker<workers; worker++)); do
    (
      for ((index=worker; index<${#jobs[@]}; index+=workers)); do
        read -r fold seed variant <<< "${jobs[$index]}"
        test_one "$fold" "$seed" "$variant" "${gpu_array[$worker]}"
      done
    ) &
    worker_pids+=("$!")
  done
  local failed=0
  for pid in "${worker_pids[@]}"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 )) || die "one or more test workers failed"

  cd "$mprt_root"
  for fold in {1..8}; do
    baseline_dir="$run_root/baselines/fold_${fold}"
    mkdir -p "$baseline_dir"
    if [[ ! -f "$baseline_dir/euclidean.json" ]]; then
      "$python_bin" "$euclidean_script" \
        --dataset-root "$data_root/fold_${fold}" \
        --split test \
        --activity-length 128 \
        --min-shared 20 \
        --output "$baseline_dir/euclidean.json" \
        --pair-output "$baseline_dir/euclidean_pairs.jsonl"
    fi

    audit_dir="$run_root/audits/fold_${fold}"
    mkdir -p "$audit_dir"
    full_seed42="$(run_directory "$fold" 42 full)/best.pt"
    if [[ ! -f "$audit_dir/permutation_full_seed42.json" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu_array[0]}" "$python_bin" "$permutation_script" \
        --dataset-root "$data_root/fold_${fold}" \
        --checkpoint "$full_seed42" \
        --split test \
        --activity-length 128 \
        --min-shared 20 \
        --device "$device" \
        --output "$audit_dir/permutation_full_seed42.json"
    fi
  done
}

aggregate_stage() {
  cd "$repo_root"
  "$python_bin" "$aggregate_script" \
    --run-root "$run_root" \
    --legacy-fold1-full "$legacy_fold1_full" \
    --data-root "$data_root" \
    --bootstrap 20000 \
    --bootstrap-seed 20260825
}

case "$action" in
  prepare) prepare_stage ;;
  audit) audit_stage ;;
  train) train_stage ;;
  lock) lock_stage ;;
  unlock) unlock_stage ;;
  test) test_stage ;;
  aggregate) aggregate_stage ;;
  all)
    prepare_stage
    audit_stage
    train_stage
    lock_stage
    unlock_stage
    test_stage
    aggregate_stage
    ;;
  *)
    echo "Usage: bash run_zebrafish_mprt_lofo8_seed42.sh {prepare|audit|train|lock|unlock|test|aggregate|all}" >&2
    exit 2
    ;;
esac
