#!/usr/bin/env bash
set -euo pipefail

action="${1:-all}"  # prepare | audit | train | all

repo_root="${REPO_ROOT:-/home/ubuntu/klb/nuclr/nuclr}"
python_bin="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/nuclr310/bin/python}"
source_root="${SOURCE_ROOT:-$repo_root/Data/Zebrafish_LOFO8_joint_from_scratch/fold_1/features}"
data_root="${DATA_ROOT:-$repo_root/Data/Zebrafish_MPRT_LOFO8_60m/fold_1}"
run_root="${RUN_ROOT:-$repo_root/runs/mprt_v1_1/zebrafish_lofo/fold1/seed42/full}"
gpu="${GPU:-0}"

cd "$repo_root/mprt_net_v1_1"

prepare_stage() {
  "$python_bin" "$repo_root/scripts/zebrafish/prepare_zebrafish_mprt_fold1.py" \
    --source-root "$source_root" \
    --output-root "$data_root" \
    --min-shared 20
}

audit_stage() {
  "$python_bin" "$repo_root/scripts/zebrafish/prepare_zebrafish_mprt_fold1.py" \
    --output-root "$data_root" \
    --min-shared 20 \
    --audit-only

  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m mprt_net.self_check \
    --dataset-root "$data_root" \
    --split train \
    --device cuda

  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m mprt_net.self_check \
    --dataset-root "$data_root" \
    --split val \
    --device cuda
}

train_stage() {
  if [[ -e "$run_root/history.jsonl" || \
        -e "$run_root/best.pt" || \
        -e "$run_root/last.pt" ]]; then
    echo "Refusing to overwrite existing run: $run_root" >&2
    exit 1
  fi

  mkdir -p "$run_root"

  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m mprt_net.train \
    --dataset-root "$data_root" \
    --output-dir "$run_root" \
    --variant full \
    --seed 42 \
    --epochs 80 \
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
    --device cuda \
    2>&1 \
    | tee "$run_root/train.log" \
    | sed -u 's/^/[Zebrafish fold1 seed42 full] /'

  grep 'best_val_top1_real=' "$run_root/train.log" | tail -n 1
}

case "$action" in
  prepare)
    prepare_stage
    ;;
  audit)
    audit_stage
    ;;
  train)
    train_stage
    ;;
  all)
    prepare_stage
    audit_stage
    train_stage
    ;;
  *)
    echo "Usage: bash run_zebrafish_mprt_fold1_seed42.sh [prepare|audit|train|all]" >&2
    exit 2
    ;;
esac
