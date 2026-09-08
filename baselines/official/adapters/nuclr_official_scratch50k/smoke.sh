#!/usr/bin/env bash
cd /home/ubuntu/klb/nuclr/nuclr || exit 1
PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/nuclr310/bin/python}"
SCRIPT="baselines/official/adapters/nuclr_official_scratch50k/run_one.py"
ROOT="baselines/official/runs/nuclr_official_scratch50k_SMOKE"
LOG="baselines/official/logs/nuclr_official_scratch50k_SMOKE"
mkdir -p "$LOG"

# Tiny engineering smoke only. These outputs are NOT paper results.
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -u "$SCRIPT" \
  --dataset atanas --fold 1 --seed 42 --device cuda \
  --target-train-steps 200 --val-every-steps 100 --save-last-every-steps 100 \
  --run-root "$ROOT" \
  2>&1 | tee "$LOG/atanas_fold1_seed42.log"
