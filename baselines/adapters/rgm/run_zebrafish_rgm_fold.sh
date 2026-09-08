#!/usr/bin/env bash
set -euo pipefail

FOLD="$1"
GPU="$2"

PY=/home/ubuntu/anaconda3/envs/nuclr310/bin/python
REPO=/home/ubuntu/klb/nuclr/RGM_official
DATA=/home/ubuntu/klb/nuclr/nuclr/Data/Zebrafish_MPRT_LOFO8_60m/fold_${FOLD}
OUT=/home/ubuntu/klb/nuclr/nuclr/runs/zebrafish_rgm_lofo8_seed42/fold_${FOLD}
LOG=/home/ubuntu/klb/nuclr/nuclr/runs/zebrafish_rgm_lofo8_seed42/fold${FOLD}.log

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

mkdir -p "$(dirname "$LOG")"

echo "================================================================================"
echo "RGM ZEBRAFISH FOLD${FOLD} SEED42"
echo "GPU  = ${GPU}"
echo "DATA = ${DATA}"
echo "OUT  = ${OUT}"
echo "================================================================================"

rm -rf "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" \
"$PY" -u "$REPO/run_rgm_zebrafish_lofo_fold.py" \
    --fold "$FOLD" \
    --data-root "$DATA" \
    --run-root "$OUT" \
    --seed 42 \
    --gpu 0 \
    --epochs 200 \
    2>&1 | tee "$LOG"

test -f "$OUT/summary.json"

echo
echo "FOLD ${FOLD} SUCCESS"
