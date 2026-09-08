#!/usr/bin/env bash
cd /home/ubuntu/klb/nuclr/nuclr
PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/nuclr310/bin/python}"

mkdir -p baselines/official/logs

"$PYTHON" -u \
  baselines/official/adapters/euclidean/evaluate_euclidean_locked.py \
  2>&1 | tee baselines/official/logs/euclidean_locked.log
