#!/usr/bin/env bash
cd /home/ubuntu/klb/nuclr/nuclr
PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/nuclr310/bin/python}"

mkdir -p benchmark_official/logs

"$PYTHON" -u \
  benchmark_official/adapters/euclidean/evaluate_euclidean_locked.py \
  2>&1 | tee benchmark_official/logs/euclidean_locked.log
