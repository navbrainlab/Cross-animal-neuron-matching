#!/usr/bin/env bash
cd /home/ubuntu/klb/nuclr/nuclr

PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/nuclr310/bin/python}"

mkdir -p benchmark_official/logs

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

"$PYTHON" -u benchmark_official/adapters/stat_atlas/evaluate_statatlas_official.py \
  --dataset rld \
  --fold 0 \
  2>&1 | tee benchmark_official/logs/stat_atlas_official_rld_cv5.log
