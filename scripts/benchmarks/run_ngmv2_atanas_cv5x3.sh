#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ubuntu/klb/nuclr/ThinkMatch_official
NUCLR=/home/ubuntu/klb/nuclr/nuclr
DATA=$NUCLR/Data/Atanas_SF_unified_000776/cv5_grouped_v1
ROOT=$NUCLR/runs/unified_benchmark/ngmv2/atanas

run_one () {
    FOLD="$1"
    SEED="$2"
    GPU="$3"

    OUT=$ROOT/fold${FOLD}/seed${SEED}
    mkdir -p "$OUT"

    echo "================================================================"
    echo "[START] fold=$FOLD seed=$SEED physical_gpu=$GPU"
    echo "================================================================"

    cd "$REPO"

    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="$REPO:$NUCLR" \
    python -u "$NUCLR/scripts/benchmarks/run_ngmv2_atanas_fold.py" \
      --data-root "$DATA" \
      --fold "$FOLD" \
      --seed "$SEED" \
      --device cuda:0 \
      --feature-dim 64 \
      --epochs 20 \
      --pairs-per-epoch 500 \
      --min-common 20 \
      --lr 2e-3 \
      --skip-test \
      --out "$OUT" \
      2>&1 | tee "$OUT/train_val.log"

    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="$REPO:$NUCLR" \
    python -u "$NUCLR/scripts/benchmarks/eval_ngmv2_atanas_locked.py" \
      --data-root "$DATA" \
      --fold "$FOLD" \
      --seed "$SEED" \
      --device cuda:0 \
      --run-dir "$OUT" \
      2>&1 | tee "$OUT/locked_test.log"

    echo "[DONE] fold=$FOLD seed=$SEED"
}

# fold0 seed42 already completed.
worker0 () {
    run_one 0   1   0
    run_one 1  42   0
    run_one 1 123   0
    run_one 2   1   0
    run_one 2  42   0
    run_one 3 123   0
    run_one 4   1   0
}

worker1 () {
    run_one 0 123   1
    run_one 1   1   1
    run_one 2 123   1
    run_one 3   1   1
    run_one 3  42   1
    run_one 4  42   1
    run_one 4 123   1
}

worker0 > "$ROOT/worker_gpu0.log" 2>&1 &
PID0=$!

worker1 > "$ROOT/worker_gpu1.log" 2>&1 &
PID1=$!

echo "GPU0 worker PID=$PID0"
echo "GPU1 worker PID=$PID1"

wait "$PID0"
wait "$PID1"

echo "ALL NGM-v2 CV5x3 CELLS COMPLETE"
