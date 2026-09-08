#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/klb/nuclr/nuclr
DATA=$ROOT/Data/Atanas_SF_unified_000776/cv5_grouped_v1
OUTROOT=$ROOT/runs/unified_benchmark/fgw_pot/atanas

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

run_fold () {
    FOLD="$1"
    OUT="$OUTROOT/fold${FOLD}"

    mkdir -p "$OUT"

    echo "============================================================"
    echo "[FGW START] fold=$FOLD"
    echo "============================================================"

    python -u "$ROOT/scripts/benchmarks/run_official_fgw_atanas.py" fit-lock \
      --data-root "$DATA" \
      --fold "$FOLD" \
      --alphas 0.25 0.50 0.75 \
      --out "$OUT" \
      2>&1 | tee "$OUT/fit_val.log"

    python -u "$ROOT/scripts/benchmarks/run_official_fgw_atanas.py" locked-test \
      --data-root "$DATA" \
      --fold "$FOLD" \
      --out "$OUT" \
      2>&1 | tee "$OUT/locked_test.log"

    echo "[FGW DONE] fold=$FOLD"
}

# fold0 already completed correctly.
run_fold 1 > "$OUTROOT/fold1_worker.log" 2>&1 &
P1=$!

run_fold 2 > "$OUTROOT/fold2_worker.log" 2>&1 &
P2=$!

run_fold 3 > "$OUTROOT/fold3_worker.log" 2>&1 &
P3=$!

run_fold 4 > "$OUTROOT/fold4_worker.log" 2>&1 &
P4=$!

echo "fold1 PID=$P1"
echo "fold2 PID=$P2"
echo "fold3 PID=$P3"
echo "fold4 PID=$P4"

wait "$P1"
wait "$P2"
wait "$P3"
wait "$P4"

echo "ALL FGW ATANAS CV5 FOLDS COMPLETE"
