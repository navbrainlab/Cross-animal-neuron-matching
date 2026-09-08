#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/klb/nuclr/nuclr || exit 1

PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/nuclr310/bin/python}"
SCRIPT="baselines/official/adapters/nuclr_official_scratch50k/run_one.py"
AUDIT="baselines/official/adapters/nuclr_official_scratch50k/audit_plan.py"
SUMMARY="baselines/official/adapters/nuclr_official_scratch50k/summarize.py"
LOGDIR="baselines/official/logs/nuclr_official_scratch50k_cv5x3"
RUNROOT="baselines/official/runs/nuclr_official_scratch50k_cv5x3"
mkdir -p "$LOGDIR"

"$PYTHON" "$AUDIT" 2>&1 | tee "$LOGDIR/audit_plan.log" || exit 2

run_job () {
  ds="$1"
  fold="$2"
  seed="$3"
  gpu="$4"
  result="${RUNROOT}/${ds}/fold_${fold}/seed_${seed}/outer_test_medoid_template_v1/result.json"
  log="${LOGDIR}/${ds}_fold${fold}_seed${seed}.log"

  if [ -f "$result" ]; then
    echo "[REUSE] dataset=${ds} fold=${fold} seed=${seed}"
    return 0
  fi

  echo "[START] dataset=${ds} fold=${fold} seed=${seed} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -u "$SCRIPT" \
    --dataset "$ds" --fold "$fold" --seed "$seed" --device cuda \
    --target-train-steps 50000 \
    --val-every-steps 1000 \
    --save-last-every-steps 1000 \
    --run-root "$RUNROOT" \
    > "$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[DONE] dataset=${ds} fold=${fold} seed=${seed}"
  else
    echo "[FAILED] dataset=${ds} fold=${fold} seed=${seed} rc=${rc} log=${log}"
  fi
  return "$rc"
}

JOBS=()
for ds in atanas rld; do
  for fold in 1 2 3 4 5; do
    for seed in 1 42 123; do
      JOBS+=("$ds $fold $seed")
    done
  done
done

worker () {
  gpu="$1"
  parity="$2"
  idx=0
  failures=0
  for spec in "${JOBS[@]}"; do
    if [ $((idx % 2)) -eq "$parity" ]; then
      read -r ds fold seed <<< "$spec"
      run_job "$ds" "$fold" "$seed" "$gpu" || failures=$((failures + 1))
    fi
    idx=$((idx + 1))
  done
  echo "[WORKER DONE] gpu=${gpu} failures=${failures}"
  return "$failures"
}

worker 0 0 & pid0=$!
worker 1 1 & pid1=$!
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?

if [ "$rc0" -ne 0 ] || [ "$rc1" -ne 0 ]; then
  echo "One or more jobs failed. The runner is resumable; rerun the same command after inspection."
  exit 3
fi

"$PYTHON" -u "$SUMMARY" 2>&1 | tee "$LOGDIR/summary.log"
