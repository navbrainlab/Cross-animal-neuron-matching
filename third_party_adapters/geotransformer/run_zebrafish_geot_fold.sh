#!/usr/bin/env bash
set -euo pipefail

FOLD="$1"
GPU="$2"

PY=/home/ubuntu/anaconda3/envs/nuclr310/bin/python

REPO=/home/ubuntu/klb/nuclr/geotransformer_official
EXP=$REPO/experiments/geotransformer.zebrafish.semantic
DATA=/home/ubuntu/klb/nuclr/nuclr/Data/Zebrafish_MPRT_LOFO8_60m/fold_${FOLD}

RUN=fold${FOLD}_seed42_k64
OUT=$REPO/output/geotransformer.zebrafish.semantic/$RUN
LOG=$REPO/zebrafish_geot_fold${FOLD}_seed42_k64.log

RESULT=/home/ubuntu/klb/nuclr/nuclr/runs/zebrafish_geotransformer_lofo8_seed42/fold_${FOLD}
mkdir -p "$RESULT"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

case "$FOLD" in
    1) EXPECTED_Q=3536 ;;
    2) EXPECTED_Q=3954 ;;
    3) EXPECTED_Q=768  ;;
    4) EXPECTED_Q=1820 ;;
    5) EXPECTED_Q=1464 ;;
    6) EXPECTED_Q=938  ;;
    7) EXPECTED_Q=2492 ;;
    8) EXPECTED_Q=2184 ;;
    *) echo "Invalid fold: $FOLD"; exit 1 ;;
esac

EXPECTED_PAIRS=$(
"$PY" - "$DATA/test" <<'PY'
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
pair_ids = set()

for p in root.rglob("*.npz"):
    with np.load(p, allow_pickle=True) as z:
        x = np.asarray(z["pair_id"])
        if x.ndim == 0:
            x = x.item()
        else:
            x = x.reshape(-1)[0]
        if isinstance(x, bytes):
            x = x.decode()
        pair_ids.add(str(x))

print(len(pair_ids))
PY
)

echo "================================================================================"
echo "GEOTRANSFORMER ZEBRAFISH FOLD $FOLD"
echo "GPU            = $GPU"
echo "expected pairs = $EXPECTED_PAIRS"
echo "expected Q     = $EXPECTED_Q"
echo "================================================================================"

cd "$EXP"

# Formal run from scratch.
rm -rf "$OUT"

SEED=42 \
RUN_TAG="$RUN" \
ZEBRAFISH_ROOT="$DATA" \
MIN_SHARED=20 \
GT_PATCH_K=32 \
GT_COARSE_K=64 \
GT_MAX_ITERS=20000 \
GT_SNAPSHOT_STEPS=500 \
GT_WARMUP=200 \
CUDA_VISIBLE_DEVICES="$GPU" \
"$PY" -u trainval.py \
2>&1 | tee "$LOG"

# ---------------------------------------------------------------
# Validation-only checkpoint selection.
# highest Val Top1; exact tie -> earliest iteration.
# ---------------------------------------------------------------
BEST_ITER=$(
"$PY" - "$LOG" <<'PY'
import re
import sys
from pathlib import Path

p = Path(sys.argv[1])

pat = re.compile(
    r'\[Val\]\s+Iter:\s*(\d+).*?'
    r'Top1:\s*([0-9.]+)'
)

rows = []

for line in p.read_text(errors="ignore").splitlines():
    m = pat.search(line)
    if m:
        rows.append(
            (int(m.group(1)), float(m.group(2)))
        )

if not rows:
    raise RuntimeError("No validation entries found")

best = sorted(rows, key=lambda x: (-x[1], x[0]))[0]

print(best[0])
PY
)

BEST_CKPT="$OUT/snapshots/iter-${BEST_ITER}.pth.tar"

if [ ! -f "$BEST_CKPT" ]; then
    echo "ERROR: selected checkpoint missing: $BEST_CKPT"
    exit 1
fi

echo
echo "BEST_ITER=$BEST_ITER"
echo "BEST_CKPT=$BEST_CKPT"

# ---------------------------------------------------------------
# LOCKED TEST
# ---------------------------------------------------------------
SEED=42 \
ZEBRAFISH_ROOT="$DATA" \
MIN_SHARED=20 \
GT_PATCH_K=32 \
GT_COARSE_K=64 \
CUDA_VISIBLE_DEVICES="$GPU" \
"$PY" -u evaluate_zebrafish_locked.py \
    --checkpoint "$BEST_CKPT" \
    --output "$RESULT/summary.json" \
    --expected-pairs "$EXPECTED_PAIRS" \
    --expected-queries "$EXPECTED_Q" \
    --device cuda \
    2>&1 | tee "$RESULT/test.log"

echo
echo "FOLD $FOLD SUCCESS"
