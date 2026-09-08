#!/usr/bin/env bash
set -uo pipefail

DATASET="$1"
GPU="${2:-0}"

ROOT=/home/ubuntu/klb/nuclr/geotransformer_official

SEEDS=(1 42 123)
FOLDS=(1 2 3 4 5)

if [ "$DATASET" = "atanas" ]; then
    EXP_NAME="geotransformer.atanas.semantic"
    K=96
elif [ "$DATASET" = "rld" ]; then
    EXP_NAME="geotransformer.rld.semantic"
    K=32
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

EXP="$ROOT/experiments/$EXP_NAME"
DATA="$ROOT/data_cv5/$DATASET"
RESULT_ROOT="$ROOT/cv5x3_results/$DATASET"

mkdir -p "$RESULT_ROOT"

cd "$EXP" || exit 1

for FOLD in "${FOLDS[@]}"; do
for SEED in "${SEEDS[@]}"; do

    RUN_TAG="${DATASET}_fold${FOLD}_seed${SEED}"

    RESULT_DIR="$RESULT_ROOT/fold_${FOLD}/seed_${SEED}"
    VAL_DIR="$RESULT_DIR/val_scan"

    mkdir -p "$VAL_DIR"

    export ATANAS_ROOT="$DATA/fold_${FOLD}"
    export SEED="$SEED"
    export GT_COARSE_K="$K"
    export RUN_TAG="$RUN_TAG"

    export GT_MAX_ITERS=20000
    export GT_WARMUP=500
    export GT_SNAPSHOT_STEPS=1000

    CKPT_DIR="$ROOT/output/$EXP_NAME/$RUN_TAG/snapshots"

    echo
    echo "=========================================================="
    echo "$DATASET fold=$FOLD seed=$SEED K=$K"
    echo "=========================================================="

    # ---------------- TRAIN ----------------
    if [ ! -f "$CKPT_DIR/iter-20000.pth.tar" ]; then

        echo "[TRAIN] $RUN_TAG"

        if ! CUDA_VISIBLE_DEVICES="$GPU" \
            python -u trainval.py \
            > "$RESULT_DIR/train.log" 2>&1
        then
            echo "[TRAIN FAILED] $RUN_TAG"
            tail -60 "$RESULT_DIR/train.log"
            exit 1
        fi

    else
        echo "[SKIP TRAIN] already complete"
    fi

    # ---------------- VAL CHECKPOINT SCAN ----------------
    echo "[VAL SCAN]"

    for CKPT in "$CKPT_DIR"/iter-*.pth.tar; do

        [ -e "$CKPT" ] || continue

        NAME=$(basename "$CKPT" .pth.tar)
        OUT="$VAL_DIR/${NAME}.txt"

        if [ ! -f "$OUT" ]; then
            if ! CUDA_VISIBLE_DEVICES="$GPU" \
                python evaluate_semantic.py \
                --checkpoint "$CKPT" \
                --split val \
                > "$OUT" 2>&1
            then
                echo "[VAL FAILED] $CKPT"
                tail -30 "$OUT"
                exit 1
            fi
        fi
    done

    # ---------------- SELECT BEST VAL TOP1 ----------------
    python - "$VAL_DIR" "$RESULT_DIR/best_checkpoint.txt" <<'PY'
import re
import sys
from pathlib import Path

val_dir = Path(sys.argv[1])
out_file = Path(sys.argv[2])

rows = []

for f in val_dir.glob("iter-*.txt"):
    text = f.read_text(errors="ignore")

    mi = re.search(r"iter-(\d+)", f.name)
    mt = re.search(r"Top-1\s*:\s*([0-9.]+)%", text)

    if mi and mt:
        iteration = int(mi.group(1))
        top1 = float(mt.group(1))
        rows.append((top1, iteration))

if not rows:
    raise RuntimeError("No valid validation results.")

# Highest validation Top-1.
# Exact tie -> earlier checkpoint.
rows.sort(key=lambda x: (-x[0], x[1]))

best_top1, best_iter = rows[0]

out_file.write_text(
    f"iter={best_iter}\n"
    f"val_top1={best_top1}\n"
)

print(
    f"[BEST] iter-{best_iter} "
    f"ValTop1={best_top1:.2f}%"
)
PY

    BEST_ITER=$(grep '^iter=' "$RESULT_DIR/best_checkpoint.txt" | cut -d= -f2)
    BEST_CKPT="$CKPT_DIR/iter-${BEST_ITER}.pth.tar"

    # ---------------- TEST ONCE ----------------
    TEST_FILE="$RESULT_DIR/test.txt"

    if [ ! -f "$TEST_FILE" ]; then
        echo "[TEST] iter-$BEST_ITER"

        if ! CUDA_VISIBLE_DEVICES="$GPU" \
            python evaluate_semantic.py \
            --checkpoint "$BEST_CKPT" \
            --split test \
            > "$TEST_FILE" 2>&1
        then
            echo "[TEST FAILED]"
            tail -40 "$TEST_FILE"
            exit 1
        fi
    else
        echo "[SKIP TEST] result already exists"
    fi

    grep -E \
      "^(Pairs|Queries|Top-1|Top-5|MRR|Hungarian Accuracy|GT Candidate Coverage)" \
      "$TEST_FILE"

done
done

echo
echo "=========================================================="
echo "DONE: $DATASET | 5 folds x seeds 1,42,123"
echo "=========================================================="
