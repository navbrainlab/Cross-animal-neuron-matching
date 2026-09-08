#!/usr/bin/env bash
set -euo pipefail

NUCLR_ROOT=/home/ubuntu/klb/nuclr/nuclr
GEO_ROOT=/home/ubuntu/klb/nuclr/geotransformer_official
PYTHON=${PYTHON:-/home/ubuntu/anaconda3/envs/nuclr310/bin/python}
DATASET=${DATASET:-atanas}
FOLD=${FOLD:-0}
GPU=${GPU:-0}
SEED=42

if [[ "$DATASET" != atanas && "$DATASET" != rld ]]; then
  echo "DATASET must be atanas or rld" >&2
  exit 2
fi
if (( FOLD < 0 || FOLD > 4 )); then
  echo "FOLD must be 0..4" >&2
  exit 2
fi

if [[ "$DATASET" == atanas ]]; then
  DATA_BASE="$NUCLR_ROOT/Data/Atanas_SF_unified_000776/cv5_grouped_v1"
  COARSE_K=96
  WARMUP=500
else
  DATA_BASE="$NUCLR_ROOT/Data/Dunn_001623/cv5_grouped_v1"
  COARSE_K=32
  WARMUP=200
fi

EXP_NAME="geotransformer.${DATASET}.semantic"
EXP="$GEO_ROOT/experiments/$EXP_NAME"
DATA_ROOT="$DATA_BASE/fold_${FOLD}"
RUN_TAG="${DATASET}_current_grouped_fold${FOLD}_seed42"
OUTPUT="$GEO_ROOT/output/$EXP_NAME/$RUN_TAG"
SELECTION="$GEO_ROOT/current_grouped_selection/$DATASET/fold${FOLD}/seed42"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -c \
  'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' || {
  echo "CUDA unavailable" >&2
  exit 3
}

for split in train val test; do
  [[ -d "$DATA_ROOT/$split" ]] || { echo "missing $DATA_ROOT/$split" >&2; exit 4; }
done

DATA_ENV=$("$PYTHON" - "$EXP/config.py" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
match = re.search(
    r'_C\.data\.dataset_root\s*=\s*os\.environ\.get\(\s*["\']([^"\']+)["\']',
    text,
    flags=re.S,
)
if not match:
    raise RuntimeError("dataset-root environment variable not found")
print(match.group(1))
PY
)

mkdir -p "$SELECTION"

echo "dataset   : $DATASET"
echo "fold      : $FOLD"
echo "seed      : $SEED"
echo "data root : $DATA_ROOT"
echo "run tag   : $RUN_TAG"
echo "output    : $OUTPUT"

if [[ ! -f "$OUTPUT/snapshots/iter-20000.pth.tar" ]]; then
  env CUDA_VISIBLE_DEVICES="$GPU" \
    "$DATA_ENV"="$DATA_ROOT" \
    RUN_TAG="$RUN_TAG" SEED=42 \
    GT_COARSE_K="$COARSE_K" GT_WARMUP="$WARMUP" \
    GT_MAX_ITERS=20000 GT_SNAPSHOT_STEPS=1000 \
    "$PYTHON" -u "$EXP/trainval.py" \
    2>&1 | tee "$SELECTION/train.log"
else
  echo "[reuse completed training] $OUTPUT/snapshots/iter-20000.pth.tar"
fi

"$PYTHON" - "$DATA_ENV" "$DATA_ROOT" "$RUN_TAG" "$OUTPUT" "$SELECTION" "$GPU" "$EXP" <<'PY'
import os, re, subprocess, sys
from pathlib import Path

data_env, data_root, run_tag, output, selection, gpu, exp = sys.argv[1:]
output, selection, exp = Path(output), Path(selection), Path(exp)
python = "/home/ubuntu/anaconda3/envs/nuclr310/bin/python"
snapshots = sorted(
    output.glob("snapshots/iter-*.pth.tar"),
    key=lambda path: int(re.search(r"iter-(\d+)", path.name).group(1)),
)
if not snapshots:
    raise RuntimeError(f"no snapshots under {output}")

rows = []
for checkpoint in snapshots:
    iteration = int(re.search(r"iter-(\d+)", checkpoint.name).group(1))
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        data_env: data_root,
        "RUN_TAG": run_tag,
        "SEED": "42",
    })
    completed = subprocess.run(
        [python, "-u", str(exp / "evaluate_semantic.py"),
         "--checkpoint", str(checkpoint), "--split", "val"],
        cwd=exp,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    text = completed.stdout
    top1 = re.search(r"Top-1\s*:\s*([0-9.]+)%", text)
    top5 = re.search(r"Top-5\s*:\s*([0-9.]+)%", text)
    mrr = re.search(r"MRR\s*:\s*([0-9.]+)", text)
    if not top1:
        raise RuntimeError(f"cannot parse validation output for {checkpoint}\n{text}")
    rows.append((
        iteration,
        float(top1.group(1)),
        float(top5.group(1)) if top5 else float("nan"),
        float(mrr.group(1)) if mrr else float("nan"),
        checkpoint.resolve(),
    ))
    print(f"iter={iteration:5d} val Top1={rows[-1][1]:.2f}%")

best = sorted(rows, key=lambda row: (-row[1], row[0]))[0]
iteration, top1, top5, mrr, checkpoint = best
selection.mkdir(parents=True, exist_ok=True)
(selection / "best_checkpoint.txt").write_text(
    f"iteration={iteration}\n"
    f"validation_top1_percent={top1:.12f}\n"
    f"validation_top5_percent={top5:.12f}\n"
    f"validation_mrr={mrr:.12f}\n"
    f"checkpoint={checkpoint}\n"
)
with (selection / "validation_scan.csv").open("w") as handle:
    handle.write("iteration,top1_percent,top5_percent,mrr,checkpoint\n")
    for row in rows:
        handle.write(f"{row[0]},{row[1]:.12f},{row[2]:.12f},{row[3]:.12f},{row[4]}\n")
print(f"selected iter={iteration} val Top1={top1:.2f}% checkpoint={checkpoint}")
PY

echo "selection: $SELECTION/best_checkpoint.txt"
