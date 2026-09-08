#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/home/ubuntu/klb/nuclr/nuclr}"
python_bin="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/nuclr310/bin/python}"
source_root="${SOURCE_ROOT:-$repo_root/Data/Zebrafish_LOFO8_joint_from_scratch/fold_1/features}"
data_root="${DATA_ROOT:-$repo_root/Data/Zebrafish_MPRT_LOFO8_60m/fold_1}"
run_root="${RUN_ROOT:-$repo_root/runs/mprt_v1_1/zebrafish_lofo/fold1/seed42/full}"
checkpoint="${CHECKPOINT:-$run_root/best.pt}"
gpu="${GPU:-0}"

cd "$repo_root"

[[ -f "$checkpoint" ]] || {
  echo "Missing validation-selected checkpoint: $checkpoint" >&2
  exit 1
}
[[ -f "$run_root/history.jsonl" ]] || {
  echo "Missing training history: $run_root/history.jsonl" >&2
  exit 1
}

"$python_bin" "$repo_root/scripts/zebrafish/unlock_zebrafish_mprt_fold1_test.py" \
  --source-root "$source_root" \
  --output-root "$data_root" \
  --checkpoint "$checkpoint" \
  --min-shared 20

cd "$repo_root/mprt_net_v1_1"

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m mprt_net.self_check \
  --dataset-root "$data_root" \
  --split test \
  --device cuda

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m mprt_net.evaluate \
  --dataset-root "$data_root" \
  --split test \
  --checkpoint "$checkpoint" \
  --activity-length 128 \
  --min-shared 20 \
  --device cuda \
  --output "$run_root/test_metrics.json" \
  2>&1 | tee "$run_root/test_eval.log"

"$python_bin" - "$run_root/test_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metrics = json.loads(path.read_text())
block = metrics.get("test", metrics)

def first(*keys):
    for key in keys:
        if key in block and block[key] is not None:
            return float(block[key])
    return None

rows = [
    ("Queries", first("num_queries", "queries"), False),
    ("Top-1", first("ranking_top1", "top1", "top1_real"), True),
    ("Top-5", first("top5", "top5_real"), True),
    ("MRR", first("mrr", "mrr_real"), False),
    ("Hungarian", first("assignment_top1", "hungarian_accuracy"), True),
]
print("=" * 72)
print("ZEBRAFISH MPRT FOLD1 LOCKED TEST")
print("=" * 72)
for label, value, percent in rows:
    if value is None:
        continue
    if label == "Queries":
        print(f"{label:<12}: {int(value)}")
    elif percent:
        print(f"{label:<12}: {100.0 * value:.2f}%")
    else:
        print(f"{label:<12}: {value:.4f}")
print("metrics     :", path)
PY
