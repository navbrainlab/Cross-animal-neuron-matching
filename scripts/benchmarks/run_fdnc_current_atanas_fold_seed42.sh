#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/klb/nuclr/nuclr
PYTHON=${PYTHON:-/home/ubuntu/anaconda3/envs/nuclr310/bin/python}
FOLD=${FOLD:-0}
GPU=${GPU:-0}
SEED=42

if (( FOLD < 0 || FOLD > 4 )); then
  echo "FOLD must be 0..4" >&2
  exit 2
fi

DATA_ROOT="$ROOT/Data/Atanas_SF_unified_000776/cv5_grouped_v1/fold_${FOLD}"
RUN_ROOT="$ROOT/runs/fdnc_current_grouped_cv_v2/atanas/fold${FOLD}/seed42"
LIST_DIR="$RUN_ROOT/locked_lists"
TRAIN="$ROOT/benchmark_official/adapters/fdnc_finetune/train_fdnc_official_finetune.py"
SELECT="$ROOT/benchmark_official/adapters/fdnc_finetune/select_fdnc_candidate.py"
EVAL="$ROOT/scripts/fair_identity/evaluate_train_reference_ensemble.py"
PRETRAINED="$ROOT/benchmark_official/third_party/fdnc_official/model/model.bin"
EXPECTED_SHA=ab529eb6a886cb6ab3f199b7aaa4e49b82562dc280b3ac736ed61e25d5138ec9

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -c \
  'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' || {
  echo "CUDA is unavailable" >&2
  exit 3
}

mkdir -p "$LIST_DIR" "$RUN_ROOT/candidates"

"$PYTHON" - "$DATA_ROOT" "$LIST_DIR" "$FOLD" <<'PY'
import hashlib, json, sys
from pathlib import Path
import numpy as np

root, out, fold = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])

def uid(path):
    with np.load(path, allow_pickle=True) as data:
        for key in ("recording_uid", "worm_id"):
            if key in data.files:
                value = np.asarray(data[key]).reshape(-1)[0]
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                return str(value)
    return path.stem

payload = {}
for split in ("train", "val", "test"):
    files = sorted((root / split).glob("*.npz"))
    if not files:
        raise RuntimeError(f"empty split: {root / split}")
    ids = [uid(path) for path in files]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate UID in {split}")
    payload[split] = {"files": files, "ids": ids}
for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = set(payload[left]["ids"]) & set(payload[right]["ids"])
    if overlap:
        raise RuntimeError(f"{left}/{right} leakage: {sorted(overlap)}")
for split in ("train", "val", "test"):
    (out / f"{split}.txt").write_text(
        "\n".join(str(path.resolve()) for path in payload[split]["files"]) + "\n"
    )
manifest = {
    "protocol": "fdnc_current_atanas_cv5_grouped_v1",
    "current_grouped_fold": fold,
    "trainer_fold_argument": fold + 1,
    "seed": 42,
    "data_root": str(root),
    **{f"{split}_ids": payload[split]["ids"] for split in ("train", "val", "test")},
    "counts": {split: len(payload[split]["ids"]) for split in ("train", "val", "test")},
    "trainer_inputs": ["train.txt", "val.txt"],
    "test_list_passed_to_trainer": False,
}
path = out / "split_manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
(out / "split_manifest.sha256").write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
print(json.dumps(manifest["counts"]))
PY

for lastn in 1 2; do
  for lr in 5e-6 1e-5; do
    tag="last${lastn}_lr${lr}"
    out="$RUN_ROOT/candidates/$tag"
    if [[ -f "$out/final_results.json" && -f "$out/best.pt" ]]; then
      echo "[reuse] $tag"
      continue
    fi
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u "$TRAIN" \
      --dataset atanas --fold "$((FOLD + 1))" --seed "$SEED" \
      --train-list "$LIST_DIR/train.txt" --val-list "$LIST_DIR/val.txt" \
      --pretrained "$PRETRAINED" --expected-pretrained-sha256 "$EXPECTED_SHA" \
      --save-dir "$out" --mask-key clean_mask --coordinate-scale 200 \
      --unfreeze-last-n "$lastn" --backbone-lr "$lr" --outlier-lr 5e-5 \
      --epochs 60 --pairs-per-epoch 200 --minimum-common 8 \
      --entropy-weight 0.1 --l2sp-weight 1e-4 --weight-decay 1e-2 \
      --scale-jitter 0.05 --coordinate-jitter 0.01 --warmup-epochs 3 \
      --patience 12 --precision fp32 --device cuda \
      2>&1 | tee "$out/train.log"
  done
done

"$PYTHON" "$SELECT" \
  --candidate-root "$RUN_ROOT/candidates" \
  --selected-dir "$RUN_ROOT/selected"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PYTHON" -u "$EVAL" \
  --method fdnc --method-label "fDNC official fine-tuned (current Atanas grouped CV)" \
  --fold-root "$DATA_ROOT" --checkpoint "$RUN_ROOT/selected/best.pt" \
  --normalization zscore --device cuda \
  --cache-dir "$RUN_ROOT/outer_test_medoid_template_v1/cache" \
  --output-dir "$RUN_ROOT/outer_test_medoid_template_v1"

echo "Atanas fDNC fold=$FOLD seed42 complete: $RUN_ROOT"
