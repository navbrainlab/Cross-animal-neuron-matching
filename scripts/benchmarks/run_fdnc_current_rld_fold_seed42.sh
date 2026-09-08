#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/klb/nuclr/nuclr
cd "$ROOT"

FOLD="${FOLD:-0}"          # current grouped fold: 0..4
SEED="${SEED:-42}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

if (( FOLD < 0 || FOLD > 4 )); then
  echo "ERROR: FOLD must be 0..4" >&2
  exit 2
fi

# The current official trainer requires fold numbering 1..5.
# Because explicit --train-list/--val-list are supplied below, this number is
# metadata/benchmark fold identity; the actual data split is the current
# cv5_grouped_v1 fold selected by FOLD.
OFFICIAL_FOLD=$((FOLD + 1))

DATA_ROOT="$ROOT/Data/Dunn_001623/cv5_grouped_v1/fold_${FOLD}"
TRAIN_SCRIPT="$ROOT/baselines/official/adapters/fdnc_finetune/train_fdnc_official_finetune.py"
MEDOID_EVAL_SCRIPT="$ROOT/scripts/fair_identity/evaluate_train_reference_ensemble.py"
PRETRAINED="$ROOT/baselines/official/third_party/fdnc_official/model/model.bin"
EXPECTED_PRETRAINED_SHA256="ab529eb6a886cb6ab3f199b7aaa4e49b82562dc280b3ac736ed61e25d5138ec9"

RUN_ROOT="$ROOT/runs/fdnc_current_grouped_cv_v2/rld/fold${FOLD}/seed${SEED}"
LIST_DIR="$RUN_ROOT/locked_lists"
SELECTED="$RUN_ROOT/selected"
CLEAN_OUT="$RUN_ROOT/outer_test_medoid_template_v1"

mkdir -p "$RUN_ROOT" "$LIST_DIR" "$RUN_ROOT/candidates"

for p in "$DATA_ROOT/train" "$DATA_ROOT/val" "$DATA_ROOT/test"; do
  [[ -d "$p" ]] || { echo "ERROR: missing $p" >&2; exit 2; }
done
for p in "$TRAIN_SCRIPT" "$MEDOID_EVAL_SCRIPT" "$PRETRAINED"; do
  [[ -f "$p" ]] || { echo "ERROR: missing $p" >&2; exit 2; }
done

# ---------------------------------------------------------------------------
# Lock current grouped train/val/test identities and write explicit list files.
# Trainer receives ONLY train.txt and val.txt. Test.txt is audit metadata only.
# ---------------------------------------------------------------------------
python - "$DATA_ROOT" "$LIST_DIR" "$FOLD" "$SEED" "$OFFICIAL_FOLD" <<'PY'
import hashlib, json, sys
from pathlib import Path
import numpy as np

data_root = Path(sys.argv[1]).resolve()
list_dir = Path(sys.argv[2]).resolve()
fold = int(sys.argv[3])
seed = int(sys.argv[4])
official_fold = int(sys.argv[5])

def uid(path: Path) -> str:
    with np.load(path, allow_pickle=True) as z:
        for key in ("recording_uid", "worm_id"):
            if key in z.files:
                a = np.asarray(z[key]).reshape(-1)
                if len(a):
                    v = a[0]
                    if isinstance(v, np.generic):
                        v = v.item()
                    if isinstance(v, bytes):
                        v = v.decode("utf-8", errors="replace")
                    s = str(v).strip()
                    if s:
                        return s
    parts = path.stem.split("__")
    return parts[1] if len(parts) >= 3 else path.stem

payload = {}
for split in ("train", "val", "test"):
    files = sorted((data_root / split).glob("*.npz"))
    if not files:
        raise RuntimeError(f"empty split: {data_root / split}")
    ids = [uid(p) for p in files]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate recording UID in {split}")
    payload[split] = {"files": files, "ids": ids}

for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = sorted(set(payload[a]["ids"]) & set(payload[b]["ids"]))
    if overlap:
        raise RuntimeError(f"{a}/{b} leakage: {overlap}")

list_dir.mkdir(parents=True, exist_ok=True)
for split in ("train", "val", "test"):
    p = list_dir / f"{split}.txt"
    p.write_text(
        "\n".join(str(x.resolve()) for x in payload[split]["files"]) + "\n",
        encoding="utf-8",
    )

manifest = {
    "protocol": "fdnc_current_rld_cv5_grouped_v1",
    "current_grouped_fold": fold,
    "trainer_fold_argument": official_fold,
    "seed": seed,
    "data_root": str(data_root),
    "train_ids": payload["train"]["ids"],
    "val_ids": payload["val"]["ids"],
    "test_ids": payload["test"]["ids"],
    "counts": {s: len(payload[s]["ids"]) for s in ("train","val","test")},
    "trainer_inputs": ["train.txt", "val.txt"],
    "test_list_passed_to_trainer": False,
}
m = list_dir / "split_manifest.json"
m.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
digest = hashlib.sha256(m.read_bytes()).hexdigest()
(list_dir / "split_manifest.sha256").write_text(
    f"{digest}  {m.name}\n", encoding="utf-8"
)
print(json.dumps(manifest["counts"]))
print("manifest_sha256 =", digest)
PY

TRAIN_LIST="$LIST_DIR/train.txt"
VAL_LIST="$LIST_DIR/val.txt"

run_candidate () {
  local name="$1"
  local unfreeze="$2"
  local lr="$3"
  local gpu="$4"
  local out="$RUN_ROOT/candidates/$name"

  if [[ -f "$out/best.pt" ]] && python - "$out" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ok = False
for p in root.rglob("*.json"):
    try:
        x = json.loads(p.read_text())
    except Exception:
        continue
    if not isinstance(x, dict):
        continue
    bv = x.get("best_validation")
    if (
        isinstance(bv, dict)
        and all(k in x for k in ("dataset", "fold", "seed", "best_epoch"))
        and "ranking_top1" in bv
        and "mrr" in bv
        and x.get("test_data_accessed", x.get("test_data_used")) is False
    ):
        ok = True
        break
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[REUSE COMPLETE] $name"
    return 0
  fi

  rm -rf "$out"
  mkdir -p "$out"

  echo "[START] $name unfreeze=$unfreeze lr=$lr gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$TRAIN_SCRIPT" \
    --dataset rld \
    --fold "$OFFICIAL_FOLD" \
    --seed "$SEED" \
    --train-list "$TRAIN_LIST" \
    --val-list "$VAL_LIST" \
    --pretrained "$PRETRAINED" \
    --expected-pretrained-sha256 "$EXPECTED_PRETRAINED_SHA256" \
    --save-dir "$out" \
    --mask-key labeled_mask \
    --coordinate-scale 200 \
    --unfreeze-last-n "$unfreeze" \
    --epochs 60 \
    --pairs-per-epoch 200 \
    --minimum-common 8 \
    --backbone-lr "$lr" \
    --outlier-lr 5e-5 \
    --weight-decay 1e-2 \
    --l2sp-weight 1e-4 \
    --gradient-clip 1.0 \
    --scale-jitter 0.05 \
    --coordinate-jitter 0.01 \
    --warmup-epochs 3 \
    --patience 12 \
    --precision bf16 \
    --device cuda \
    2>&1 | tee "$out/train.log"
}

# Historical final fDNC candidate grid:
#   unfreeze {1,2} × backbone LR {1e-5,5e-6}
run_candidate last1_lr1e-5 1 1e-5 "$GPU0" &
P1=$!
run_candidate last2_lr1e-5 2 1e-5 "$GPU1" &
P2=$!
status=0
wait "$P1" || status=1
wait "$P2" || status=1
(( status == 0 )) || { echo "ERROR: first candidate pair failed" >&2; exit 3; }

run_candidate last1_lr5e-6 1 5e-6 "$GPU0" &
P3=$!
run_candidate last2_lr5e-6 2 5e-6 "$GPU1" &
P4=$!
status=0
wait "$P3" || status=1
wait "$P4" || status=1
(( status == 0 )) || { echo "ERROR: second candidate pair failed" >&2; exit 3; }

# ---------------------------------------------------------------------------
# Validation-only selection + clean current-grouped medoid evaluation.
# The recovery helper scans the CURRENT trainer's terminal JSON schema, selects
# by validation ranking_top1 -> MRR -> stable candidate name, copies best.pt,
# and evaluates the locked clean test with the same fair medoid evaluator.
# ---------------------------------------------------------------------------
RECOVERY="$ROOT/scripts/robustness/recover_fdnc_current_rld_fold_seed42.py"
[[ -f "$RECOVERY" ]] || {
  echo "ERROR: missing recovery helper: $RECOVERY" >&2
  exit 4
}

CUDA_VISIBLE_DEVICES="$GPU0" python -u "$RECOVERY" \
  --fold "$FOLD" \
  --seed "$SEED" \
  --gpu "$GPU0"

echo
echo "===================================================================================================="
echo "CURRENT GROUPED-CV fDNC FOLD ${FOLD} SEED ${SEED} COMPLETE"
echo "===================================================================================================="
echo "selected checkpoint:"
ls -lh "$RUN_ROOT/selected/best.pt"
echo
echo "selection:"
cat "$RUN_ROOT/selected/selection.json"
echo
echo "clean medoid metrics:"
cat "$RUN_ROOT/outer_test_medoid_template_v1/metrics.json"
