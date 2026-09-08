#!/usr/bin/env bash
# Official fDNC fine-tuning benchmark: validation-only grid -> selected checkpoint -> locked test.
# No `set -euo pipefail`.

ROOT="${ROOT:-/home/ubuntu/klb/nuclr/nuclr}"
PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/nuclr310/bin/python}"
GPUS="${GPUS:-0 1}"
PRETRAINED="$ROOT/baselines/official/third_party/fdnc_official/model/model.bin"
SHA="ab529eb6a886cb6ab3f199b7aaa4e49b82562dc280b3ac736ed61e25d5138ec9"
BASE="$ROOT/baselines/official/runs/fdnc_official_finetuned"
TRAIN="$ROOT/baselines/official/adapters/fdnc_finetune/train_fdnc_official_finetune.py"
SELECT="$ROOT/baselines/official/adapters/fdnc_finetune/select_fdnc_candidate.py"
EVAL="$ROOT/baselines/official/adapters/fdnc_finetune/evaluate_fdnc_official_finetuned.py"

cd "$ROOT" || exit 1
mkdir -p "$BASE" "$ROOT/baselines/official/logs"

if [ ! -f "$PRETRAINED" ]; then
  echo "ERROR missing official pretrained: $PRETRAINED"
  exit 2
fi

observed="$(sha256sum "$PRETRAINED" | awk '{print $1}')"
if [ "$observed" != "$SHA" ]; then
  echo "ERROR official pretrained SHA mismatch: $observed"
  exit 3
fi

read -r -a GPU_ARRAY <<< "$GPUS"
if [ "${#GPU_ARRAY[@]}" -eq 0 ]; then GPU_ARRAY=(0); fi

# Small, pre-declared validation-only grid.
LAST_NS=(1 2)
BACKBONE_LRS=("5e-6" "1e-5")

job=0
pids=()

launch_candidate () {
  dataset="$1"; fold="$2"; seed="$3"; lastn="$4"; lr="$5"
  gpu="${GPU_ARRAY[$((job % ${#GPU_ARRAY[@]}))]}"
  tag="last${lastn}_lr${lr}"
  run="$BASE/$dataset/fold_${fold}/seed_${seed}/candidates/$tag"
  log="$ROOT/baselines/official/logs/fdnc_ft_${dataset}_f${fold}_s${seed}_${tag}.log"

  if [ -f "$run/final_results.json" ] && [ -f "$run/best.pt" ]; then
    echo "[reuse] $dataset f$fold s$seed $tag"
  else
    mkdir -p "$run"
    echo "[train] $dataset f$fold s$seed $tag GPU=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$TRAIN" \
      --dataset "$dataset" \
      --fold "$fold" \
      --seed "$seed" \
      --pretrained "$PRETRAINED" \
      --expected-pretrained-sha256 "$SHA" \
      --save-dir "$run" \
      --unfreeze-last-n "$lastn" \
      --backbone-lr "$lr" \
      --outlier-lr 5e-5 \
      --epochs 60 \
      --pairs-per-epoch 200 \
      --minimum-common 8 \
      --entropy-weight 0.1 \
      --l2sp-weight 1e-4 \
      --weight-decay 1e-2 \
      --scale-jitter 0.05 \
      --coordinate-jitter 0.01 \
      --warmup-epochs 3 \
      --patience 12 \
      --precision fp32 \
      --device cuda:0 \
      > "$log" 2>&1 &
    pids+=("$!")
  fi

  job=$((job+1))
  if [ "${#pids[@]}" -ge "${#GPU_ARRAY[@]}" ]; then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
}

for dataset in atanas rld; do
  for fold in 1 2 3 4 5; do
    for seed in 1 42 123; do
      for lastn in "${LAST_NS[@]}"; do
        for lr in "${BACKBONE_LRS[@]}"; do
          launch_candidate "$dataset" "$fold" "$seed" "$lastn" "$lr"
        done
      done
    done
  done
done

for pid in "${pids[@]}"; do wait "$pid"; done

# Validation-only selection followed by exactly one locked test per fold/seed.
for dataset in atanas rld; do
  for fold in 1 2 3 4 5; do
    for seed in 1 42 123; do
      root="$BASE/$dataset/fold_${fold}/seed_${seed}"
      selected="$root/selected"
      testdir="$root/outer_test_medoid_template_v1"

      "$PYTHON" "$SELECT" \
        --candidate-root "$root/candidates" \
        --selected-dir "$selected" || exit 4

      if [ -f "$testdir/result.json" ]; then
        echo "[reuse test] $dataset f$fold s$seed"
      else
        gpu="${GPU_ARRAY[$(((fold + seed) % ${#GPU_ARRAY[@]}))]}"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$EVAL" \
          --dataset "$dataset" \
          --fold "$fold" \
          --seed "$seed" \
          --checkpoint "$selected/best.pt" \
          --save-dir "$testdir" \
          --precision fp32 \
          --device cuda:0 \
          2>&1 | tee "$ROOT/baselines/official/logs/fdnc_test_medoid_template_v1_${dataset}_f${fold}_s${seed}.log"
      fi
    done
  done
done

"$PYTHON" - <<'PY'
from pathlib import Path
import json, numpy as np, pandas as pd

root=Path("/home/ubuntu/klb/nuclr/nuclr/baselines/official/runs/fdnc_official_finetuned")
out=Path("/home/ubuntu/klb/nuclr/nuclr/baselines/official/results")
out.mkdir(parents=True,exist_ok=True)
rows=[]
for ds in ("atanas","rld"):
    for fold in range(1,6):
        for seed in (1,42,123):
            p=root/ds/f"fold_{fold}"/f"seed_{seed}"/"outer_test_medoid_template_v1"/"result.json"
            x=json.loads(p.read_text())
            rows.append({"dataset":ds,"fold":fold,"seed":seed,**x["metrics"]})
df=pd.DataFrame(rows)
df.to_csv(out/"fdnc_official_finetuned_medoid_template_v1_cv5x3.csv",index=False)

print("\nFINAL fDNC OFFICIAL FINE-TUNED")
for ds,g in df.groupby("dataset"):
    # Main descriptive number: mean±SD across 15 fold×seed runs.
    x=g["ranking_top1"].to_numpy()
    print(f"{ds:7s}: Top1={100*x.mean():.2f}% ± {100*x.std(ddof=1):.2f}%  n_runs={len(x)}")
print("\nSaved:",out/"fdnc_official_finetuned_medoid_template_v1_cv5x3.csv")
PY
