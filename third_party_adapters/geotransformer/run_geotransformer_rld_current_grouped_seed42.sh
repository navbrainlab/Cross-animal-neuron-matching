#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/klb/nuclr/geotransformer_official
DATA_BASE=/home/ubuntu/klb/nuclr/nuclr/Data/Dunn_001623/cv5_grouped_v1
EXP="$ROOT/experiments/geotransformer.rld.semantic"
PY=/home/ubuntu/anaconda3/envs/nuclr310/bin/python

SEED=42
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

cd "$ROOT"

# ----------------------------------------------------------------------
# Discover the exact environment variable used by RLD config.py
# for cfg.data.dataset_root. Do not guess it.
# ----------------------------------------------------------------------
DATA_ENV=$("$PY" - <<'PY'
from pathlib import Path
import re

p = Path("experiments/geotransformer.rld.semantic/config.py")
s = p.read_text()

m = re.search(
    r'_C\.data\.dataset_root\s*=\s*os\.environ\.get\(\s*["\']([^"\']+)["\']',
    s,
    flags=re.S,
)

if not m:
    raise SystemExit(
        "Could not detect dataset-root environment variable from RLD config.py"
    )

print(m.group(1))
PY
)

echo "GeoTransformer RLD dataset environment variable: $DATA_ENV"

# ----------------------------------------------------------------------
# Strong split audit.
# ----------------------------------------------------------------------
"$PY" - <<'PY'
from pathlib import Path
import numpy as np

base = Path(
    "/home/ubuntu/klb/nuclr/nuclr/Data/Dunn_001623/cv5_grouped_v1"
)

def uid(p):
    with np.load(p, allow_pickle=True) as z:
        for k in ("recording_uid", "worm_id"):
            if k in z.files:
                a = np.asarray(z[k]).reshape(-1)
                if len(a):
                    return str(a[0])
    return p.stem

for f in range(5):
    root = base / f"fold_{f}"

    sets = {}
    print(f"\n===== fold{f} =====")

    for split in ("train", "val", "test"):
        files = sorted((root / split).rglob("*.npz"))
        if not files:
            raise RuntimeError(f"fold{f} {split}: EMPTY")

        ids = [uid(p) for p in files]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"fold{f} {split}: duplicate UID")

        sets[split] = set(ids)
        print(f"{split:5s}: {len(files):3d} worms")

    for a, b in (("train","val"),("train","test"),("val","test")):
        x = sets[a] & sets[b]
        if x:
            raise RuntimeError(
                f"fold{f}: {a}/{b} leakage: {sorted(x)}"
            )

    print("disjoint: ✓")

print("\nCURRENT GROUPED CV SPLIT AUDIT PASSED")
PY

run_fold () {
    local fold="$1"
    local gpu="$2"

    local data="$DATA_BASE/fold_${fold}"
    local tag="rld_current_grouped_fold${fold}_seed42"
    local out="$ROOT/output/geotransformer.rld.semantic/$tag"
    local sel="$ROOT/current_grouped_selection/rld/fold${fold}/seed42"

    mkdir -p "$sel"

    echo
    echo "===================================================================================================="
    echo "GeoTransformer CURRENT GROUPED RLD | fold=${fold} seed=42 gpu=${gpu}"
    echo "===================================================================================================="
    echo "data root : $data"
    echo "run tag   : $tag"
    echo "output    : $out"

    # --------------------------------------------------------------
    # Train.
    # Existing completed snapshots are not silently reused unless
    # the final 20k snapshot exists.
    # --------------------------------------------------------------
    if [[ ! -f "$out/snapshots/iter-20000.pth.tar" ]]; then
        rm -rf "$out"

        env \
          CUDA_VISIBLE_DEVICES="$gpu" \
          "$DATA_ENV"="$data" \
          RUN_TAG="$tag" \
          SEED=42 \
          GT_MAX_ITERS=20000 \
          GT_SNAPSHOT_STEPS=1000 \
          "$PY" -u "$EXP/trainval.py" \
          2>&1 | tee "$sel/train.log"
    else
        echo "[REUSE COMPLETE TRAINING] $out"
    fi

    # --------------------------------------------------------------
    # Validation-only checkpoint selection.
    # Scan snapshots; TEST IS NOT OPENED HERE.
    # Highest validation Top-1 wins.
    # Exact tie -> earlier iteration.
    # --------------------------------------------------------------
    "$PY" - "$DATA_ENV" "$data" "$tag" "$out" "$sel" "$gpu" <<'PY'
import os
import re
import subprocess
import sys
from pathlib import Path

data_env, data_root, run_tag, out_dir, sel_dir, gpu = sys.argv[1:]
out_dir = Path(out_dir)
sel_dir = Path(sel_dir)

root = Path("/home/ubuntu/klb/nuclr/geotransformer_official")
exp = root / "experiments/geotransformer.rld.semantic"
py = "/home/ubuntu/anaconda3/envs/nuclr310/bin/python"

snapshots = sorted(
    out_dir.glob("snapshots/iter-*.pth.tar"),
    key=lambda p: int(re.search(r"iter-(\d+)", p.name).group(1))
)

if not snapshots:
    raise RuntimeError(f"No snapshots: {out_dir}")

rows = []

for ckpt in snapshots:
    iteration = int(re.search(r"iter-(\d+)", ckpt.name).group(1))

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env[data_env] = data_root
    env["RUN_TAG"] = run_tag
    env["SEED"] = "42"

    cmd = [
        py,
        "-u",
        str(exp / "evaluate_semantic.py"),
        "--checkpoint",
        str(ckpt),
        "--split",
        "val",
    ]

    cp = subprocess.run(
        cmd,
        cwd=exp,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    text = cp.stdout

    m1 = re.search(r"Top-1\s*:\s*([0-9.]+)%", text)
    m5 = re.search(r"Top-5\s*:\s*([0-9.]+)%", text)
    mm = re.search(r"MRR\s*:\s*([0-9.]+)", text)

    if not m1:
        print(text)
        raise RuntimeError(
            f"Could not parse validation Top-1 from {ckpt}"
        )

    top1 = float(m1.group(1))
    top5 = float(m5.group(1)) if m5 else float("nan")
    mrr = float(mm.group(1)) if mm else float("nan")

    rows.append((iteration, top1, top5, mrr, ckpt))

    print(
        f"fold validation: iter={iteration:5d} "
        f"Top1={top1:6.2f}% "
        f"Top5={top5:6.2f}% "
        f"MRR={mrr:.4f}"
    )

# Highest validation Top1; tie -> earliest checkpoint.
best = sorted(rows, key=lambda x: (-x[1], x[0]))[0]

iteration, top1, top5, mrr, ckpt = best

sel_dir.mkdir(parents=True, exist_ok=True)

(sel_dir / "best_checkpoint.txt").write_text(
    f"iteration={iteration}\n"
    f"validation_top1_percent={top1:.12f}\n"
    f"validation_top5_percent={top5:.12f}\n"
    f"validation_mrr={mrr:.12f}\n"
    f"checkpoint={ckpt.resolve()}\n"
)

with (sel_dir / "validation_scan.csv").open("w") as f:
    f.write("iteration,top1_percent,top5_percent,mrr,checkpoint\n")
    for it, t1, t5, mr, p in rows:
        f.write(
            f"{it},{t1:.12f},{t5:.12f},{mr:.12f},{p.resolve()}\n"
        )

print()
print("[SELECTED]")
print("iteration =", iteration)
print("val Top1 =", f"{top1:.2f}%")
print("checkpoint =", ckpt)
print("selection =", sel_dir / "best_checkpoint.txt")
PY

    echo
    echo "===== fold${fold} SELECTION ====="
    cat "$sel/best_checkpoint.txt"
}

# Two GPUs: folds are distributed, but each fold itself stays isolated.
run_fold 0 "$GPU0" &
P0=$!
run_fold 1 "$GPU1" &
P1=$!

status=0
wait "$P0" || status=1
wait "$P1" || status=1
(( status == 0 )) || exit 1

run_fold 2 "$GPU0" &
P2=$!
run_fold 3 "$GPU1" &
P3=$!

status=0
wait "$P2" || status=1
wait "$P3" || status=1
(( status == 0 )) || exit 1

run_fold 4 "$GPU0"

echo
echo "===================================================================================================="
echo "GeoTransformer CURRENT GROUPED RLD seed42 TRAIN + VAL SELECTION COMPLETE"
echo "===================================================================================================="

for f in 0 1 2 3 4; do
    echo
    echo "fold$f:"
    cat "$ROOT/current_grouped_selection/rld/fold${f}/seed42/best_checkpoint.txt"
done
