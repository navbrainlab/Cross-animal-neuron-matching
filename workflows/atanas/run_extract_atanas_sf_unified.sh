#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-$HOME/klb/nuclr/nuclr}"

PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/dandi310/bin/python}"
SCRIPT="${SCRIPT:-./workflows/atanas/extract_atanas_sf_unified_from_dandi.py}"
SPLIT_ROOT="${SPLIT_ROOT:-Data/Atanas_activity_npz_70_15_15}"
OUTPUT_ROOT="${OUTPUT_ROOT:-Data/Atanas_SF_unified_000776}"
LOCAL_NWB_ROOT="${LOCAL_NWB_ROOT:-/media/ubuntu/65ccd0d4-7e99-4548-b09a-bee59b6ae7fb/klb_data/dandi_data/000776}"
CACHE_DIR="${CACHE_DIR:-/tmp/remfile_sf_unified_cache}"

"$PYTHON_BIN" - <<'PY'
missing = []
for name in ("numpy", "h5py", "dandi", "remfile"):
    try:
        __import__(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit(
        "Missing packages: " + ", ".join(missing) +
        "\nInstall with: python -m pip install -U dandi remfile h5py numpy"
    )
print("Dependencies available")
PY

"$PYTHON_BIN" -u "$SCRIPT" \
  --split-root "$SPLIT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --local-nwb-root "$LOCAL_NWB_ROOT" \
  --cache-dir "$CACHE_DIR"

printf '\nUnified SF dataset written to: %s\n' "$OUTPUT_ROOT"
printf 'Clean model-ready split: %s/clean\n' "$OUTPUT_ROOT"
printf 'Full population-context split: %s/full\n' "$OUTPUT_ROOT"
