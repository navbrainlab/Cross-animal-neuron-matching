#!/usr/bin/env python3
"""Fail-closed audit for the shared, hashed RLD corruption manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()
    path = args.manifest.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol") != "RLD_CV5_seed42_controlled_robustness_v2_hashed":
        raise RuntimeError("Not the locked v2 manifest")
    rows = data.get("conditions", [])
    # coordinate + activity each use 0 plus four levels x three draws;
    # missing + distractors each use 0 plus five levels x three draws.
    # coord: 4 nonzero levels; activity/missing/outlier: 5 nonzero levels.
    expected = 5 * ((1 + 4 * 3) + 3 * (1 + 5 * 3))
    if len(rows) != expected:
        raise RuntimeError(f"condition count {len(rows)} != {expected}")
    keys = set()
    files_checked = 0
    for row in rows:
        key = (int(row["fold"]), str(row["kind"]), float(row["severity"]),
               int(row["perturbation_seed"]))
        if key in keys:
            raise RuntimeError(f"duplicate condition {key}")
        keys.add(key)
        if len(row.get("files", [])) != 19:
            raise RuntimeError(f"{key}: expected 19 worms")
        for item in row["files"]:
            source, output = Path(item["source"]), Path(item["output"])
            if sha256_file(source) != item["source_sha256"]:
                raise RuntimeError(f"source hash mismatch: {source}")
            if sha256_file(output) != item["output_sha256"]:
                raise RuntimeError(f"output hash mismatch: {output}")
            with np.load(output, allow_pickle=True) as z:
                n = len(np.asarray(z["xyz"]))
                if n != int(item["output_neuron_rows"]):
                    raise RuntimeError(f"row mismatch: {output}")
                if row["kind"] == "outlier" and float(row["severity"]) > 0:
                    n0 = int(item["source_neuron_rows"])
                    valid = np.asarray(z["valid_xyz_mask"], dtype=bool)
                    if not valid[n0:].all():
                        raise RuntimeError(f"distractor filtered by valid_xyz_mask: {output}")
                    for mask in ("labeled_mask", "certain_mask", "clean_mask"):
                        if mask in z.files and np.asarray(z[mask], dtype=bool)[n0:].any():
                            raise RuntimeError(f"distractor is supervised in {mask}: {output}")
                if row["kind"] == "activity_noise":
                    with np.load(source, allow_pickle=True) as source_z:
                        if not np.array_equal(z["xyz"], source_z["xyz"]):
                            raise RuntimeError(f"activity noise changed xyz: {output}")
                        if np.asarray(z["activity_raw"]).shape != np.asarray(
                            source_z["activity_raw"]
                        ).shape:
                            raise RuntimeError(f"activity noise changed trace shape: {output}")
                        equal = np.array_equal(z["activity_raw"], source_z["activity_raw"])
                        if (float(row["severity"]) == 0.0) != equal:
                            raise RuntimeError(
                                f"activity severity/change invariant failed: {output}"
                            )
            files_checked += 1
    digest = sha256_file(path)
    print(json.dumps({
        "status": "passed", "manifest": str(path), "manifest_sha256": digest,
        "conditions": len(rows), "files_checked": files_checked,
        "folds": [0, 1, 2, 3, 4], "worms_per_condition": 19,
    }, indent=2))


if __name__ == "__main__":
    main()
