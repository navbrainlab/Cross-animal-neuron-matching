#!/usr/bin/env python3
"""Fail-closed structural checks for the public NeuRID repository."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RECORDINGS = {"atanas": 38, "kato_rld": 95, "zebrafish": 1616}
FORBIDDEN_NAMES = {
    "proposed_revision.py",
    "train_proposed_revision.py",
    "train_dynamic_atlas.py",
    "train_population_relative.py",
    "train_transfer.py",
}
FORBIDDEN_PATHS = {
    "baselines/adapters/rgm",
    "baselines/gwot_md",
    "baselines/official/adapters/euclidean",
    "baselines/official/adapters/gwot_md",
    "scripts/lib/benchmark_cv5x3_common.py",
    "src",
}


def require_paths() -> None:
    required = [
        ROOT / "neurid" / "mprt_net" / "model.py",
        ROOT / "neurid" / "mprt_net" / "train.py",
        ROOT / "neurid" / "tests" / "test_model.py",
        ROOT / "scripts" / "neurid" / "run_cv5.py",
        ROOT / "data" / "README.md",
        ROOT / "data" / "SHA256SUMS",
        ROOT / "results" / "PAPER_RESULTS_MANIFEST.md",
        ROOT / "results" / "main_benchmark_full" / "unified_summary.csv",
        ROOT / "results" / "main_benchmark_audit" / "predictions" / "canonical_complete_predictions.csv",
        ROOT / "results" / "main_benchmark_shared" / "unified_summary.csv",
        ROOT / "results" / "component_ablation" / "seed42" / "component_ablation_seed42_summary.csv",
        ROOT / "results" / "fixed_matchers_cv5_seed42" / "summary.json",
        ROOT / "results" / "atlas_relation_mask_cv5_seed42" / "summary.csv",
        ROOT / "results" / "activity_window_cv5_seed42" / "summary.csv",
        ROOT / "results" / "robustness" / "rejection_targets" / "summary.csv",
        ROOT / "results" / "zebrafish_lofo8" / "zebrafish_benchmark_complete.csv",
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("missing release files: " + ", ".join(missing))


def validate_source() -> tuple[int, int]:
    python_files = [path for path in ROOT.rglob("*.py") if ".git" not in path.parts]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    json_files = [path for path in ROOT.rglob("*.json") if ".git" not in path.parts]
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    return len(python_files), len(json_files)


def validate_boundary() -> None:
    forbidden = sorted(path for path in ROOT.rglob("*") if path.name in FORBIDDEN_NAMES)
    if forbidden:
        raise RuntimeError("alternative model files present: " + ", ".join(str(path) for path in forbidden))
    forbidden_paths = sorted(
        relative for relative in FORBIDDEN_PATHS if (ROOT / relative).exists()
    )
    if forbidden_paths:
        raise RuntimeError(
            "non-paper model paths present: " + ", ".join(forbidden_paths)
        )
    caches = sorted(
        path for path in ROOT.rglob("*")
        if path.name in {"__pycache__", ".pytest_cache"} and ".git" not in path.parts
    )
    if caches:
        raise RuntimeError("cache directories present in public tree: " + ", ".join(str(path) for path in caches))
    oversized = [
        path for path in ROOT.rglob("*")
        if path.is_file() and not path.is_symlink() and path.stat().st_size >= 100_000_000
    ]
    if oversized:
        raise RuntimeError("GitHub 100 MB limit exceeded: " + ", ".join(str(path) for path in oversized))


def validate_data() -> int:
    physical_counts = {
        "atanas": len(list((ROOT / "data" / "atanas" / "recordings").glob("*.npz"))),
        "kato_rld": len(list((ROOT / "data" / "kato_rld" / "recordings").glob("*.npz"))),
        "zebrafish": len(list((ROOT / "data" / "zebrafish").glob("fold_*/*/*.npz"))),
    }
    if physical_counts != EXPECTED_RECORDINGS:
        raise RuntimeError(f"unexpected dataset counts: {physical_counts}")
    broken = [path for path in (ROOT / "data").rglob("*.npz") if path.is_symlink() and not path.exists()]
    if broken:
        raise RuntimeError("broken dataset links: " + ", ".join(str(path) for path in broken[:10]))

    lines = [line for line in (ROOT / "data" / "SHA256SUMS").read_text().splitlines() if line]
    expected_checksum_rows = sum(EXPECTED_RECORDINGS.values())
    if len(lines) != expected_checksum_rows:
        raise RuntimeError(f"SHA256SUMS has {len(lines)} rows, expected {expected_checksum_rows}")
    for line in lines:
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = ROOT / relative
        if len(digest) != 64 or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"invalid checksum entry: {line}")

    # Load one real file from each dataset and check the public contract.
    import numpy as np

    examples = [
        next((ROOT / "data" / "atanas" / "recordings").glob("*.npz")),
        next((ROOT / "data" / "kato_rld" / "recordings").glob("*.npz")),
        next((ROOT / "data" / "zebrafish").glob("fold_*/*/*.npz")),
    ]
    for path in examples:
        with np.load(path, allow_pickle=False) as payload:
            missing = {"activity_raw", "xyz", "cell_id"}.difference(payload.files)
            if missing:
                raise RuntimeError(f"{path} misses required arrays: {sorted(missing)}")
    return expected_checksum_rows


def validate_checksum_contents() -> None:
    for line in (ROOT / "data" / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative.lstrip("*")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {path.relative_to(ROOT)}")


def validate_paper_results() -> int:
    """Check the headline values against the frozen paper tables."""

    def rows(relative: str) -> list[dict[str, str]]:
        with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    main = rows("results/main_benchmark_full/unified_summary.csv")
    observed = {
        (row["dataset"], row["method"]): float(row["top1_mean"])
        for row in main
        if row["top1_mean"]
    }
    expected = {
        ("atanas", "NeurID (population atlas)"): 0.749153076086016,
        ("rld", "NeurID (population atlas)"): 0.6393422709575947,
    }
    for key, value in expected.items():
        if abs(observed.get(key, -1.0) - value) > 1e-12:
            raise RuntimeError(f"paper main-table mismatch for {key}: {observed.get(key)}")

    fish = rows("results/zebrafish_lofo8/zebrafish_benchmark_complete.csv")
    ours = next(row for row in fish if row["method"] == "Ours")
    if abs(float(ours["top1_mean_pct"]) - 97.69) > 1e-12:
        raise RuntimeError("zebrafish paper-table mismatch")

    result_files = [path for path in (ROOT / "results").rglob("*") if path.is_file()]
    if len(result_files) < 1000:
        raise RuntimeError(f"paper result bundle unexpectedly small: {len(result_files)} files")
    return len(result_files)


def main() -> None:
    require_paths()
    python_count, json_count = validate_source()
    validate_boundary()
    data_count = validate_data()
    result_count = validate_paper_results()
    print(
        "release checks PASS: "
        f"{python_count} Python files; {json_count} JSON files; "
        f"{data_count} physical NPZ files indexed; {result_count} result files"
    )


if __name__ == "__main__":
    main()
