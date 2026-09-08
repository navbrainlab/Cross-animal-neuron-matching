#!/usr/bin/env python3
"""Fail-closed, dependency-free checks for the GitHub source release."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results" / "main_benchmark_seed42"
EXPECTED_MISSING = {
    (dataset, method)
    for dataset in ("atanas", "rld")
    for method in ("CRF_ID", "GWOT-MD", "GWOT-MD (our adaptation)")
}
DETERMINISTIC_METHODS = {"CPD", "StatAtlas", "Vanilla FGW"}


def require_paths() -> None:
    required = [
        ROOT / "neurid" / "mprt_net" / "model.py",
        ROOT / "neurid" / "mprt_net" / "train.py",
        ROOT / "neurid" / "mprt_net" / "evaluate.py",
        ROOT / "MODEL_CODE_GUIDE.md",
        ROOT / "RUN_DATASETS.md",
        ROOT / "scripts" / "zm9624" / "run_strict_loso.sh",
        ROOT / "baselines" / "README.md",
        ROOT / "results" / "README.md",
        MAIN / "VERIFIED_RESULTS.md",
        MAIN / "verified_fold_cells.csv",
        MAIN / "verified_summary.csv",
        ROOT / "results" / "component_ablation" / "seed42" / "SUMMARY.md",
        ROOT / "results" / "zebrafish_lofo8" / "zebrafish_benchmark_complete.md",
        ROOT / "results" / "scaling" / "runtime" / "full_wallclock_runtime.csv",
        ROOT
        / "results"
        / "scaling"
        / "training_population"
        / "test_scaling_summary.csv",
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("missing required release files: " + ", ".join(missing))


def validate_json() -> int:
    paths = sorted(ROOT.rglob("*.json"))
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    return len(paths)


def validate_python() -> int:
    paths = sorted(ROOT.rglob("*.py"))
    for path in paths:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
    return len(paths)


def validate_main_readiness() -> tuple[int, int]:
    readiness = json.loads((MAIN / "readiness.json").read_text(encoding="utf-8"))
    seen_missing: set[tuple[str, str]] = set()
    publishable = 0
    for row in readiness:
        key = (row["dataset"], row["method"])
        statuses = row["statuses"]
        if row["fold_cells"] != 5 or len(statuses) != 5:
            raise RuntimeError(f"{key} does not contain exactly five fold cells")
        if key in EXPECTED_MISSING:
            if row["publishable"] or set(statuses) != {"MISSING"}:
                raise RuntimeError(f"unresolved group is not fail-closed: {key}")
            seen_missing.add(key)
        else:
            if not row["publishable"] or set(statuses) != {"PASS"}:
                raise RuntimeError(f"populated group failed its audit: {key}")
            publishable += 1
    if seen_missing != EXPECTED_MISSING:
        raise RuntimeError("readiness file does not contain the expected unresolved groups")

    with (MAIN / "verified_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    if len(summary_rows) != publishable:
        raise RuntimeError(
            f"verified_summary has {len(summary_rows)} rows; expected {publishable}"
        )
    for row in summary_rows:
        expected_seed = "deterministic" if row["method"] in DETERMINISTIC_METHODS else "42"
        if row["seed"] != expected_seed or int(row["folds"]) != 5:
            raise RuntimeError(
                "formal result is not CV5 x seed42 (or deterministic): "
                f"{row['dataset']}/{row['method']} folds={row['folds']} seed={row['seed']}"
            )
    return publishable, len(seen_missing)


def validate_manifest_hash() -> None:
    manifest = MAIN / "canonical_fold_manifest.csv"
    expected = (MAIN / "canonical_fold_manifest.sha256").read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("canonical fold manifest SHA-256 mismatch")


def validate_release_boundary() -> None:
    forbidden_suffixes = {".ckpt", ".h5", ".mat", ".npy", ".npz", ".pkl", ".pt"}
    forbidden = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix in forbidden_suffixes]
    if forbidden:
        names = ", ".join(path.relative_to(ROOT).as_posix() for path in forbidden[:10])
        raise RuntimeError("dataset/checkpoint artifacts present in source release: " + names)


def main() -> None:
    require_paths()
    json_count = validate_json()
    python_count = validate_python()
    publishable, unresolved = validate_main_readiness()
    validate_manifest_hash()
    validate_release_boundary()
    result_count = sum(path.is_file() for path in (ROOT / "results").rglob("*"))
    print(
        "release checks PASS: "
        f"{python_count} Python files parsed; {json_count} JSON files parsed; "
        f"{publishable} main-table groups publishable; {unresolved} explicitly unresolved; "
        f"{result_count} compact result files indexed"
    )


if __name__ == "__main__":
    main()
