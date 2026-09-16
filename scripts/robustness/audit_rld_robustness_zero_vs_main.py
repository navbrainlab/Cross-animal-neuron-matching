#!/usr/bin/env python3
"""Audit severity-zero robustness cells against the canonical Kato main table."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs/rld_robustness_cv5_seed42_v2"
DEFAULT_MAIN = ROOT / "runs/unified_main_benchmark_cv5_seed42_v2"
METHODS = {
    "ours": "NeurID (population atlas)",
    "fdnc": "fDNC",
    "cpd": "CPD",
    "nuclr": "NuCLR",
    "geo": "GeoTransformer",
}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uid(path: str | Path):
    parts = Path(path).stem.split("__")
    return parts[1] if len(parts) >= 3 else Path(path).stem


def row_key_hash(rows):
    payload = "\n".join(
        f"{worm}\t{index}\t{identity}" for worm, index, identity in sorted(rows)
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def emitted_rows(method: str, run: Path, fold: int):
    base = run / "results"
    if method == "cpd":
        path = base / f"cpd/fold{fold}/coord_noise_l0.00_p0/original_runner/per_query.csv"
        rows = read_csv(path)
        return {(row["query_uid"], int(row["query_index"]), row["identity"]) for row in rows}
    if method == "fdnc":
        path = base / f"fdnc/fold{fold}/coord_noise_l0.00_p0/per_query.csv"
        rows = read_csv(path)
        return {(row["query_uid"], int(row["query_index"]), row["identity"]) for row in rows}
    if method == "ours":
        path = base / f"ours_static_main_reference_seed42/fold{fold}/coord_noise_l0.00_p0/queries.csv"
        rows = read_csv(path)
        return {(row["uid"], int(row["node_index"]), row["identity"]) for row in rows}
    if method == "nuclr":
        path = base / f"nuclr/fold{fold}/coord_noise_l0.00_p0/query_level.csv"
        # NuCLR stores the integer target rather than the identity string.  Row
        # identity is filled from the canonical manifest after row-key matching.
        return {(row["query_worm"], int(row["query_row"]), None) for row in read_csv(path)}
    if method == "geo":
        path = base / f"geotransformer/fold{fold}/coord_noise_l0.00_p0/query_level.csv"
        rows = read_csv(path)
        return {(row["query_uid"], int(row["query_index"]), row["identity"]) for row in rows}
    raise ValueError(method)


def observed_reference(
    method: str,
    run: Path,
    fold: int,
    condition: str = "coord_noise_l0.00_p0",
):
    base = run / "results"
    if method == "cpd":
        report = read_json(
            base / f"cpd/fold{fold}/{condition}/original_runner/metrics.json"
        )
        return report["template_selection"]["template_uid"]
    if method == "fdnc":
        report = read_json(base / f"fdnc/fold{fold}/{condition}/metrics.json")
        return report["template_selection"]["template_uid"]
    if method == "nuclr":
        report = read_json(base / f"nuclr/fold{fold}/{condition}/result.json")
        return report["frozen_reference_worm"]
    if method == "geo":
        report = read_json(
            base / f"geotransformer/fold{fold}/{condition}/metrics.json"
        )
        return report["reference_uid"]
    if method == "ours":
        report = read_json(
            base
            / f"ours_static_main_reference_seed42/fold{fold}/{condition}/metrics.json"
        )
        checkpoint = Path(report["checkpoint"])
        return f"learned_atlas:{checkpoint.resolve()}:{sha256(checkpoint)}"
    raise ValueError(method)


def expected_reference(method: str, main: Path, fold: int, main_fold: dict):
    if method in {"cpd", "fdnc", "nuclr"}:
        rows = read_csv(main / "single_medoid_reference_audit.csv")
        name = METHODS[method]
        return next(
            row["reference_uid"]
            for row in rows
            if row["dataset"] == "rld"
            and row["method"] == name
            and int(row["fold"]) == fold
        )
    if method == "geo":
        rows = read_csv(ROOT / main_fold["source"])
        references = {row["reference_uid"] for row in rows}
        if len(references) != 1:
            raise RuntimeError(f"GeoTransformer main-table reference is ambiguous: {references}")
        return next(iter(references))
    if method == "ours":
        report = read_json(
            ROOT
            / f"runs/mprt_v1_1_atlas_medoid_rld_cv5_test_seeds_42_v1/"
            f"fold{fold}/seed42/metrics.json"
        )
        checkpoint = Path(report["checkpoint"])
        return f"learned_atlas:{checkpoint.resolve()}:{sha256(checkpoint)}"
    raise ValueError(method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run = args.run_root.resolve()
    main_root = args.main_root.resolve()
    output = args.output_dir or run / "zero_strength_main_table_audit"

    main_folds = {
        (row["method"], int(row["fold"])): row
        for row in read_csv(main_root / "unified_fold_metrics.csv")
        if row["dataset"] == "rld" and row["top1"]
    }
    fold_values = {
        (row["method"], int(row["fold"])): row
        for row in read_csv(run / "canonical_fold_level.csv")
        if row["kind"] == "coord_noise" and float(row["severity"]) == 0.0
    }
    canonical = {}
    for row in read_csv(main_root / "canonical_query_manifest.csv"):
        if row["dataset"] != "rld":
            continue
        canonical.setdefault(int(row["fold"]), set()).add(
            (row["uid"], int(row["node_index"]), row["identity"])
        )
    corruption = read_json(run / "corruptions/MANIFEST.json")
    clean_conditions = {
        int(row["fold"]): row
        for row in corruption["conditions"]
        if row["kind"] == "coord_noise" and float(row["severity"]) == 0.0
    }

    rows = []
    for method, main_name in METHODS.items():
        for fold in range(5):
            expected = main_folds[(main_name, fold)]
            observed = fold_values[(method, fold)]
            canonical_rows = canonical[fold]
            emitted = emitted_rows(method, run, fold)
            if method == "nuclr":
                canonical_by_position = {
                    (worm, index): identity for worm, index, identity in canonical_rows
                }
                emitted = {
                    (worm, index, canonical_by_position.get((worm, index)))
                    for worm, index, _ in emitted
                }
            manifest_uids = {
                str(record["recording_uid"])
                for record in clean_conditions[fold]["files"]
            }
            robustness_denominator_rows = set()
            canonical_by_worm = {}
            for worm, index, identity in canonical_rows:
                canonical_by_worm.setdefault(worm, set()).add((index, identity))
            for record in clean_conditions[fold]["files"]:
                worm = str(record["recording_uid"])
                kept = [int(value) for value in record["kept_source_indices"]]
                source_to_current = {
                    source: current for current, source in enumerate(kept)
                }
                evaluable = {
                    int(item["row"]): str(item["cell_id"])
                    for item in record["evaluable_query_rows_after_corruption"]
                }
                for source_index, identity in canonical_by_worm.get(worm, set()):
                    current_index = source_to_current.get(source_index)
                    if current_index is not None and evaluable.get(current_index) == identity:
                        robustness_denominator_rows.add((worm, source_index, identity))
            main_test_uids = {
                uid(path)
                for path in (ROOT / f"Data/Dunn_001623/cv5_grouped_v1/fold_{fold}/test").glob("*.npz")
            }
            expected_ref = expected_reference(method, main_root, fold, expected)
            observed_ref = observed_reference(method, run, fold)
            q = int(expected["canonical_queries"])
            correct = round(q * float(expected["top1"]))
            observed_q = int(round(float(observed["queries"])))
            observed_correct = int(round(float(observed["top1_correct"])))
            emitted_subset = None if emitted is None else emitted <= canonical_rows
            emitted_equal = None if emitted is None else emitted == canonical_rows
            row = {
                "method": main_name,
                "fold": fold,
                "main_reference": expected_ref,
                "robustness_reference": observed_ref,
                "reference_exact": int(expected_ref == observed_ref),
                "main_test_worms": len(main_test_uids),
                "robustness_test_worms": len(manifest_uids),
                "test_worm_uids_exact": int(main_test_uids == manifest_uids),
                "main_query_sha256": row_key_hash(canonical_rows),
                "robustness_denominator_sha256": row_key_hash(robustness_denominator_rows),
                "query_rows_exact": int(canonical_rows == robustness_denominator_rows),
                "main_queries": q,
                "robustness_queries": observed_q,
                "emitted_reference_covered_queries": "" if emitted is None else len(emitted),
                "emitted_rows_are_canonical_subset": "" if emitted_subset is None else int(emitted_subset),
                "emitted_rows_equal_canonical": "" if emitted_equal is None else int(emitted_equal),
                "main_top1_correct": correct,
                "robustness_top1_correct": observed_correct,
                "main_top1": float(expected["top1"]),
                "robustness_top1": float(observed["top1"]),
            }
            row["passed"] = int(
                row["reference_exact"]
                and row["test_worm_uids_exact"]
                and row["query_rows_exact"]
                and q == observed_q
                and correct == observed_correct
                and abs(row["main_top1"] - row["robustness_top1"]) <= 1e-12
                and emitted_subset is not False
            )
            rows.append(row)

    reference_cells_total = 0
    reference_cells_passed = 0
    geo_checkpoint_cells_total = 0
    geo_checkpoint_cells_passed = 0
    for condition_row in corruption["conditions"]:
        if condition_row["kind"] not in {"coord_noise", "missing", "outlier"}:
            continue
        fold = int(condition_row["fold"])
        condition_name = Path(condition_row["root"]).name
        if not condition_row.get("train_val_are_original_symlinks"):
            raise RuntimeError(f"{condition_name}: train/val are not declared clean symlinks")
        for method, main_name in METHODS.items():
            expected = main_folds[(main_name, fold)]
            expected_ref = expected_reference(method, main_root, fold, expected)
            observed_ref = observed_reference(
                method, run, fold, condition=condition_name
            )
            reference_cells_total += 1
            reference_cells_passed += int(expected_ref == observed_ref)
            if method == "geo":
                expected_checkpoint = read_json(
                    ROOT
                    / f"runs/unified_medoid_covered_only_retest_v1/"
                    f"geotransformer/rld/fold{fold}/metrics.json"
                )["checkpoint_sha256"]
                observed_checkpoint = read_json(
                    run
                    / f"results/geotransformer/fold{fold}/"
                    f"{condition_name}/metrics.json"
                )["checkpoint_sha256"]
                geo_checkpoint_cells_total += 1
                geo_checkpoint_cells_passed += int(
                    expected_checkpoint == observed_checkpoint
                )

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "fold_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = (
        all(row["passed"] for row in rows)
        and reference_cells_passed == reference_cells_total
        and geo_checkpoint_cells_passed == geo_checkpoint_cells_total
    )
    report = {
        "status": "passed" if passed else "failed",
        "severity_zero_cells_passed": sum(row["passed"] for row in rows),
        "severity_zero_cells_total": len(rows),
        "query_only_reference_cells_passed": reference_cells_passed,
        "query_only_reference_cells_total": reference_cells_total,
        "geotransformer_checkpoint_cells_passed": geo_checkpoint_cells_passed,
        "geotransformer_checkpoint_cells_total": geo_checkpoint_cells_total,
        "denominator": "canonical main-table query cohort; method-reference misses score zero",
        "fold_comparison": str(csv_path.resolve()),
    }
    (output / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Kato robustness severity-zero audit",
        "",
        f"Status: **{'PASSED' if passed else 'FAILED'}** "
        f"({sum(row['passed'] for row in rows)}/{len(rows)} method×fold cells).",
        "",
        f"Frozen-reference check: **{reference_cells_passed}/{reference_cells_total}** "
        "method×condition cells use the exact main-table reference.",
        "",
        f"GeoTransformer checkpoint check: **{geo_checkpoint_cells_passed}/"
        f"{geo_checkpoint_cells_total}** conditions use the exact Table-1 checkpoint.",
        "",
        "The denominator is the canonical main-table query cohort. Rows absent "
        "from a method's frozen reference remain in Q and score zero.",
        "",
        "| Method | Fold | Reference | Canonical Q | Emitted/covered | Top-1 correct | Top-1 | Pass |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        reference = row["main_reference"]
        if reference.startswith("learned_atlas:"):
            _, checkpoint, digest = reference.split(":", 2)
            reference = f"`{Path(checkpoint).name}` ({digest[:12]}…)"
        emitted = row["emitted_reference_covered_queries"] or "aggregate only"
        lines.append(
            f"| {row['method']} | {row['fold']} | {reference} | "
            f"{row['main_queries']} | {emitted} | {row['main_top1_correct']} | "
            f"{100 * row['main_top1']:.4f}% | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "For CPD, fDNC, NuCLR, and GeoTransformer, emitted/covered is the number of query rows "
        "represented by the single-medoid reference; it is not the denominator. "
        "All four now provide exact query-level membership for the severity-zero audit.",
    ])
    (output / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError(f"severity-zero audit failed: {csv_path}")


if __name__ == "__main__":
    main()
