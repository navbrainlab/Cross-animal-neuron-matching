#!/usr/bin/env python3
"""Freeze and audit the unified five-fold, seed-42 benchmark inputs.

This script is deliberately fail-closed: an artifact is publishable only when
all five fold cells exist and each cell carries enough provenance to tie it to
the canonical ``cv5_grouped_v1`` fold.  It never reads the legacy cv5x3
summary as a source of benchmark numbers.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/unified_main_benchmark_cv5_seed42_v1"
DATASETS = {
    "atanas": REPO / "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
    "rld": REPO / "Data/Dunn_001623/cv5_grouped_v1",
}
SPLITS = ("train", "val", "test")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def scalar_text(value: Any) -> str:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"expected scalar identity, got shape {np.asarray(value).shape}")
    item = array[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def uid_for(path: Path) -> str:
    with np.load(path, allow_pickle=True) as data:
        for key in ("recording_uid", "worm_id"):
            if key in data.files:
                return scalar_text(data[key])
    return path.stem


def split_files(root: Path, fold: int, split: str) -> list[Path]:
    folder = root / f"fold_{fold}" / split
    paths = sorted(p for p in folder.iterdir() if p.suffix == ".npz")
    if not paths:
        raise RuntimeError(f"empty canonical split: {folder}")
    return paths


def freeze_manifest() -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], list[str]]]:
    rows: list[dict[str, Any]] = []
    identities: dict[tuple[str, int, str], list[str]] = {}
    for dataset, root in DATASETS.items():
        test_seen: set[str] = set()
        for fold in range(5):
            fold_sets: dict[str, set[str]] = {}
            for split in SPLITS:
                paths = split_files(root, fold, split)
                uids = [uid_for(path) for path in paths]
                if len(uids) != len(set(uids)):
                    raise RuntimeError(f"duplicate UID in {dataset} fold {fold} {split}")
                identities[(dataset, fold, split)] = uids
                fold_sets[split] = set(uids)
                for order, (uid, path) in enumerate(zip(uids, paths)):
                    rows.append(
                        {
                            "dataset": dataset,
                            "fold": fold,
                            "split": split,
                            "order": order,
                            "uid": uid,
                            "link_path": str(path.relative_to(REPO)),
                            "resolved_path": str(path.resolve()),
                        }
                    )
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                overlap = fold_sets[left] & fold_sets[right]
                if overlap:
                    raise RuntimeError(f"{dataset} fold {fold}: {left}/{right} overlap: {sorted(overlap)}")
            overlap = test_seen & fold_sets["test"]
            if overlap:
                raise RuntimeError(f"{dataset}: subjects repeated across test folds: {sorted(overlap)}")
            test_seen |= fold_sets["test"]
    return rows, identities


def same_path(left: str | Path | None, right: Path) -> bool:
    if left is None:
        return False
    return Path(left).resolve() == right.resolve()


@dataclass(frozen=True)
class Candidate:
    method: str
    dataset: str
    fold: int
    seed: str
    metrics_path: Path
    protocol_path: Path | None = None
    kind: str = "fold_root_metrics"


def candidates() -> list[Candidate]:
    found: list[Candidate] = []
    for dataset in DATASETS:
        for fold in range(5):
            found.extend(
                [
                    Candidate("CPD", dataset, fold, "deterministic", REPO / f"runs/unified_main_benchmark_cv5_seed42_v1/rerun/cpd/{dataset}/fold{fold}/metrics.json"),
                    Candidate("Ours", dataset, fold, "42", REPO / f"runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1/{dataset}/fold{fold}/seed42/atlas_identity_test/metrics.json", kind="dataset_root_metrics"),
                    Candidate("RGM", dataset, fold, "42", REPO / f"runs/unified_benchmark/rgm/{dataset}/fold{fold}/seed42/test_metrics.json", REPO / f"runs/unified_benchmark/rgm/{dataset}/fold{fold}/seed42/train.log", "rgm_log"),
                    Candidate("NGM-v2", dataset, fold, "42", REPO / f"runs/unified_main_benchmark_cv5_seed42_v1/rerun/ngmv2/{dataset}/fold{fold}/seed42/test_metrics.json", REPO / f"runs/unified_main_benchmark_cv5_seed42_v1/rerun/ngmv2/{dataset}/fold{fold}/seed42/LOCKED_BEFORE_TEST.json", "cv_root_metrics"),
                    Candidate("Vanilla FGW", dataset, fold, "deterministic", REPO / f"runs/unified_main_benchmark_cv5_seed42_v1/rerun/fgw_pot/{dataset}/fold{fold}/test_metrics.json", REPO / f"runs/unified_main_benchmark_cv5_seed42_v1/rerun/fgw_pot/{dataset}/fold{fold}/LOCKED_BEFORE_TEST.json", "cv_root_metrics"),
                    Candidate("StatAtlas", dataset, fold, "deterministic", REPO / f"runs/unified_main_benchmark_cv5_seed42_v1/rerun/statatlas/{dataset}/fold{fold}/test_metrics.json", kind="deterministic_cv_metrics"),
                    Candidate("CRF_ID", dataset, fold, "42", REPO / f"runs/unified_main_benchmark_cv5_seed42_v1/rerun/crf_id/{dataset}/fold{fold}/seed42/test_metrics.json", kind="crf_cv_metrics"),
                    Candidate("GWOT-MD", dataset, fold, "42", REPO / f"runs/unified_main_benchmark_cv5_seed42_v1/rerun/gwot_md_official/{dataset}/fold{fold}/seed42/test_metrics.json", kind="gwot_cv_metrics"),
                    Candidate("GWOT-MD (our adaptation)", dataset, fold, "42", REPO / f"runs/unified_main_benchmark_cv5_seed42_v1/rerun/gwot_md_adaptation/{dataset}/fold{fold}/seed42/test_metrics.json", kind="gwot_cv_metrics"),
                ]
            )
    for fold in range(5):
        found.extend(
            [
                Candidate("fDNC", "rld", fold, "42", REPO / f"runs/fdnc_current_grouped_cv_v2/rld/fold{fold}/seed42/outer_test_medoid_template_v1/metrics.json", REPO / f"runs/fdnc_current_grouped_cv_v2/rld/fold{fold}/seed42/locked_lists/split_manifest.json", "fdnc_manifest"),
                Candidate("NuCLR", "rld", fold, "42", REPO / f"benchmark_official/runs/nuclr_official_scratch50k_current_cv_seed42/rld/fold_{fold}/seed_42/outer_test_medoid_template_v1/result.json", REPO / f"benchmark_official/runs/nuclr_official_scratch50k_current_cv_seed42/rld/fold_{fold}/seed_42/current_cv_protocol.json", "nuclr_manifest"),
                Candidate("GeoTransformer", "rld", fold, "42", REPO / f"runs/rld_robustness_cv5_seed42_v2/results/geotransformer/fold{fold}/clean_reference/result.json", kind="geotransformer_grouped_checkpoint"),
                Candidate("fDNC", "atanas", fold, "42", REPO / f"runs/fdnc_current_grouped_cv_v2/atanas/fold{fold}/seed42/outer_test_medoid_template_v1/metrics.json", REPO / f"runs/fdnc_current_grouped_cv_v2/atanas/fold{fold}/seed42/locked_lists/split_manifest.json", "fdnc_manifest"),
                Candidate("NuCLR", "atanas", fold, "42", REPO / f"benchmark_official/runs/nuclr_official_scratch50k_current_cv_seed42/atanas/fold_{fold}/seed_42/outer_test_medoid_template_v1/result.json", REPO / f"benchmark_official/runs/nuclr_official_scratch50k_current_cv_seed42/atanas/fold_{fold}/seed_42/current_cv_protocol.json", "nuclr_manifest"),
                Candidate("GeoTransformer", "atanas", fold, "42", REPO / f"runs/atanas_geotransformer_current_grouped_seed42_v1/fold{fold}/clean_reference/result.json", kind="geotransformer_grouped_checkpoint"),
            ]
        )
    return found


def audit_candidate(candidate: Candidate, identities: dict[tuple[str, int, str], list[str]]) -> dict[str, Any]:
    expected_root = DATASETS[candidate.dataset] / f"fold_{candidate.fold}"
    base = {
        "dataset": candidate.dataset,
        "method": candidate.method,
        "fold": candidate.fold,
        "seed": candidate.seed,
        "metrics_path": str(candidate.metrics_path.relative_to(REPO)),
        "protocol_path": str(candidate.protocol_path.relative_to(REPO)) if candidate.protocol_path else "",
    }
    if not candidate.metrics_path.is_file():
        return {**base, "status": "MISSING", "reason": "metrics file absent"}
    metrics = read_json(candidate.metrics_path)
    if candidate.kind == "fold_root_metrics":
        passed = same_path(metrics.get("fold_root"), expected_root)
        return {**base, "status": "PASS" if passed else "FAIL", "reason": "exact canonical fold_root" if passed else "fold_root mismatch"}
    if candidate.kind == "dataset_root_metrics":
        passed = same_path(metrics.get("dataset_root"), expected_root)
        return {**base, "status": "PASS" if passed else "FAIL", "reason": "exact canonical dataset_root" if passed else "dataset_root mismatch"}
    if candidate.kind == "cv_root_metrics":
        expected_counts = {split: len(identities[(candidate.dataset, candidate.fold, split)]) for split in SPLITS}
        root_ok = same_path(metrics.get("data_root"), DATASETS[candidate.dataset])
        passed = root_ok and metrics.get("fold") == candidate.fold and metrics.get("split_counts") == expected_counts
        return {**base, "status": "PASS" if passed else "FAIL", "reason": "exact canonical CV root + fold + split counts" if passed else "CV root/fold/count mismatch"}
    if candidate.kind == "deterministic_cv_metrics":
        expected_counts = {split: len(identities[(candidate.dataset, candidate.fold, split)]) for split in SPLITS}
        root_ok = same_path(metrics.get("fold_root"), expected_root)
        seed_ok = metrics.get("model_seed") is None and metrics.get("deterministic") is True
        passed = root_ok and metrics.get("fold") == candidate.fold and metrics.get("split_counts") == expected_counts and seed_ok
        return {
            **base,
            "status": "PASS" if passed else "FAIL",
            "reason": (
                "exact canonical fold root + split counts; deterministic with no model seed"
                if passed
                else "canonical root/fold/count/determinism mismatch"
            ),
        }
    if candidate.kind == "crf_cv_metrics":
        expected_counts = {split: len(identities[(candidate.dataset, candidate.fold, split)]) for split in SPLITS}
        root_ok = same_path(metrics.get("fold_root"), expected_root)
        seed_ok = metrics.get("model_seed") == 42 and metrics.get("deterministic") is False
        protocol = metrics.get("training_protocol", {})
        leakage_ok = protocol.get("train_only_atlas") is True and protocol.get("test_labels_used_for_training") is False
        passed = (
            root_ok
            and metrics.get("fold") == candidate.fold
            and metrics.get("split_counts") == expected_counts
            and seed_ok
            and leakage_ok
        )
        return {
            **base,
            "status": "PASS" if passed else "FAIL",
            "reason": (
                "exact canonical fold root + split counts + seed42; train-only relational atlas"
                if passed
                else "canonical root/fold/count/seed/train-only-atlas mismatch"
            ),
        }
    if candidate.kind == "gwot_cv_metrics":
        expected_counts = {split: len(identities[(candidate.dataset, candidate.fold, split)]) for split in SPLITS}
        root_ok = same_path(metrics.get("fold_root"), expected_root)
        seed_ok = metrics.get("seed") == 42
        test_ok = metrics.get("evaluation_split") == "test"
        protocol = metrics.get("training_protocol", {})
        leakage_ok = protocol.get("test_used_for_selection") is False
        passed = (
            root_ok
            and metrics.get("fold") == candidate.fold
            and metrics.get("split_counts") == expected_counts
            and seed_ok
            and test_ok
            and leakage_ok
        )
        return {
            **base,
            "status": "PASS" if passed else "FAIL",
            "reason": (
                "exact canonical fold root + split counts + seed42; held-out test only"
                if passed
                else "canonical root/fold/count/seed/test-only mismatch"
            ),
        }
    if candidate.kind == "fdnc_manifest":
        if not candidate.protocol_path or not candidate.protocol_path.is_file():
            return {**base, "status": "FAIL", "reason": "locked split manifest absent"}
        protocol = read_json(candidate.protocol_path)
        matches = all(protocol.get(f"{split}_ids") == identities[(candidate.dataset, candidate.fold, split)] for split in SPLITS)
        root_ok = same_path(protocol.get("data_root"), expected_root)
        passed = matches and root_ok and protocol.get("seed") == 42
        return {**base, "status": "PASS" if passed else "FAIL", "reason": "exact train/val/test UID lists + root + seed42" if passed else "locked manifest mismatch"}
    if candidate.kind == "nuclr_manifest":
        if not candidate.protocol_path or not candidate.protocol_path.is_file():
            return {**base, "status": "FAIL", "reason": "current-CV protocol absent"}
        protocol = read_json(candidate.protocol_path)
        train_ok = protocol.get("train_ids") == identities[(candidate.dataset, candidate.fold, "train")]
        val_ok = protocol.get("val_ids") == identities[(candidate.dataset, candidate.fold, "val")]
        root_ok = same_path(protocol.get("cv_root"), DATASETS[candidate.dataset])
        result_root_ok = same_path(metrics.get("protocol", {}).get("cv_root"), DATASETS[candidate.dataset])
        passed = train_ok and val_ok and root_ok and result_root_ok and protocol.get("seed") == metrics.get("seed") == 42
        return {**base, "status": "PASS" if passed else "FAIL", "reason": "exact train/val lists; test is canonical remainder; root + seed42" if passed else "current-CV protocol mismatch"}
    if candidate.kind == "geotransformer_grouped_checkpoint":
        checkpoint = str(metrics.get("checkpoint", ""))
        token = f"{candidate.dataset}_current_grouped_fold{candidate.fold}_seed42"
        train_log = REPO.parent / f"geotransformer_official/current_grouped_selection/{candidate.dataset}/fold{candidate.fold}/seed42/train.log"
        expected_text = f'"dataset_root": "{expected_root.resolve()}"'
        log_ok = train_log.is_file() and expected_text in train_log.read_text(encoding="utf-8", errors="replace")
        query_uids = {
            item["query"].split("__", 2)[1]
            for item in metrics.get("test", {}).get("per_test_worm", [])
            if "__" in item.get("query", "")
        }
        test_ok = bool(query_uids) and query_uids <= set(identities[(candidate.dataset, candidate.fold, "test")])
        passed = metrics.get("seed") == 42 and metrics.get("robustness_fold") == candidate.fold and token in checkpoint and log_ok and test_ok
        return {**base, "status": "PASS" if passed else "FAIL", "reason": "training config has exact canonical fold root + seed42; evaluated query UIDs are canonical test subjects" if passed else "checkpoint/config/test UID mismatch"}
    if candidate.kind == "rgm_log":
        if not candidate.protocol_path or not candidate.protocol_path.is_file():
            return {**base, "status": "FAIL", "reason": "training log absent"}
        log = candidate.protocol_path.read_text(encoding="utf-8", errors="replace")
        counts = {split: len(identities[(candidate.dataset, candidate.fold, split)]) for split in SPLITS}
        root_ok = f"data root : {expected_root.resolve()}" in log
        seed_ok = "seed      : 42" in log and metrics.get("seed") == 42
        count_ok = f"PRE-TEST: train={counts['train']} val={counts['val']}" in log
        passed = root_ok and seed_ok and count_ok
        return {**base, "status": "PASS" if passed else "FAIL", "reason": "training log has exact canonical fold root, train/val counts, and seed42" if passed else "training-log provenance mismatch"}
    return {**base, "status": "REVIEW", "reason": "metrics exist but do not embed canonical root or exact split UID manifest"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows, identities = freeze_manifest()
    manifest_path = OUT / "canonical_fold_manifest.csv"
    fields = ("dataset", "fold", "split", "order", "uid", "link_path", "resolved_path")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    manifest_hash = sha256_bytes(manifest_path.read_bytes())
    (OUT / "canonical_fold_manifest.sha256").write_text(f"{manifest_hash}  {manifest_path.name}\n", encoding="utf-8")

    medoids: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for fold in range(5):
            path = REPO / f"runs/fair_identity_medoid_template_v1/cpd/{dataset}/fold{fold}/metrics.json"
            data = read_json(path)
            uid = data["template_selection"]["template_uid"]
            if uid not in identities[(dataset, fold, "train")]:
                raise RuntimeError(f"frozen medoid is not in canonical train split: {dataset} fold {fold} {uid}")
            medoids.append({"dataset": dataset, "fold": fold, "template_uid": uid, "source": str(path.relative_to(REPO))})
    write_json(
        OUT / "frozen_train_medoid_templates.json",
        {
            "scope": "canonical fair-identity single-template evaluator (CPD/fDNC/NuCLR)",
            "note": "Other single-specimen methods may apply their locked method-native preprocessing before outer-train-only medoid selection; no method may select from validation or test.",
            "templates": medoids,
        },
    )

    cpd_reproduction: list[dict[str, Any]] = []
    metric_keys = ("queries", "top1", "top5", "mrr", "hungarian_accuracy")
    for dataset in DATASETS:
        for fold in range(5):
            historical_path = REPO / f"runs/fair_identity_medoid_template_v1/cpd/{dataset}/fold{fold}/metrics.json"
            rerun_path = OUT / f"rerun/cpd/{dataset}/fold{fold}/metrics.json"
            historical = read_json(historical_path)
            rerun = read_json(rerun_path)
            old_values = historical["metrics"]["template_score"]
            new_values = rerun["metrics"]["template_score"]
            exact = all(old_values[key] == new_values[key] for key in metric_keys)
            cpd_reproduction.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "exact_metric_reproduction": exact,
                    "same_template_uid": historical["template_selection"]["template_uid"] == rerun["template_selection"]["template_uid"],
                    "historical_path": str(historical_path.relative_to(REPO)),
                    "rerun_path": str(rerun_path.relative_to(REPO)),
                    "metrics": {key: new_values[key] for key in metric_keys},
                }
            )
    if not all(row["exact_metric_reproduction"] and row["same_template_uid"] for row in cpd_reproduction):
        raise RuntimeError("CPD clean rerun did not exactly reproduce a prior grouped-fold cell")
    write_json(OUT / "cpd_reproduction_audit.json", cpd_reproduction)

    audits = [audit_candidate(item, identities) for item in candidates()]
    audit_fields = ("dataset", "method", "fold", "seed", "status", "reason", "metrics_path", "protocol_path")
    with (OUT / "artifact_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(sorted(audits, key=lambda row: (row["dataset"], row["method"], row["fold"])))

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in audits:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    readiness = []
    for (dataset, method), rows in sorted(groups.items()):
        statuses = [row["status"] for row in rows]
        ready = len(rows) == 5 and statuses == ["PASS"] * 5
        readiness.append({"dataset": dataset, "method": method, "fold_cells": len(rows), "statuses": statuses, "publishable": ready})
    write_json(OUT / "readiness.json", readiness)

    counts = {
        dataset: {
            str(fold): {split: len(identities[(dataset, fold, split)]) for split in SPLITS}
            for fold in range(5)
        }
        for dataset in DATASETS
    }
    protocol = {
        "protocol_id": "unified_main_benchmark_cv5_seed42_v1",
        "canonical_manifest": str(manifest_path.relative_to(REPO)),
        "canonical_manifest_sha256": manifest_hash,
        "fold_roots": {key: str(value.relative_to(REPO)) for key, value in DATASETS.items()},
        "fold_counts": counts,
        "learned_method_seed": 42,
        "deterministic_method_seed": None,
        "aggregation": "unweighted mean and sample SD over exactly five fold-level values",
        "publication_gate": "all five artifact audit cells must be PASS; REVIEW/MISSING/FAIL cells are excluded",
        "legacy_exclusion": "runs/fair_identity_medoid_template_v1/cv5x3_summary/cells.csv is forbidden as a formal-table source",
        "reference_rule": "single-specimen references are selected from outer train only; CPD/fDNC/NuCLR use the separately frozen canonical medoid UIDs, while methods with locked native preprocessing record their method-native train medoid",
        "method_specific": {
            "StatAtlas": "official geometry-only adaptation; deterministic/no model seed; statistical atlas fit on outer train only; validation and test labels excluded from atlas construction",
            "CRF_ID": "official MATLAB relational-atlas method; seed 42 fixes RNG; relational atlas fit on outer train only; validation may tune fixed hyperparameters; test labels are evaluation-only",
            "GWOT-MD": "seed 42 fixes stochastic initialization/teacher sampling; all fitting and hyperparameter selection use outer train/validation only; formal metrics are from held-out test subjects",
            "GWOT-MD (our adaptation)": "same canonical folds and seed 42; any Appendix-F teacher/hyperparameter selection is confined to outer train/validation; formal metrics are from held-out test subjects",
        },
    }
    write_json(OUT / "PROTOCOL.json", protocol)
    print(json.dumps({"output": str(OUT), "manifest_sha256": manifest_hash, "readiness": readiness}, indent=2))


if __name__ == "__main__":
    main()
