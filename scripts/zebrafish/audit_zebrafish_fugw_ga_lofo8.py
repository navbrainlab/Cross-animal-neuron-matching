#!/usr/bin/env python3
"""Fail-closed coupling replay and LOFO8 summary for official FUGW G+A."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from NeuRID_reproducibility.scripts.zebrafish.query_record_io import (  # noqa: E402
    QUERY_COLUMNS,
    candidate_order_sha256,
    records_from_score_matrix,
)
from scripts.zebrafish.evaluate_zebrafish_fugw_ga_fold import (  # noqa: E402
    EXPECTED_QUERIES,
    METHOD_KEY,
    METHOD_NAME,
    load_pairs,
    sha256,
    target_vectors,
)


DEFAULT_RUN = REPO / "runs/zebrafish_fugw_official_ga_lofo8"
DEFAULT_DATA = REPO / "Data/Zebrafish_MPRT_LOFO8_60m"
FISH = {
    1: "func_20150410", 2: "func_20150417", 3: "func_20161004",
    4: "islet_20170202", 5: "islet_20170216", 6: "lineage_20160328",
    7: "lineage_20170111", 8: "lineage_20170925",
}


def read_csv(path: Path, *, gzipped: bool = False) -> list[dict[str, str]]:
    opener = gzip.open if gzipped else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if gzipped and tuple(reader.fieldnames or ()) != QUERY_COLUMNS:
            raise RuntimeError(f"{path}: query schema mismatch")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_rows_equal(
    expected: list[dict[str, Any]], archived: list[dict[str, str]], context: str
) -> None:
    if len(expected) != len(archived):
        raise AssertionError(f"{context}: row count {len(archived)} != {len(expected)}")
    for row_index, (left, right) in enumerate(zip(expected, archived)):
        for column in QUERY_COLUMNS:
            left_value, right_value = str(left[column]), str(right[column])
            if column == "reciprocal_rank":
                if not math.isclose(float(left_value), float(right_value), abs_tol=1e-15):
                    raise AssertionError(f"{context}/{row_index}/{column}: {left_value} != {right_value}")
            elif left_value != right_value:
                raise AssertionError(f"{context}/{row_index}/{column}: {left_value!r} != {right_value!r}")


def metric(rows: list[dict[str, str]]) -> dict[str, float | int]:
    q = len(rows)
    return {
        "queries": q,
        "top1": sum(int(row["top1_correct"]) for row in rows) / q,
        "top5": sum(int(row["top5_correct"]) for row in rows) / q,
        "mrr": sum(float(row["reciprocal_rank"]) for row in rows) / q,
        "hungarian": sum(int(row["hungarian_correct"]) for row in rows) / q,
        "coverage": sum(int(row["covered"]) for row in rows) / q,
    }


def audit_fold(run_root: Path, data_root: Path, fold: int) -> dict[str, Any]:
    folder = run_root / f"fold_{fold}"
    lock_path = folder / "LOCKED_BEFORE_TEST.json"
    metrics_path = folder / "test_metrics.json"
    query_path = folder / "query_predictions.csv.gz"
    pair_path = folder / "pair_manifest.csv"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    archived = read_csv(query_path, gzipped=True)
    manifest = read_csv(pair_path)
    pairs = load_pairs(data_root / f"fold_{fold}", "test")

    assert lock["status"] == "LOCKED_BEFORE_TEST"
    assert lock["test_opened_before_lock"] is False
    assert lock["selection_split"] == "outer validation only"
    assert lock["fugw_version"] == "0.1.1"
    assert lock["official_source_sha256"] == sha256(Path(lock["official_source"]))
    assert report["method_key"] == METHOD_KEY
    assert report["lock_sha256"] == sha256(lock_path)
    assert report["query_predictions_sha256"] == sha256(query_path)
    assert report["pair_manifest_sha256"] == sha256(pair_path)
    assert len(pairs) == len(manifest)

    cursor = 0
    recomputed_all: list[dict[str, Any]] = []
    for pair_index, (pair, manifest_row) in enumerate(zip(pairs, manifest)):
        assert int(manifest_row["pair_index"]) == pair_index
        assert manifest_row["pair_id"] == pair["pair_id"]
        coupling_path = Path(manifest_row["coupling"])
        assert sha256(coupling_path) == manifest_row["coupling_sha256"]
        with np.load(coupling_path, allow_pickle=False) as data:
            plan = np.asarray(data["plan"], dtype=np.float64)
            assert str(np.asarray(data["pair_id"]).reshape(-1)[0]) == pair["pair_id"]
            assert np.asarray(data["q_ids"]).astype(str).tolist() == pair["q"].ids.astype(str).tolist()
            assert np.asarray(data["r_ids"]).astype(str).tolist() == pair["r"].ids.astype(str).tolist()
        q, r = pair["q"], pair["r"]
        assert plan.shape == (len(q.ids), len(r.ids))
        qtargets, rtargets = target_vectors(q, r)
        recomputed = records_from_score_matrix(
            method=METHOD_KEY, fold=fold, seed=None, pair_index=pair_index,
            pair_id=pair["pair_id"], score=plan, q_uid=q.uid, r_uid=r.uid,
            q_ids=q.ids, r_ids=r.ids, row_target=qtargets, col_target=rtargets,
        )
        block = archived[cursor:cursor + len(recomputed)]
        assert_rows_equal(recomputed, block, f"fold{fold}/pair{pair_index}")
        cursor += len(recomputed)
        recomputed_all.extend(recomputed)
        assert int(manifest_row["q_candidates"]) == len(q.ids)
        assert int(manifest_row["r_candidates"]) == len(r.ids)
        qr = [row for row in block if row["direction"] == "q_to_r"]
        rq = [row for row in block if row["direction"] == "r_to_q"]
        assert len(qr) == len(rq)
        assert all(row["candidate_order_sha256"] == candidate_order_sha256(r.ids) for row in qr)
        assert all(row["candidate_order_sha256"] == candidate_order_sha256(q.ids) for row in rq)
    assert cursor == len(archived)
    assert_rows_equal(recomputed_all, archived, f"fold{fold}/all")

    values = metric(archived)
    assert values["queries"] == EXPECTED_QUERIES[fold]
    assert values["coverage"] == 1.0
    for name in ("top1", "top5", "mrr", "hungarian", "coverage"):
        assert math.isclose(float(values[name]), float(report["test"][name]), abs_tol=1e-12)
    return {
        "method": METHOD_KEY, "display_name": METHOD_NAME, "fold": fold,
        "test_fish": FISH[fold], "seed": "", "pairs": len(pairs),
        **values, **lock["selected"], "status": "PASS",
    }


def format_percent(value: dict[str, float]) -> str:
    return f"{100*value['mean']:.2f} ± {100*value['sd']:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--publish-query-root", type=Path, default=None,
        help="Optionally publish fold_<k>.csv.gz under <root>/fugw_ga_official/.",
    )
    args = parser.parse_args()
    run_root, data_root = args.run_root.resolve(), args.data_root.resolve()
    folds = [audit_fold(run_root, data_root, fold) for fold in range(1, 9)]
    summary = {
        name: {
            "mean": statistics.mean(float(row[name]) for row in folds),
            "sd": statistics.stdev(float(row[name]) for row in folds),
        }
        for name in ("top1", "top5", "mrr", "hungarian")
    }
    aggregate = {
        "status": "PASS", "method": METHOD_NAME,
        "implementation": "official fugw==0.1.1; dense fugw.mappings.FUGW",
        "protocol": "Zebrafish LOFO8 paired q/r; validation-only selection; canonical scoring",
        "folds": folds, "summary": summary,
        "aggregation": "unweighted mean and sample SD (ddof=1) over eight held-out-fish folds",
        "runner": str((REPO / "scripts/zebrafish/evaluate_zebrafish_fugw_ga_fold.py").resolve()),
        "runner_sha256": sha256(REPO / "scripts/zebrafish/evaluate_zebrafish_fugw_ga_fold.py"),
        "auditor": str(Path(__file__).resolve()),
        "auditor_sha256": sha256(Path(__file__)),
    }
    write_json_path = run_root / "AUDIT_AND_SUMMARY.json"
    write_json_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    write_csv(run_root / "fold_results.csv", folds)
    summary_row: dict[str, Any] = {"method": METHOD_KEY, "display_name": METHOD_NAME, "folds": 8}
    for name, value in summary.items():
        summary_row[f"{name}_mean"] = value["mean"]
        summary_row[f"{name}_sd"] = value["sd"]
    write_csv(run_root / "summary.csv", [summary_row])
    lines = [
        "# Official FUGW (G+A) — Zebrafish LOFO8", "",
        "Official dense solver: `fugw==0.1.1`. Hyperparameters are selected on each outer validation fold before test access. XYZ supplies the linear geometry term; archived 128-point activity-trace distances supply the GW term.",
        "", "| Method | Top-1 | Top-5 | MRR | Hungarian |", "|---|---:|---:|---:|---:|",
        f"| {METHOD_NAME} | {format_percent(summary['top1'])} | {format_percent(summary['top5'])} | "
        f"{summary['mrr']['mean']:.4f} ± {summary['mrr']['sd']:.4f} | {format_percent(summary['hungarian'])} |",
        "", "Values are the unweighted mean ± sample SD (`ddof=1`) over the eight held-out fish. Every saved coupling was replayed against the canonical query records; the exact per-fold query audit passed.",
    ]
    (run_root / "FORMAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.publish_query_root is not None:
        destination = args.publish_query_root.resolve() / METHOD_KEY
        destination.mkdir(parents=True, exist_ok=True)
        for fold in range(1, 9):
            shutil.copyfile(
                run_root / f"fold_{fold}" / "query_predictions.csv.gz",
                destination / f"fold_{fold}.csv.gz",
            )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
