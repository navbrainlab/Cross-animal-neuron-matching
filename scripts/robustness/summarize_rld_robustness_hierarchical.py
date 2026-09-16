#!/usr/bin/env python3
"""One canonical-query estimand and hierarchical bootstrap for every RLD method.

The bootstrap resamples biological folds, then worms within each sampled fold,
then shared corruption-draw IDs.  Identical resampling indices are used for all
methods, which permits paired method contrasts once every result root is present.

The denominator is the canonical main-table query cohort, not the subset whose
identity happens to occur in a method's reference.  Coordinate noise and
distractors retain the clean canonical cohort.  Missing-neuron conditions retain
the canonical rows whose original source indices survive the shared deletion
mask.  A method-reference miss is therefore retained and scores zero.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs/rld_robustness_cv5_seed42_v2"
DEFAULT_CANONICAL = (
    ROOT / "runs/unified_main_benchmark_cv5_seed42_v2/canonical_query_manifest.csv"
)
DEFAULT_MAIN_FOLDS = (
    ROOT / "runs/unified_main_benchmark_cv5_seed42_v2/unified_fold_metrics.csv"
)
COND_RE = re.compile(r"^(coord_noise|missing|outlier)_l([0-9]+(?:\.[0-9]+)?)_p([0-9]+)$")
METRICS = ("top1", "top5", "mrr", "hungarian", "coverage", "effective_top1", "retention")
MAIN_METHODS = {
    "cpd": "CPD",
    "fdnc": "fDNC",
    "ours": "NeurID (population atlas)",
    "nuclr": "NuCLR",
    "geo": "GeoTransformer",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def condition(name: str):
    m = COND_RE.match(name)
    if not m:
        raise ValueError(name)
    return m.group(1), float(m.group(2)), int(m.group(3))


def uid_from_geo(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("__")
    if len(parts) < 2:
        raise ValueError(name)
    return parts[1]


def aggregate_query_rows(rows: list[dict], adapter: str, allowed_rows: dict[str, set[int]]):
    out = defaultdict(lambda: np.zeros(5, dtype=float))
    for row in rows:
        if adapter in {"cpd", "fdnc"}:
            uid = row["query_uid"]
            node_index = int(row["query_index"])
            values = (1, float(row["top1"]), float(row["top5"]),
                      float(row["rr"]), 0.0)
        elif adapter == "ours":
            uid = row["uid"]
            node_index = int(row["node_index"])
            rank = int(row["rank"])
            values = (1, float(row["correct"]), float(rank <= 5),
                      1.0 / rank, float(row["hungarian_correct"]))
        elif adapter == "geo":
            uid = row["query_uid"]
            node_index = int(row["query_index"])
            values = (1, float(row["top1"]), float(row["top5"]),
                      float(row["rr"]), float(row["hungarian_correct"]))
        elif adapter == "nuclr":
            uid = row["query_worm"]
            node_index = int(row["query_row"])
            values = (1, float(row["top1"]), float(row["top5"]),
                      float(row["rr"]), float(row["hungarian_top1"]))
        else:
            raise ValueError(adapter)
        if uid not in allowed_rows or node_index not in allowed_rows[uid]:
            raise RuntimeError(
                f"{adapter} emitted non-canonical query row: "
                f"uid={uid}, current_node_index={node_index}"
            )
        out[uid] += np.asarray(values)
    return dict(out)


def load_cell(
    method: str,
    root: Path,
    fold: int,
    name: str,
    allowed_rows: dict[str, set[int]],
):
    base = root / f"fold{fold}" / name
    if method == "cpd":
        report_root = base / "original_runner"
        out = aggregate_query_rows(
            read_csv(report_root / "per_query.csv"), method, allowed_rows
        )
        report = json.loads((report_root / "metrics.json").read_text(encoding="utf-8"))
        for row in report["per_animal"]:
            out[row["query_uid"]][4] = float(row["hungarian_accuracy"]) * int(row["queries"])
        return out
    if method == "fdnc":
        out = aggregate_query_rows(read_csv(base / "per_query.csv"), method, allowed_rows)
        report = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
        for row in report["per_animal"]:
            out[row["query_uid"]][4] = float(row["hungarian_accuracy"]) * int(row["queries"])
        return out
    if method == "ours":
        return aggregate_query_rows(read_csv(base / "queries.csv"), method, allowed_rows)
    if method == "nuclr":
        return aggregate_query_rows(read_csv(base / "query_level.csv"), method, allowed_rows)
    if method == "geo":
        query_level = base / "query_level.csv"
        if query_level.is_file():
            return aggregate_query_rows(read_csv(query_level), method, allowed_rows)
        report = json.loads((base / "result.json").read_text(encoding="utf-8"))
        out = {}
        for row in report["test"]["per_test_worm"]:
            uid = uid_from_geo(row["query"])
            if uid not in allowed_rows:
                raise RuntimeError(f"geo emitted non-canonical test worm: uid={uid}")
            if int(row["queries"]) > len(allowed_rows[uid]):
                raise RuntimeError(
                    f"geo emitted {row['queries']} queries for {uid}, but the "
                    f"canonical surviving cohort has only {len(allowed_rows[uid])}"
                )
            out[uid] = np.asarray([
                row["queries"], row["top1_correct"], row["top5_correct"],
                row["rr_sum"], row["hungarian_correct"],
            ], dtype=float)
        return out
    raise ValueError(method)


def canonical_queries(path: Path):
    queries = defaultdict(lambda: defaultdict(set))
    for row in read_csv(path):
        if row["dataset"] != "rld":
            continue
        queries[int(row["fold"])][row["uid"]].add(
            (int(row["node_index"]), row["identity"])
        )
    if sorted(queries) != list(range(5)):
        raise RuntimeError(f"canonical RLD query manifest is incomplete: {path}")
    return queries


def parse_manifest(path: Path, canonical_path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    canonical = canonical_queries(canonical_path)
    rows = []
    worms = defaultdict(set)
    common_queries = {}
    allowed_rows = {}
    for row in report["conditions"]:
        fold = int(row["fold"])
        name = Path(row["root"]).name
        if COND_RE.match(name) is None:
            # Activity noise has a separate modality-aware comparison because
            # geometry-only methods are invariant/not applicable there.
            continue
        kind, severity, draw = condition(name)
        rows.append((fold, name, kind, severity, draw))
        common_queries[(fold, kind, severity, draw)] = {}
        allowed_rows[(fold, kind, severity, draw)] = {}
        for rec in row["files"]:
            uid = str(rec["recording_uid"])
            worms[fold].add(uid)
            kept = [int(x) for x in rec["kept_source_indices"]]
            source_to_current = {source: current for current, source in enumerate(kept)}
            observed = {
                int(item["row"]): str(item["cell_id"])
                for item in rec["evaluable_query_rows_after_corruption"]
            }
            surviving = set()
            for source_index, identity in canonical[fold].get(uid, set()):
                current_index = source_to_current.get(source_index)
                if current_index is None:
                    continue
                if observed.get(current_index) != identity:
                    raise RuntimeError(
                        "canonical query does not survive with the expected identity: "
                        f"fold={fold}, condition={name}, uid={uid}, "
                        f"source_index={source_index}, current_index={current_index}, "
                        f"expected={identity!r}, observed={observed.get(current_index)!r}"
                    )
                surviving.add(current_index)
            common_queries[(fold, kind, severity, draw)][uid] = len(surviving)
            allowed_rows[(fold, kind, severity, draw)][uid] = surviving

    manifest_worms = {k: sorted(v) for k, v in worms.items()}
    for fold in sorted(canonical):
        missing_worms = set(canonical[fold]) - set(manifest_worms.get(fold, []))
        if missing_worms:
            raise RuntimeError(
                f"fold{fold} canonical query worms absent from robustness manifest: "
                f"{sorted(missing_worms)}"
            )
    return sorted(rows), manifest_worms, common_queries, allowed_rows


def ratios(counts: np.ndarray, current_q: np.ndarray, clean_q: np.ndarray):
    reported_q, top1, top5, rr, hung = counts.sum(axis=0)
    q, cq = current_q.sum(), clean_q.sum()
    if reported_q > q + 1e-9:
        raise RuntimeError(f"method emitted {reported_q} queries outside common cohort of {q}")
    if q <= 0 or cq <= 0:
        return np.full(6, np.nan)
    return np.asarray([top1/q, top5/q, rr/q, hung/q, q/cq, top1/cq])


def point_for_method(cells, common_queries, worms, folds, kind, severity):
    fold_values = [
        point_for_fold(cells, common_queries, worms, fold, kind, severity)[0]
        for fold in folds
    ]
    value = np.nanmean(fold_values, axis=0)
    clean_eff = point_for_clean(cells, common_queries, worms, folds)
    retention = value[5] / clean_eff if clean_eff > 0 else float("nan")
    return np.r_[value, retention]


def point_for_fold(cells, common_queries, worms, fold, kind, severity):
    uids = worms[fold]
    clean_q = np.asarray([
        common_queries[(fold, "coord_noise", 0.0, 0)][uid] for uid in uids
    ])
    draws = sorted(k[3] for k in cells if k[:3] == (fold, kind, severity))
    draw_values = []
    draw_queries = []
    draw_correct = []
    for draw in draws:
        cell = cells[(fold, kind, severity, draw)]
        counts = np.stack([cell.get(uid, np.zeros(5)) for uid in uids])
        current_q = np.asarray([
            common_queries[(fold, kind, severity, draw)][uid] for uid in uids
        ])
        draw_values.append(ratios(counts, current_q, clean_q))
        draw_queries.append(float(current_q.sum()))
        draw_correct.append(float(counts[:, 1].sum()))
    value = np.nanmean(draw_values, axis=0)
    clean_effective = point_for_clean(cells, common_queries, worms, [fold])
    return (
        np.r_[value, value[5] / clean_effective],
        float(np.mean(draw_queries)),
        float(np.mean(draw_correct)),
        int(clean_q.sum()),
        len(draws),
    )


def point_for_clean(cells, common_queries, worms, folds):
    values = []
    for fold in folds:
        uids = worms[fold]
        clean = cells[(fold, "coord_noise", 0.0, 0)]
        counts = np.stack([clean.get(uid, np.zeros(5)) for uid in uids])
        clean_q = np.asarray([
            common_queries[(fold, "coord_noise", 0.0, 0)][uid] for uid in uids
        ])
        values.append(ratios(counts, clean_q, clean_q)[5])
    return float(np.nanmean(values))


def bootstrap_once(cells, common_queries, worms, folds, kind, severity, rng):
    sampled_folds = rng.choice(folds, size=len(folds), replace=True)
    fold_values, clean_values = [], []
    for fold in sampled_folds:
        uids = worms[int(fold)]
        picked = rng.integers(0, len(uids), size=len(uids))
        clean = cells[(int(fold), "coord_noise", 0.0, 0)]
        clean_counts = np.stack([clean.get(uids[i], np.zeros(5)) for i in picked])
        clean_q = np.asarray([
            common_queries[(int(fold), "coord_noise", 0.0, 0)][uids[i]] for i in picked
        ])
        clean_values.append(ratios(clean_counts, clean_q, clean_q)[5])
        draws = sorted(k[3] for k in cells if k[:3] == (int(fold), kind, severity))
        sampled_draws = rng.choice(draws, size=len(draws), replace=True)
        draw_values = []
        for draw in sampled_draws:
            cell = cells[(int(fold), kind, severity, int(draw))]
            counts = np.stack([cell.get(uids[i], np.zeros(5)) for i in picked])
            current_q = np.asarray([
                common_queries[(int(fold), kind, severity, int(draw))][uids[i]]
                for i in picked
            ])
            draw_values.append(ratios(counts, current_q, clean_q))
        fold_values.append(np.nanmean(draw_values, axis=0))
    value = np.nanmean(fold_values, axis=0)
    clean_eff = float(np.nanmean(clean_values))
    return np.r_[value, value[5] / clean_eff]


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main_table_rows(path: Path):
    out = {}
    for row in read_csv(path):
        if row["dataset"] != "rld" or not row["top1"]:
            continue
        out[(row["method"], int(row["fold"]))] = row
    return out


def clean_gate(method: str, fold_rows: list[dict], main_rows: dict, path: Path):
    method_name = MAIN_METHODS[method]
    failures = []
    for row in fold_rows:
        if row["kind"] != "coord_noise" or float(row["severity"]) != 0.0:
            continue
        expected = main_rows.get((method_name, int(row["fold"])))
        if expected is None:
            failures.append({"fold": row["fold"], "reason": "missing main-table fold"})
            continue
        q = int(expected["canonical_queries"])
        correct = round(float(expected["top1"]) * q)
        observed_q = int(round(float(row["queries"])))
        observed_correct = int(round(float(row["top1_correct"])))
        differences = {}
        if observed_q != q:
            differences["queries"] = {"main": q, "robustness": observed_q}
        if observed_correct != correct:
            differences["top1_correct"] = {
                "main": correct, "robustness": observed_correct
            }
        for metric, main_key in (
            ("top1", "top1"), ("top5", "top5"), ("hungarian", "hungarian")
        ):
            if abs(float(row[metric]) - float(expected[main_key])) > 1e-12:
                differences[metric] = {
                    "main": float(expected[main_key]),
                    "robustness": float(row[metric]),
                }
        if differences:
            failures.append({"fold": row["fold"], "differences": differences})
    if failures:
        raise RuntimeError(
            f"{method} severity-zero canonical gate failed against {path}: {failures}"
        )
    print(f"[CANONICAL CLEAN GATE PASSED] {method}: 5/5 folds", flush=True)


def macro_rows(fold_rows: list[dict]):
    output = []
    metrics = ("top1", "top5", "mrr", "hungarian", "coverage", "effective_top1", "retention")
    groups = sorted({(row["method"], row["kind"], row["severity"]) for row in fold_rows})
    for method, kind, severity in groups:
        part = [
            row for row in fold_rows
            if (row["method"], row["kind"], row["severity"])
            == (method, kind, severity)
        ]
        item = {
            "method": method,
            "kind": kind,
            "severity": severity,
            "folds": len(part),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in part])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--methods", default="cpd,fdnc,ours,nuclr,geo")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--canonical-query-manifest", type=Path, default=DEFAULT_CANONICAL)
    ap.add_argument("--main-fold-metrics", type=Path, default=DEFAULT_MAIN_FOLDS)
    args = ap.parse_args()
    method_roots = {
        "cpd": args.run_root / "results/cpd",
        "fdnc": args.run_root / "results/fdnc",
        "ours": args.run_root / "results/ours_static_main_reference_seed42",
        "nuclr": args.run_root / "results/nuclr",
        "geo": args.run_root / "results/geotransformer",
    }
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    manifest_rows, worms, common_queries, allowed_rows = parse_manifest(
        args.run_root / "corruptions/MANIFEST.json",
        args.canonical_query_manifest,
    )
    expected_main = main_table_rows(args.main_fold_metrics)
    folds = sorted(worms)
    expected = len(manifest_rows)
    summaries = []
    all_fold_rows = []
    sample_cache = {}
    estimate_cache = {}
    for method in methods:
        cells = {}
        missing = []
        for fold, name, kind, severity, draw in manifest_rows:
            try:
                cells[(fold, kind, severity, draw)] = load_cell(
                    method,
                    method_roots[method],
                    fold,
                    name,
                    allowed_rows[(fold, kind, severity, draw)],
                )
            except FileNotFoundError as exc:
                missing.append(str(exc.filename or exc))
        if missing:
            message = f"{method}: {len(missing)}/{expected} condition outputs missing"
            if not args.allow_incomplete:
                raise RuntimeError(message)
            print("[SKIP]", message)
            continue
        grid = sorted({(kind, severity) for _, _, kind, severity, _ in manifest_rows})
        method_fold_rows = []
        clean_top1_by_fold = {}
        for fold in folds:
            clean_value, _, _, _, _ = point_for_fold(
                cells, common_queries, worms, fold, "coord_noise", 0.0
            )
            clean_top1_by_fold[fold] = float(clean_value[0])
        for kind, severity in grid:
            for fold in folds:
                value, queries, correct, clean_queries, draw_count = point_for_fold(
                    cells, common_queries, worms, fold, kind, severity
                )
                item = {
                    "method": method,
                    "kind": kind,
                    "severity": severity,
                    "fold": fold,
                    "corruption_draws": draw_count,
                    "queries": queries,
                    "clean_queries": clean_queries,
                    "top1_correct": correct,
                }
                for index, metric in enumerate(METRICS[:-1]):
                    item[metric] = float(value[index])
                item["retention"] = (
                    float(value[5]) / clean_top1_by_fold[fold]
                    if clean_top1_by_fold[fold] > 0 else float("nan")
                )
                method_fold_rows.append(item)
        clean_gate(method, method_fold_rows, expected_main, args.main_fold_metrics)
        all_fold_rows.extend(method_fold_rows)
        rng = np.random.default_rng(args.seed)
        for kind, severity in grid:
            estimate = point_for_method(cells, common_queries, worms, folds, kind, severity)
            samples = np.stack([
                bootstrap_once(cells, common_queries, worms, folds, kind, severity, rng)
                for _ in range(args.bootstrap)
            ])
            sample_cache[(method, kind, severity)] = samples
            estimate_cache[(method, kind, severity)] = estimate
            row = {"method": method, "kind": kind, "severity": severity,
                   "folds": len(folds), "bootstrap_replicates": args.bootstrap}
            for j, metric in enumerate(METRICS):
                valid = samples[:, j][np.isfinite(samples[:, j])]
                row[f"{metric}_mean"] = float(estimate[j])
                row[f"{metric}_ci_low"] = float(np.quantile(valid, 0.025))
                row[f"{metric}_ci_high"] = float(np.quantile(valid, 0.975))
            summaries.append(row)
    if not summaries:
        raise RuntimeError("no complete method outputs")
    write_csv(args.run_root / "canonical_fold_level.csv", all_fold_rows)
    write_csv(args.run_root / "canonical_macro_summary.csv", macro_rows(all_fold_rows))
    out = args.run_root / "hierarchical_bootstrap_summary.csv"
    write_csv(out, summaries)
    pairwise = []
    completed_methods = sorted({row["method"] for row in summaries})
    grid = sorted({(row["kind"], row["severity"]) for row in summaries})
    for method_a, method_b in combinations(completed_methods, 2):
        for kind, severity in grid:
            key_a = (method_a, kind, severity)
            key_b = (method_b, kind, severity)
            if key_a not in sample_cache or key_b not in sample_cache:
                continue
            diffs = sample_cache[key_a] - sample_cache[key_b]
            estimate = estimate_cache[key_a] - estimate_cache[key_b]
            row = {"method_a": method_a, "method_b": method_b, "kind": kind,
                   "severity": severity, "contrast": "method_a_minus_method_b"}
            for j, metric in enumerate(METRICS):
                valid = diffs[:, j][np.isfinite(diffs[:, j])]
                row[f"{metric}_difference"] = float(estimate[j])
                row[f"{metric}_ci_low"] = float(np.quantile(valid, 0.025))
                row[f"{metric}_ci_high"] = float(np.quantile(valid, 0.975))
            pairwise.append(row)
    if pairwise:
        write_csv(args.run_root / "hierarchical_bootstrap_pairwise.csv", pairwise)
    meta = {
        "estimand": "unweighted draw mean within fold; unweighted biological-fold mean",
        "query_cohort": (
            "canonical main-table query rows; coordinate noise and distractors keep "
            "the clean cohort; missing-neuron conditions keep canonical rows whose "
            "source indices survive; method-unrepresentable labels score zero"
        ),
        "canonical_query_manifest": str(args.canonical_query_manifest.resolve()),
        "severity_zero_gate": str(args.main_fold_metrics.resolve()),
        "bootstrap": "resample folds, worms within fold, and shared corruption draw IDs; paired indices across methods",
        "replicates": args.bootstrap,
        "seed": args.seed,
        "methods": completed_methods,
        "metrics": list(METRICS),
    }
    (args.run_root / "hierarchical_bootstrap_summary.json").write_text(
        json.dumps({"metadata": meta, "rows": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
