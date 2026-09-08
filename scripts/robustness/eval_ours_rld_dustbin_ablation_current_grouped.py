#!/usr/bin/env python3
"""Evaluate unknown rejection and an inference-time no-dustbin ablation on RLD.

The script reuses the exact grouped-CV5, seed-42 checkpoints and the materialized
held-out distractor conditions from the formal robustness benchmark.  Synthetic
``__OUTLIER_*`` nodes are unknown positives; atlas-eligible labeled test nodes
are negatives.  Native unlabeled neurons are population context but are not
used as binary unknown targets.

``capacity_dustbin`` is the production transport. ``ordinary`` keeps the model,
encodings, relation transport, atlas, and checkpoint fixed, but solves transport
without a dustbin.  It is therefore an inference-time mechanism ablation, not a
separately retrained model.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "mprt_net_v1_1"
RUNS = ROOT / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1/rld"
CORR = ROOT / "runs/rld_robustness_cv5_seed42_v2/corruptions"
OUT = ROOT / "runs/rld_robustness_cv5_seed42_v2/formal_dustbin_ablation_ours"
MODES = ("capacity_dustbin", "ordinary")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint(fold: int) -> Path:
    path = RUNS / f"fold{fold}/seed42/dynamic/low_rank_r8/best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def saved_clean(fold: int) -> dict[str, Any]:
    path = RUNS / f"fold{fold}/seed42/atlas_identity_test/metrics.json"
    return read_json(path)["modes"]["static"]


def conditions(manifest: dict[str, Any], fold: int) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in manifest["conditions"]
        if int(row["fold"]) == fold and str(row["kind"]) == "outlier"
    ]
    return sorted(rows, key=lambda row: (
        float(row["severity"]), int(row["perturbation_seed"])
    ))


def identity_targets(sample: Any, identity_to_slot: dict[str, int]) -> torch.Tensor:
    from mprt_net.data import unique_identity_map

    target = torch.full(
        (sample.num_nodes,), -1, dtype=torch.long, device=sample.xyz.device
    )
    for identity, node_index in unique_identity_map(sample).items():
        slot = identity_to_slot.get(str(identity))
        if slot is not None:
            target[int(node_index)] = int(slot)
    return target


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = labels == 1
    negative = ~positive
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    n_pos = int(labels.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / n_pos)


def summarize_queries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if row["target_type"] == "known"]
    unknown = [row for row in rows if row["target_type"] == "synthetic_unknown"]
    binary = known + unknown
    labels = np.asarray(
        [int(row["target_type"] == "synthetic_unknown") for row in binary],
        dtype=np.int64,
    )
    scores = np.asarray([float(row["dustbin_score"]) for row in binary])
    tp = sum(int(row["rejected"]) for row in unknown)
    fp = sum(int(row["rejected"]) for row in known)
    fn = len(unknown) - tp
    has_dustbin = bool(rows) and rows[0]["mode"] == "capacity_dustbin"
    precision = (
        tp / (tp + fp) if tp + fp else 0.0
    ) if unknown and has_dustbin else None
    recall = tp / len(unknown) if unknown else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else (0.0 if unknown else None)
    )
    return {
        "known_queries": len(known),
        "synthetic_unknown_queries": len(unknown),
        "known_top1_real": (
            sum(int(row["correct_real"]) for row in known) / len(known)
            if known else None
        ),
        "known_top1_with_dustbin": (
            sum(int(row["correct_with_dustbin"]) for row in known) / len(known)
            if known else None
        ),
        "known_false_reject_rate": fp / len(known) if known else None,
        "unknown_recall": recall,
        "forced_match_rate": 1.0 - recall if recall is not None else None,
        "dustbin_precision": precision,
        "dustbin_f1": f1 if has_dustbin else None,
        "unknown_auroc": roc_auc(labels, scores) if unknown and has_dustbin else None,
        "unknown_auprc": (
            average_precision(labels, scores) if unknown and has_dustbin else None
        ),
    }


def evaluate_condition(
    *,
    model: Any,
    identity_to_slot: dict[str, int],
    cache: Any,
    paths: list[Path],
    fold: int,
    severity: float,
    perturbation_seed: int,
    modes: tuple[str, ...],
    out_dir: Path,
) -> list[dict[str, Any]]:
    from mprt_net.experiments.dustbin_robustness import match_with_solver

    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    atlas = model.atlas_encoding()
    with torch.inference_mode():
        for path in paths:
            sample_cpu = cache.get(path)
            sample = sample_cpu.to(atlas.nodes.device)
            query = model.encode_population(sample)
            target = identity_targets(sample, identity_to_slot)
            synthetic = torch.tensor(
                [identity.startswith("__OUTLIER_") for identity in sample.cell_ids],
                dtype=torch.bool,
                device=target.device,
            )
            if bool(((target >= 0) & synthetic).any()):
                raise RuntimeError(f"synthetic distractor became a known target: {path}")
            expected_synthetic = int(round(severity * (sample.num_nodes - int(synthetic.sum()))))
            if severity > 0 and int(synthetic.sum()) == 0:
                raise RuntimeError(f"no synthetic distractors found in {path}")

            for mode in modes:
                output = match_with_solver(model, query, atlas, mode)
                probabilities = output.row
                real = probabilities[:, :-1]
                real_prediction = real.argmax(dim=1)
                prediction = probabilities.argmax(dim=1)
                dustbin_score = probabilities[:, -1]
                known_indices = torch.nonzero(target >= 0, as_tuple=False).flatten()
                unknown_indices = torch.nonzero(synthetic, as_tuple=False).flatten()

                for node_index in known_indices.detach().cpu().tolist():
                    slot = int(target[node_index])
                    by_mode[mode].append({
                        "fold": fold, "severity": severity,
                        "perturbation_seed": perturbation_seed,
                        "mode": mode, "uid": sample_cpu.uid,
                        "node_index": node_index, "target_type": "known",
                        "target_slot": slot,
                        "prediction_slot": int(real_prediction[node_index]),
                        "dustbin_score": float(dustbin_score[node_index]),
                        "rejected": int(prediction[node_index] == real.shape[1]),
                        "correct_real": int(real_prediction[node_index] == slot),
                        "correct_with_dustbin": int(prediction[node_index] == slot),
                    })
                for node_index in unknown_indices.detach().cpu().tolist():
                    by_mode[mode].append({
                        "fold": fold, "severity": severity,
                        "perturbation_seed": perturbation_seed,
                        "mode": mode, "uid": sample_cpu.uid,
                        "node_index": node_index,
                        "target_type": "synthetic_unknown", "target_slot": -1,
                        "prediction_slot": int(real_prediction[node_index]),
                        "dustbin_score": float(dustbin_score[node_index]),
                        "rejected": int(prediction[node_index] == real.shape[1]),
                        "correct_real": "", "correct_with_dustbin": "",
                    })
            if severity > 0 and abs(int(synthetic.sum()) - expected_synthetic) > 1:
                raise RuntimeError(
                    f"unexpected distractor count in {path}: "
                    f"observed={int(synthetic.sum())}, expected~={expected_synthetic}"
                )

    results = []
    for mode, query_rows in by_mode.items():
        metrics = {
            "fold": fold, "severity": severity,
            "perturbation_seed": perturbation_seed, "mode": mode,
            "recordings": len(paths), **summarize_queries(query_rows),
        }
        results.append(metrics)
        mode_dir = out_dir / mode
        write_json(mode_dir / "metrics.json", metrics)
        mode_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(mode_dir / "queries.csv.gz", "wt", newline="", encoding="utf-8") as handle:
            if query_rows:
                writer = csv.DictWriter(handle, fieldnames=list(query_rows[0]))
                writer.writeheader()
                writer.writerows(query_rows)
    return results


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def aggregate(cells: list[dict[str, Any]], out_root: Path) -> None:
    metric_names = (
        "known_top1_real", "known_top1_with_dustbin",
        "known_false_reject_rate", "unknown_recall", "forced_match_rate",
        "dustbin_precision", "dustbin_f1", "unknown_auroc", "unknown_auprc",
    )
    # Older completed condition JSONs can be re-summarized without inference.
    # Normalize undefined detection metrics fail-closed: clean has no positive
    # unknown class, and ordinary transport has no dustbin score.
    cells = [dict(row) for row in cells]
    for row in cells:
        if float(row["severity"]) == 0.0 or row["mode"] == "ordinary":
            for metric in (
                "dustbin_precision", "dustbin_f1", "unknown_auroc", "unknown_auprc"
            ):
                row[metric] = None
    write_csv(out_root / "condition_cells.csv", cells)
    fold_rows = []
    keys = sorted({(row["mode"], row["severity"], row["fold"]) for row in cells})
    for mode, severity, fold in keys:
        part = [
            row for row in cells
            if (row["mode"], row["severity"], row["fold"]) == (mode, severity, fold)
        ]
        item: dict[str, Any] = {
            "mode": mode, "severity": severity, "fold": fold,
            "perturbation_replicates": len(part),
        }
        for metric in metric_names:
            values = [
                float(row[metric]) for row in part
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            item[metric] = float(np.mean(values)) if values else ""
        fold_rows.append(item)
    write_csv(out_root / "fold_level.csv", fold_rows)

    summary = []
    for mode, severity in sorted({(row["mode"], row["severity"]) for row in fold_rows}):
        part = [
            row for row in fold_rows
            if (row["mode"], row["severity"]) == (mode, severity)
        ]
        item = {"mode": mode, "severity": severity, "folds": len(part)}
        for metric in metric_names:
            values = [float(row[metric]) for row in part if row[metric] != ""]
            if values:
                item[f"{metric}_mean"], item[f"{metric}_sd"] = mean_sd(values)
            else:
                item[f"{metric}_mean"], item[f"{metric}_sd"] = "", ""
        summary.append(item)
    write_csv(out_root / "summary.csv", summary)

    def pm(row: dict[str, Any], metric: str) -> str:
        mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
        return "N/A" if mean == "" else f"{100*float(mean):.2f} ± {100*float(sd):.2f}%"

    lines = [
        "# Ours unknown/dustbin robustness and no-dustbin ablation",
        "",
        "Kato/RLD grouped CV5 × seed42. Synthetic distractors are unknown positives; "
        "atlas-eligible labeled held-out neurons are negatives. Native unlabeled cells "
        "remain context but are excluded from binary unknown targets.",
        "",
        "Perturbation replicates are averaged within fold first; values are the "
        "unweighted mean ± sample SD across five biological folds.",
        "",
        "| Transport | Distractor severity | Benchmark Top-1 ↑ | "
        "Reject-aware Top-1 ↑ | Unknown Recall ↑ | "
        "Forced match ↓ | Dustbin Precision ↑ | Dustbin F1 ↑ | AUROC ↑ | AUPRC ↑ | "
        "Known false reject ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"capacity_dustbin": "With dustbin", "ordinary": "No dustbin"}
    for row in summary:
        lines.append(
            f"| {labels[row['mode']]} | {float(row['severity']):.2f} | "
            f"{pm(row, 'known_top1_real')} | "
            f"{pm(row, 'known_top1_with_dustbin')} | "
            f"{pm(row, 'unknown_recall')} | "
            f"{pm(row, 'forced_match_rate')} | "
            f"{pm(row, 'dustbin_precision')} | {pm(row, 'dustbin_f1')} | "
            f"{pm(row, 'unknown_auroc')} | {pm(row, 'unknown_auprc')} | "
            f"{pm(row, 'known_false_reject_rate')} |"
        )
    lines.extend([
        "",
        "`No dustbin` is an inference-only transport ablation: the checkpoint, atlas, "
        "node encodings, and relation transport are unchanged. With no rejection "
        "state, every synthetic unknown is forcibly assigned to a known identity.",
        "`Benchmark Top-1` ranks only real identity slots, matching the Main Benchmark. "
        "`Reject-aware Top-1` additionally counts a known neuron as incorrect when the "
        "dustbin is its highest-probability output.",
    ])
    (out_root / "TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit = {
        "protocol": "RLD grouped CV5 x seed42 held-out distractor dustbin audit",
        "positive_class": "materialized __OUTLIER_* synthetic distractors",
        "negative_class": "atlas-eligible labeled held-out neurons",
        "excluded_binary_targets": "native unlabeled neurons",
        "fold_first_aggregation": True,
        "evaluation_devices": sorted({
            str(row.get("execution_device", "unknown")) for row in cells
        }),
        "modes": {
            "capacity_dustbin": "production inference",
            "ordinary": "inference-time no-dustbin transport; no retraining",
        },
        "cells": len(cells), "summary_rows": len(summary),
    }
    write_json(out_root / "AUDIT.json", audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CORR / "MANIFEST.json")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--severities", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    folds = tuple(int(value) for value in args.folds.split(",") if value.strip())
    severities = {float(value) for value in args.severities.split(",") if value.strip()}
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    if not set(modes) <= set(MODES):
        parser.error(f"modes must be selected from {MODES}")

    sys.path.insert(0, str(PACKAGE.resolve()))
    from mprt_net.data import WormCache, split_files
    from mprt_net.evaluate import load_checkpoint

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = read_json(args.manifest)
    cells = []
    for fold in folds:
        ckpt = checkpoint(fold)
        model, state = load_checkpoint(ckpt, device)
        model.eval()
        raw_mapping = state.get("atlas_identity_to_slot")
        if not isinstance(raw_mapping, dict) or not raw_mapping:
            raise RuntimeError(f"missing atlas_identity_to_slot: {ckpt}")
        identity_to_slot = {str(key): int(value) for key, value in raw_mapping.items()}
        cache = WormCache(activity_length=512, max_items=24)
        clean = saved_clean(fold)
        seen_clean = False
        for row in conditions(manifest, fold):
            severity = float(row["severity"])
            if severity not in severities:
                continue
            pseed = int(row["perturbation_seed"])
            condition_name = Path(row["root"]).name
            condition_out = args.output / "cells" / f"fold{fold}" / condition_name
            existing = [condition_out / mode / "metrics.json" for mode in modes]
            if all(path.is_file() for path in existing) and not args.force:
                results = [read_json(path) for path in existing]
            else:
                paths = split_files(Path(row["root"]), "test")
                results = evaluate_condition(
                    model=model, identity_to_slot=identity_to_slot, cache=cache,
                    paths=paths, fold=fold, severity=severity,
                    perturbation_seed=pseed, modes=modes, out_dir=condition_out,
                )
            for result in results:
                if severity == 0.0 and result["mode"] == "capacity_dustbin":
                    if abs(float(result["known_top1_real"]) - float(clean["top1_real"])) > 1e-12:
                        raise RuntimeError(
                            f"fold{fold} severity=0 does not reproduce Main Benchmark: "
                            f"{result['known_top1_real']} != {clean['top1_real']}"
                        )
                    seen_clean = True
                elif severity > 0 and not seen_clean:
                    raise RuntimeError(f"fold{fold}: nonzero severity before clean guard")
                result = dict(result)
                result["execution_device"] = str(device)
                result["checkpoint"] = str(ckpt.resolve())
                result["corruption_root"] = str(Path(row["root"]).resolve())
                cells.append(result)
            print(
                f"[DONE] fold={fold} severity={severity:.2f} p={pseed} "
                f"modes={','.join(modes)}", flush=True,
            )
    aggregate(cells, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
