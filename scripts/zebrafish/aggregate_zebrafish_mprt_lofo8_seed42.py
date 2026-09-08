#!/usr/bin/env python3
"""Aggregate the seed42 eight-fold zebrafish LOFO experiment."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


SEEDS = (42,)
VARIANTS = (
    ("full", "Full NeuRID"),
    ("no_transport", "No relation transport"),
    ("geometry_only", "Geometry-only"),
    ("activity_only", "Activity-only"),
)
METRICS = ("top1_real", "top5_real", "mrr_real", "hungarian_accuracy")


def run_dir(run_root: Path, legacy: Path, fold: int, seed: int, variant: str) -> Path:
    if fold == 1 and seed == 42 and variant == "full":
        return legacy
    return run_root / f"fold_{fold}" / f"seed{seed}" / variant


def read_metric(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    return data.get("test", data)


def exact_signflip_p(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    observed = abs(float(delta.mean()))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(delta)):
        values.append(abs(float((delta * np.asarray(signs)).mean())))
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def hierarchical_ci(values: np.ndarray, iterations: int, seed: int) -> list[float]:
    """Resample held-out fish; also resample seeds when more than one exists."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != 8 or values.shape[1] != len(SEEDS):
        raise ValueError(f"expected [8 fish,{len(SEEDS)} seeds], got {values.shape}")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        fish_index = rng.integers(0, 8, size=8)
        fish_values = []
        for fish in fish_index:
            seed_index = rng.integers(0, len(SEEDS), size=len(SEEDS))
            fish_values.append(float(values[fish, seed_index].mean()))
        estimates[iteration] = float(np.mean(fish_values))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def summarize_matrix(values: np.ndarray, bootstrap: int, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    fish_means = values.mean(axis=1)
    seed_macros = values.mean(axis=0)
    return {
        "mean": float(values.mean()),
        "sd_across_runs": float(values.reshape(-1).std(ddof=1)),
        "fish_mean_sd": float(fish_means.std(ddof=1)),
        "seed_macro_sd": (
            float(seed_macros.std(ddof=1)) if len(seed_macros) > 1 else None
        ),
        "hierarchical_ci95": hierarchical_ci(values, bootstrap, seed),
        "fish_means_after_seed_average": fish_means.tolist(),
        "seed_macro_means": {str(s): float(x) for s, x in zip(SEEDS, seed_macros)},
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--legacy-fold1-full", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    legacy = args.legacy_fold1_full.resolve()
    data_root = args.data_root.resolve()
    rows = []
    fish_names = {}

    for fold in range(1, 9):
        protocol = json.loads((data_root / f"fold_{fold}" / "seed42_protocol.json").read_text())
        test_fish = protocol.get("test_fish") or protocol["splits"]["test"]["specimens"][0]
        fish_names[fold] = test_fish
        query_counts = set()
        for seed in SEEDS:
            for variant, label in VARIANTS:
                directory = run_dir(run_root, legacy, fold, seed, variant)
                metrics_path = directory / "test_metrics.json"
                metrics = read_metric(metrics_path)
                row = {
                    "fold": fold,
                    "test_fish": test_fish,
                    "seed": seed,
                    "variant": variant,
                    "label": label,
                    "queries": int(metrics["queries"]),
                    **{key: float(metrics[key]) for key in METRICS},
                    "metrics_path": str(metrics_path),
                }
                rows.append(row)
                query_counts.add(row["queries"])
        euclidean_path = run_root / "baselines" / f"fold_{fold}" / "euclidean.json"
        euclidean = read_metric(euclidean_path)
        query_counts.add(int(euclidean["queries"]))
        if len(query_counts) != 1:
            raise RuntimeError(f"fold {fold}: query-count mismatch {sorted(query_counts)}")

        audit_path = run_root / "audits" / f"fold_{fold}" / "permutation_full_seed42.json"
        audit = json.loads(audit_path.read_text())
        if not audit.get("pass"):
            raise RuntimeError(f"fold {fold}: permutation audit failed: {audit_path}")

    expected = 8 * len(SEEDS) * len(VARIANTS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} learned rows, got {len(rows)}")

    aggregate_dir = run_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    csv_path = aggregate_dir / "all_run_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    arrays: dict[str, dict[str, np.ndarray]] = {}
    summary = {}
    for variant, label in VARIANTS:
        arrays[variant] = {}
        metric_summary = {}
        for metric_index, metric in enumerate(METRICS):
            values = np.empty((8, len(SEEDS)), dtype=float)
            for fold in range(1, 9):
                for seed_index, seed in enumerate(SEEDS):
                    match = [
                        row for row in rows
                        if row["fold"] == fold and row["seed"] == seed and row["variant"] == variant
                    ]
                    if len(match) != 1:
                        raise RuntimeError(f"missing/duplicate row fold={fold} seed={seed} {variant}")
                    values[fold - 1, seed_index] = match[0][metric]
            arrays[variant][metric] = values
            metric_summary[metric] = summarize_matrix(
                values,
                args.bootstrap,
                args.bootstrap_seed + 100 * metric_index + len(summary),
            )
        summary[variant] = {"label": label, "metrics": metric_summary}

    # Euclidean has one deterministic result per held-out fish.
    euclidean_summary = {}
    for metric in METRICS:
        values = np.asarray([
            float(read_metric(run_root / "baselines" / f"fold_{fold}" / "euclidean.json")[metric])
            for fold in range(1, 9)
        ])
        rng = np.random.default_rng(args.bootstrap_seed + 900 + METRICS.index(metric))
        boot = values[rng.integers(0, 8, size=(args.bootstrap, 8))].mean(axis=1)
        euclidean_summary[metric] = {
            "mean": float(values.mean()),
            "fish_sd": float(values.std(ddof=1)),
            "fish_values": values.tolist(),
            "bootstrap_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        }

    contrast_specs = (
        ("full", "no_transport", "Full_minus_NoTransport"),
        ("full", "geometry_only", "Full_minus_GeometryOnly"),
        ("full", "activity_only", "Full_minus_ActivityOnly"),
    )
    contrasts = {}
    for index, (left, right, name) in enumerate(contrast_specs):
        delta = arrays[left]["top1_real"] - arrays[right]["top1_real"]
        fish_delta = delta.mean(axis=1)
        contrasts[name] = {
            "mean_pp": float(100.0 * delta.mean()),
            "hierarchical_ci95_pp": [
                100.0 * value
                for value in hierarchical_ci(
                    delta,
                    args.bootstrap,
                    args.bootstrap_seed + 2000 + index,
                )
            ],
            "wins_runs": int((delta > 0).sum()),
            "ties_runs": int((delta == 0).sum()),
            "wins_fish_after_seed_average": int((fish_delta > 0).sum()),
            "exact_two_sided_signflip_p_on_8_fish": exact_signflip_p(fish_delta),
            "fish_deltas_pp": (100.0 * fish_delta).tolist(),
            "run_deltas_pp": (100.0 * delta).tolist(),
        }

    payload = {
        "protocol": (
            "8-fold leave-one-fish-out with seed42; next fish cyclically "
            "used for validation and remaining six for training; all settings "
            "locked before held-out evaluation; 60-minute q/r gap."
        ),
        "task_scope": (
            "same tracked neuron across time in an unseen fish; not cross-fish "
            "canonical identity-atlas matching"
        ),
        "primary_estimate": "equal-weight macro mean across 8 held-out fish",
        "uncertainty": "held-out-fish bootstrap and across-fish SD",
        "fish_names": fish_names,
        "learned_variants": summary,
        "euclidean": euclidean_summary,
        "paired_top1_contrasts": contrasts,
        "permutation_audit": "PASS for Full/seed42 on every held-out fish",
        "run_metrics_csv": str(csv_path),
    }
    json_path = aggregate_dir / "zebrafish_mprt_lofo8_seed42_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Zebrafish NeuRID: 8-fold LOFO, seed42",
        "",
        "Primary point estimate averages the 8 held-out fish. The ± term below "
        "is the SD across held-out fish; 95% CIs bootstrap held-out fish.",
        "",
        "| Method | Top-1 | Top-5 | MRR | Hungarian | Hierarchical 95% CI (Top-1) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    eu = euclidean_summary
    lines.append(
        "| Euclidean | "
        f"{pct(eu['top1_real']['mean'])} ± {pct(eu['top1_real']['fish_sd'])} | "
        f"{pct(eu['top5_real']['mean'])} | {eu['mrr_real']['mean']:.4f} | "
        f"{pct(eu['hungarian_accuracy']['mean'])} | "
        f"[{pct(eu['top1_real']['bootstrap_ci95'][0])}, {pct(eu['top1_real']['bootstrap_ci95'][1])}] |"
    )
    for variant, label in VARIANTS:
        metrics = summary[variant]["metrics"]
        top1 = metrics["top1_real"]
        lines.append(
            f"| {label} | {pct(top1['mean'])} ± {pct(top1['fish_mean_sd'])} | "
            f"{pct(metrics['top5_real']['mean'])} | {metrics['mrr_real']['mean']:.4f} | "
            f"{pct(metrics['hungarian_accuracy']['mean'])} | "
            f"[{pct(top1['hierarchical_ci95'][0])}, {pct(top1['hierarchical_ci95'][1])}] |"
        )
    lines.extend(["", "## Paired Top-1 contrasts", ""])
    for name, values in contrasts.items():
        lo, hi = values["hierarchical_ci95_pp"]
        lines.append(
            f"- {name}: {values['mean_pp']:+.2f} pp, 95% CI "
            f"[{lo:+.2f}, {hi:+.2f}], run wins "
            f"{values['wins_runs']}/8, fish wins "
            f"{values['wins_fish_after_seed_average']}/8, exact sign-flip "
            f"p={values['exact_two_sided_signflip_p_on_8_fish']:.4f}."
        )
    lines.extend([
        "",
        "All eight Full/seed42 row-permutation audits passed.",
        "",
        "Task interpretation: unseen-fish longitudinal matching of the same tracked "
        "neurons, not a shared cross-fish identity atlas.",
    ])
    md_path = aggregate_dir / "zebrafish_mprt_lofo8_seed42_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nJSON:", json_path)
    print("CSV :", csv_path)


if __name__ == "__main__":
    main()
