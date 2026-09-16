#!/usr/bin/env python3
"""Atlas ablation: Full static atlas vs one train-only geometry medoid.

The historical invocation is the validation-only CV5 × 3-seed pilot. A
formal main-table-compatible invocation uses ``--split test --seeds 42``.

Protocol
--------
* Dataset: Atanas or Kato/RLD grouped outer CV5
* Model seeds: configurable; defaults to the historical 1, 42, 123
* Evaluation split: validation or held-out test
* Full arm: existing final static-atlas MPRT checkpoint
* w/o Atlas arm: same exact checkpoint/model, but replace the multi-animal static
  atlas at inference with ONE outer-train geometry-medoid worm.
* No retraining and no validation/test-based template selection.
* Medoid selection uses XYZ only, no cell identity labels.
* Query universe is LOCKED to the Full Atlas training-identity vocabulary.
  If a GT identity is absent from the medoid worm, that query is counted WRONG
  (it is NOT dropped from the denominator).

The script first re-evaluates Full Atlas and requires exact agreement with the
existing component-ablation metric on the selected split before accepting
medoid results.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_DEFAULT = Path("/home/ubuntu/klb/nuclr/nuclr")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (1, 42, 123)
TOL = 1e-7


@dataclass
class Totals:
    queries: int = 0
    covered_queries: int = 0
    top1: int = 0
    top5: int = 0
    reciprocal_rank_sum: float = 0.0
    hungarian_correct: int = 0

    @property
    def top1_real(self) -> float:
        return self.top1 / max(self.queries, 1)

    @property
    def top5_real(self) -> float:
        return self.top5 / max(self.queries, 1)

    @property
    def mrr_real(self) -> float:
        return self.reciprocal_rank_sum / max(self.queries, 1)

    @property
    def hungarian_accuracy(self) -> float:
        return self.hungarian_correct / max(self.queries, 1)

    @property
    def candidate_coverage(self) -> float:
        return self.covered_queries / max(self.queries, 1)

    @property
    def top1_covered(self) -> float:
        """Top-1 conditional on the GT identity being in the medoid vocabulary."""
        return self.top1 / max(self.covered_queries, 1)

    def metrics(self) -> dict[str, Any]:
        return {
            "queries": int(self.queries),
            "covered_queries": int(self.covered_queries),
            "candidate_coverage": float(self.candidate_coverage),
            "top1_real": float(self.top1_real),
            "top1_covered": float(self.top1_covered),
            "top5_real": float(self.top5_real),
            "mrr_real": float(self.mrr_real),
            "hungarian_queries": int(self.queries),
            "hungarian_accuracy": float(self.hungarian_accuracy),
        }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize_xyz_for_medoid(xyz: np.ndarray) -> np.ndarray:
    """Existing fair-template protocol: per-animal median center, then /200."""
    x = np.asarray(xyz, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"XYZ must be [N,3], got {x.shape}")
    finite = np.isfinite(x).all(axis=1)
    x = x[finite]
    if x.shape[0] == 0:
        raise ValueError("No finite XYZ rows")
    x = x - np.median(x, axis=0, keepdims=True)
    return x / 200.0


def symmetric_mean_nn(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour Euclidean distance."""
    # Worms are small enough that dense pairwise distances are cheap.
    diff = a[:, None, :] - b[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    return 0.5 * (float(d.min(axis=1).mean()) + float(d.min(axis=0).mean()))


def select_train_geometry_medoid(train_samples: list[Any]) -> dict[str, Any]:
    """Label-free, deterministic train-only medoid."""
    xyzs = [normalize_xyz_for_medoid(s.xyz.detach().cpu().numpy()) for s in train_samples]
    n = len(train_samples)
    if n < 2:
        raise ValueError("Need at least two outer-train animals for medoid selection")

    pairwise = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            value = symmetric_mean_nn(xyzs[i], xyzs[j])
            pairwise[i, j] = value
            pairwise[j, i] = value

    means = pairwise.sum(axis=1) / float(n - 1)
    # Deterministic tie break: distance, then UID, then original train order.
    order = sorted(
        range(n),
        key=lambda i: (float(means[i]), str(train_samples[i].uid), int(i)),
    )
    idx = int(order[0])
    return {
        "index": idx,
        "uid": str(train_samples[idx].uid),
        "source_path": str(train_samples[idx].source_path),
        "mean_distance": float(means[idx]),
        "all_mean_distances": [
            {
                "index": int(i),
                "uid": str(train_samples[i].uid),
                "source_path": str(train_samples[i].source_path),
                "mean_distance": float(means[i]),
            }
            for i in order
        ],
        "selection_split": "outer_train_only",
        "uses_validation": False,
        "uses_test": False,
        "uses_identity_labels": False,
        "geometry_normalization": "per-animal median centering, then divide XYZ by 200",
        "distance": "symmetric mean nearest-neighbour Euclidean distance",
        "criterion": "minimum mean distance to all other outer-training animals",
        "tie_break": "lexicographic worm UID, then original train-list index",
    }


def subset_encoding(encoding: Any, indices: torch.Tensor) -> Any:
    """Subset a PopulationEncoding while preserving any extra future fields."""
    import copy

    def subset_pairwise(x: torch.Tensor) -> torch.Tensor:
        return x.index_select(0, indices).index_select(1, indices)

    result = copy.copy(encoding)
    result.nodes = encoding.nodes.index_select(0, indices)
    result.relations = subset_pairwise(encoding.relations)
    result.geometry_relations = subset_pairwise(encoding.geometry_relations)
    result.activity_relations = subset_pairwise(encoding.activity_relations)
    return result


def hungarian_assignment(plan: torch.Tensor) -> dict[int, int]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError("scipy is required for Hungarian accuracy") from exc

    matrix = plan[:-1, :-1].detach().float().cpu().numpy()
    rows, cols = linear_sum_assignment(-matrix)
    return {int(r): int(c) for r, c in zip(rows, cols)}


def canonical_queries(sample: Any, full_identity_to_slot: dict[str, int]) -> list[tuple[int, str]]:
    """Exactly the Full Atlas evaluator's query universe for one animal."""
    from mprt_net.data import unique_identity_map

    found = unique_identity_map(sample)
    rows = [
        (int(node_index), str(identity))
        for identity, node_index in found.items()
        if str(identity) in full_identity_to_slot
    ]
    rows.sort(key=lambda x: x[0])
    return rows


def evaluate_output(
    output: Any,
    queries: list[tuple[int, str]],
    candidate_identity_to_col: dict[str, int],
) -> tuple[Totals, list[dict[str, Any]]]:
    """Evaluate on a FIXED canonical query universe.

    Missing GT candidate => zero credit, not dropped.
    """
    probs = output.row_conditional.detach()
    real = probs[:, :-1]
    assignment = hungarian_assignment(output.plan)

    totals = Totals()
    records: list[dict[str, Any]] = []
    candidate_col_to_identity = {
        int(column): str(identity)
        for identity, column in candidate_identity_to_col.items()
    }

    for node_index, identity in queries:
        totals.queries += 1
        target_col = candidate_identity_to_col.get(identity)
        row = real[int(node_index)]
        predicted_col = int(torch.argmax(row).item())
        stable_order = torch.argsort(row, descending=True, stable=True)
        stable_top5_cols = [
            int(value) for value in stable_order[: min(5, int(real.shape[1]))].tolist()
        ]
        hp = int(assignment.get(int(node_index), -1))
        decoded = {
            "predicted_col": predicted_col,
            "predicted_identity": candidate_col_to_identity.get(predicted_col, ""),
            "top5_cols": "|".join(map(str, stable_top5_cols)),
            "top5_identities": "|".join(
                candidate_col_to_identity.get(column, "") for column in stable_top5_cols
            ),
            "hungarian_prediction_col": hp,
            "hungarian_prediction_identity": candidate_col_to_identity.get(hp, ""),
        }

        if target_col is None:
            records.append(
                {
                    "node_index": int(node_index),
                    "identity": identity,
                    "candidate_present": 0,
                    "target_col": -1,
                    "rank": -1,
                    "top1": 0,
                    "top5": 0,
                    "rr": 0.0,
                    "hungarian_correct": 0,
                    "strictly_greater_candidates": "",
                    "equal_score_candidates": "",
                    "rank_min": "",
                    "rank_max": "",
                    **decoded,
                }
            )
            continue

        if target_col < 0 or target_col >= real.shape[1]:
            raise RuntimeError(
                f"Target column {target_col} out of bounds for {identity}; M={real.shape[1]}"
            )

        totals.covered_queries += 1
        score = row[int(target_col)]
        greater = int((row > score).sum().item())
        tied = int((row == score).sum().item())
        rank = 1 + greater
        top1 = int(rank == 1)
        top5 = int(rank <= min(5, int(real.shape[1])))
        rr = 1.0 / float(rank)
        hc = int(hp == int(target_col))

        totals.top1 += top1
        totals.top5 += top5
        totals.reciprocal_rank_sum += rr
        totals.hungarian_correct += hc

        records.append(
            {
                "node_index": int(node_index),
                "identity": identity,
                "candidate_present": 1,
                "target_col": int(target_col),
                "rank": int(rank),
                "top1": top1,
                "top5": top5,
                "rr": float(rr),
                "hungarian_correct": hc,
                "strictly_greater_candidates": greater,
                "equal_score_candidates": tied,
                "rank_min": greater + 1,
                "rank_max": greater + tied,
                **decoded,
            }
        )

    return totals, records


def exact_guard(
    observed: dict[str, Any], saved: dict[str, Any], fold: int, split: str
) -> None:
    keys = ("queries", "top1_real", "top5_real", "mrr_real", "hungarian_accuracy")
    diffs = {}
    for key in keys:
        a = observed[key]
        b = saved[key]
        if key == "queries":
            ok = int(a) == int(b)
        else:
            ok = math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=TOL)
        if not ok:
            diffs[key] = {"recomputed": a, "saved": b}
    if diffs:
        raise RuntimeError(
            f"Fold{fold}: Full Atlas {split} baseline was NOT exactly reproduced: {diffs}"
        )


def mean_sd(xs: list[float]) -> tuple[float, float]:
    return (
        statistics.mean(xs),
        statistics.stdev(xs) if len(xs) > 1 else 0.0,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=REPO_DEFAULT)
    p.add_argument(
        "--dataset",
        choices=("atanas", "rld"),
        default="atanas",
        help="Use rld for the Kato/RLD benchmark dataset",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--activity-length", type=int, default=512)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="Model seeds to evaluate (formal main-table protocol: --seeds 42)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Default: historical validation directory for val/1,42,123; "
            "otherwise a split/seed-specific directory"
        ),
    )
    args = p.parse_args()

    repo = args.repo_root.resolve()
    package_root = repo / "mprt_net_v1_1"
    dataset_roots = {
        "atanas": repo / "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
        "rld": repo / "Data/Dunn_001623/cv5_grouped_v1",
    }
    data_root = dataset_roots[args.dataset]
    final_root = (
        repo / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1" / args.dataset
    )
    component_root = (
        repo / "runs/mprt_v1_1_component_ablation_cv5x3_v1" / args.dataset
    )
    seeds = tuple(dict.fromkeys(args.seeds))
    if not seeds:
        raise ValueError("At least one model seed is required")
    if args.output_root is not None:
        out_root = args.output_root.resolve()
    elif args.dataset == "atanas" and args.split == "val" and seeds == SEEDS:
        out_root = repo / "runs/mprt_v1_1_atlas_medoid_atanas_cv5x3_v1"
    else:
        seed_tag = "_".join(str(seed) for seed in seeds)
        out_root = (
            repo
            / f"runs/mprt_v1_1_atlas_medoid_{args.dataset}_cv5_{args.split}_seeds_{seed_tag}_v1"
        )

    protocol = (
        f"{args.dataset}_full_atlas_vs_single_train_geometry_medoid_"
        f"{args.split}_cv5_seed42_v1"
    )
    if seeds != (42,):
        seed_tag = "_".join(map(str, seeds))
        protocol = (
            f"{args.dataset}_full_atlas_vs_single_train_geometry_medoid_"
            f"{args.split}_cv5_seeds_{seed_tag}_v1"
        )

    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    sys.path.insert(0, str(package_root))

    from mprt_net.data import WormCache, split_files, unique_identity_map
    from mprt_net.evaluate import load_checkpoint

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 120)
    print(
        f"{args.dataset.upper()} — FULL STATIC ATLAS vs SINGLE TRAIN-ONLY MEDOID — "
        f"{args.split.upper()} — CV5 × {len(seeds)} SEED(S)"
    )
    print("=" * 120)
    print("device      :", device)
    print("data root   :", data_root)
    print("output root :", out_root)
    print("seeds       :", ", ".join(map(str, seeds)))
    print("medoid rule : outer-train geometry only; no validation/test selection")
    print()

    fold_results: list[dict[str, Any]] = []

    for fold in FOLDS:
        fold_root = data_root / f"fold_{fold}"

        # The train-only geometry medoid depends only on the biological fold,
        # not on the stochastic model seed. Select it ONCE per fold.
        train_dir = fold_root / "train"
        eval_dir = fold_root / args.split
        for required in (train_dir, eval_dir):
            if not required.exists():
                raise FileNotFoundError(required)

        train_paths = split_files(fold_root, "train")
        eval_paths = split_files(fold_root, args.split)
        if not train_paths or not eval_paths:
            raise RuntimeError(f"fold{fold}: empty train or {args.split} split")

        cache = WormCache(
            activity_length=args.activity_length,
            max_items=max(16, len(train_paths) + len(eval_paths)),
        )
        train_samples = [cache.get(path) for path in train_paths]
        medoid_info = select_train_geometry_medoid(train_samples)
        template_cpu = train_samples[int(medoid_info["index"])]

        print(
            f"\\n[FOLD {fold}] fixed train-only medoid = {medoid_info['uid']} "
            f"(mean_distance={medoid_info['mean_distance']:.6f})",
            flush=True,
        )

        for seed in seeds:
            checkpoint_path = (
                final_root / f"fold{fold}/seed{seed}/static_atlas/anchored_pure.pt"
            )
            saved_full_path = (
                component_root / f"fold{fold}/seed{seed}/metrics/{args.split}/full.json"
            )

            for required in (checkpoint_path, saved_full_path):
                if not required.exists():
                    raise FileNotFoundError(required)

            model, checkpoint = load_checkpoint(checkpoint_path, device)
            model.eval()

            if not getattr(model, "atlas_is_initialized", False):
                raise RuntimeError(
                    f"fold{fold} seed{seed}: static checkpoint has no initialized atlas"
                )

            raw_mapping = checkpoint.get("atlas_identity_to_slot")
            if not isinstance(raw_mapping, dict) or not raw_mapping:
                raise RuntimeError(
                    f"fold{fold} seed{seed}: checkpoint has no atlas_identity_to_slot"
                )

            full_identity_to_slot = {str(k): int(v) for k, v in raw_mapping.items()}
            full_atlas = model.atlas_encoding()

            # Same biological medoid, encoded by the current model seed.
            template = template_cpu.to(device)
            with torch.inference_mode():
                template_encoding_all = model.encode_population(template)

            template_unique = {
                str(identity): int(node)
                for identity, node in unique_identity_map(template_cpu).items()
                if str(identity) in full_identity_to_slot
            }
            if not template_unique:
                raise RuntimeError(
                    f"fold{fold} seed{seed}: medoid has no identity in Full Atlas vocabulary"
                )

            candidate_pairs = sorted(
                ((node, identity) for identity, node in template_unique.items()),
                key=lambda x: (x[0], x[1]),
            )
            candidate_indices = torch.tensor(
                [node for node, _ in candidate_pairs],
                dtype=torch.long,
                device=device,
            )
            candidate_labels = [identity for _, identity in candidate_pairs]
            medoid_identity_to_col = {
                identity: col for col, identity in enumerate(candidate_labels)
            }
            medoid_encoding = subset_encoding(template_encoding_all, candidate_indices)

            full_totals = Totals()
            medoid_totals = Totals()
            per_query: list[dict[str, Any]] = []

            with torch.inference_mode():
                for number, path in enumerate(eval_paths, start=1):
                    sample_cpu = cache.get(path)
                    sample = sample_cpu.to(device)
                    qenc = model.encode_population(sample)

                    full_out = model.match_encodings(qenc, full_atlas)
                    medoid_out = model.match_encodings(qenc, medoid_encoding)

                    queries = canonical_queries(sample_cpu, full_identity_to_slot)

                    full_total, full_rows = evaluate_output(
                        full_out, queries, full_identity_to_slot
                    )
                    med_total, med_rows = evaluate_output(
                        medoid_out, queries, medoid_identity_to_col
                    )

                    for attr in (
                        "queries",
                        "covered_queries",
                        "top1",
                        "top5",
                        "reciprocal_rank_sum",
                        "hungarian_correct",
                    ):
                        setattr(
                            full_totals,
                            attr,
                            getattr(full_totals, attr) + getattr(full_total, attr),
                        )
                        setattr(
                            medoid_totals,
                            attr,
                            getattr(medoid_totals, attr) + getattr(med_total, attr),
                        )

                    if len(full_rows) != len(med_rows):
                        raise RuntimeError("Full and medoid canonical query counts differ")

                    for fr, mr in zip(full_rows, med_rows):
                        if (fr["node_index"], fr["identity"]) != (
                            mr["node_index"], mr["identity"]
                        ):
                            raise RuntimeError("Full/medoid query order mismatch")

                        per_query.append(
                            {
                                "fold": fold,
                                "seed": seed,
                                "uid": str(sample_cpu.uid),
                                "source_path": str(sample_cpu.source_path),
                                "node_index": int(fr["node_index"]),
                                "identity": fr["identity"],
                                "medoid_uid": medoid_info["uid"],
                                "medoid_candidate_present": int(mr["candidate_present"]),
                                "full_rank": int(fr["rank"]),
                                "medoid_rank": int(mr["rank"]),
                                "full_top1": int(fr["top1"]),
                                "medoid_top1": int(mr["top1"]),
                                "full_top5": int(fr["top5"]),
                                "medoid_top5": int(mr["top5"]),
                                "full_rr": float(fr["rr"]),
                                "medoid_rr": float(mr["rr"]),
                                "full_hungarian_correct": int(fr["hungarian_correct"]),
                                "medoid_hungarian_correct": int(mr["hungarian_correct"]),
                                "full_predicted_identity": fr["predicted_identity"],
                                "medoid_predicted_identity": mr["predicted_identity"],
                                "full_top5_identities": fr["top5_identities"],
                                "medoid_top5_identities": mr["top5_identities"],
                                "full_hungarian_prediction_identity": fr["hungarian_prediction_identity"],
                                "medoid_hungarian_prediction_identity": mr["hungarian_prediction_identity"],
                                "full_rank_min": fr["rank_min"],
                                "full_rank_max": fr["rank_max"],
                                "medoid_rank_min": mr["rank_min"],
                                "medoid_rank_max": mr["rank_max"],
                            }
                        )

                    print(
                        f"fold{fold} seed{seed} {args.split} "
                        f"{number:02d}/{len(eval_paths):02d} "
                        f"uid={sample_cpu.uid} canonical_queries={len(queries)}",
                        flush=True,
                    )

            full_metrics = full_totals.metrics()
            full_metrics["covered_queries"] = full_metrics["queries"]
            full_metrics["candidate_coverage"] = 1.0

            saved_full = json.loads(saved_full_path.read_text(encoding="utf-8"))
            exact_guard(full_metrics, saved_full, fold, args.split)

            medoid_metrics = medoid_totals.metrics()

            delta = {
                key: float(full_metrics[key]) - float(medoid_metrics[key])
                for key in (
                    "top1_real",
                    "top5_real",
                    "mrr_real",
                    "hungarian_accuracy",
                )
            }

            result = {
                "protocol": protocol,
                "dataset": args.dataset,
                "split": args.split,
                "fold": fold,
                "seed": seed,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "full_atlas_size": int(len(full_identity_to_slot)),
                "train_animals": len(train_paths),
                "evaluation_animals": len(eval_paths),
                "medoid": medoid_info,
                "medoid_identity_candidates": len(candidate_labels),
                "query_universe": (
                    f"unique supervised {args.split} identities present in the Full train-only atlas; "
                    "GT absent from medoid counts as incorrect, never dropped"
                ),
                "full_baseline_guard": {
                    "saved_metrics": str(saved_full_path.resolve()),
                    "status": "NUMERICALLY_EXACT_REPRODUCTION",
                },
                "full_atlas": full_metrics,
                "single_medoid": medoid_metrics,
                "full_minus_medoid": delta,
            }

            cell_dir = out_root / f"fold{fold}" / f"seed{seed}"
            write_json(cell_dir / "metrics.json", result)

            import csv
            with (cell_dir / "query_level.csv").open(
                "w", encoding="utf-8", newline=""
            ) as h:
                if per_query:
                    writer = csv.DictWriter(h, fieldnames=list(per_query[0]))
                    writer.writeheader()
                    writer.writerows(per_query)

            fold_results.append(result)

            print(
                f"\\n[CELL fold{fold} seed{seed}] "
                f"coverage={100*medoid_metrics['candidate_coverage']:.2f}% "
                f"Full={100*full_metrics['top1_real']:.2f}% "
                f"Medoid={100*medoid_metrics['top1_real']:.2f}% "
                f"Δ={100*delta['top1_real']:+.2f} pp\\n",
                flush=True,
            )

            del (
                model,
                checkpoint,
                full_atlas,
                template,
                template_encoding_all,
                medoid_encoding,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    arm_metrics = (
        "top1_real",
        "top1_covered",
        "top5_real",
        "mrr_real",
        "hungarian_accuracy",
    )
    delta_metrics = ("top1_real", "top5_real", "mrr_real", "hungarian_accuracy")
    summary: dict[str, Any] = {
        "protocol": protocol,
        "dataset": args.dataset,
        "split": args.split,
        "folds": list(FOLDS),
        "seeds": list(seeds),
        "fold_results": fold_results,
        "aggregate": {},
    }

    for arm_key in ("full_atlas", "single_medoid"):
        summary["aggregate"][arm_key] = {}
        for metric in arm_metrics:
            vals = [float(r[arm_key][metric]) for r in fold_results]
            mean, sd = mean_sd(vals)
            summary["aggregate"][arm_key][metric] = {
                "mean": mean,
                "sample_sd": sd,
            }

    coverage_vals = [
        float(r["single_medoid"]["candidate_coverage"]) for r in fold_results
    ]
    cm, csd = mean_sd(coverage_vals)
    summary["aggregate"]["single_medoid"]["candidate_coverage"] = {
        "mean": cm,
        "sample_sd": csd,
    }

    summary["aggregate"]["full_minus_medoid"] = {}
    for metric in delta_metrics:
        vals = [float(r["full_minus_medoid"][metric]) for r in fold_results]
        mean, sd = mean_sd(vals)
        summary["aggregate"]["full_minus_medoid"][metric] = {
            "mean": mean,
            "sample_sd": sd,
            "fold_wins_full": int(sum(v > 0 for v in vals)),
            "fold_losses_full": int(sum(v < 0 for v in vals)),
            "fold_ties": int(sum(v == 0 for v in vals)),
        }

    write_json(out_root / "summary.json", summary)

    F = summary["aggregate"]["full_atlas"]
    M = summary["aggregate"]["single_medoid"]
    D = summary["aggregate"]["full_minus_medoid"]

    def pct(item: dict[str, float]) -> str:
        return f"{100*item['mean']:.2f}±{100*item['sample_sd']:.2f}%"

    def num(item: dict[str, float]) -> str:
        return f"{item['mean']:.4f}±{item['sample_sd']:.4f}"

    print("\\n" + "=" * 124)
    print(
        f"{args.dataset.upper()} — FULL ATLAS vs SINGLE MEDOID — "
        f"{args.split.upper()} — 5 FOLDS × {len(seeds)} SEED(S)"
    )
    print("=" * 124)
    print(
        f"{'Arm':24s} {'Top-1':>16s} {'Top-5':>16s} {'MRR':>18s} "
        f"{'Hungarian':>16s} {'Coverage':>16s}"
    )
    print(
        f"{'Single Medoid':24s} {pct(M['top1_real']):>16s} {pct(M['top5_real']):>16s} "
        f"{num(M['mrr_real']):>18s} {pct(M['hungarian_accuracy']):>16s} "
        f"{pct(M['candidate_coverage']):>16s}"
    )
    print(
        f"{'Full Atlas':24s} {pct(F['top1_real']):>16s} {pct(F['top5_real']):>16s} "
        f"{num(F['mrr_real']):>18s} {pct(F['hungarian_accuracy']):>16s} "
        f"{'100.00%':>16s}"
    )
    print(f"Single Medoid covered-only Top-1: {pct(M['top1_covered'])}")
    print()
    dt = D["top1_real"]
    print(
        "Full Atlas - Single Medoid Top-1 = "
        f"{100*dt['mean']:+.2f}±{100*dt['sample_sd']:.2f} pp; "
        f"Full wins {dt['fold_wins_full']}/{len(fold_results)} cells"
    )

    print("\\nCell-level Top-1:")
    for r in fold_results:
        print(
            f"  fold{r['fold']} seed{r['seed']}: "
            f"Full={100*r['full_atlas']['top1_real']:.2f}%  "
            f"Medoid={100*r['single_medoid']['top1_real']:.2f}%  "
            f"Δ={100*r['full_minus_medoid']['top1_real']:+.2f} pp  "
            f"Coverage={100*r['single_medoid']['candidate_coverage']:.2f}%  "
            f"MedoidUID={r['medoid']['uid']}"
        )

    print(
        f"\\nFold-averaged paired ΔTop-1 "
        f"(mean over {len(seeds)} model seed(s)):"
    )
    fold_deltas = []
    for fold in FOLDS:
        values = [
            float(r["full_minus_medoid"]["top1_real"])
            for r in fold_results
            if int(r["fold"]) == fold
        ]
        if len(values) != len(seeds):
            raise RuntimeError(
                f"fold{fold}: expected {len(seeds)} seeds, got {len(values)}"
            )
        value = statistics.mean(values)
        fold_deltas.append(value)
        print(f"  fold{fold}: {100*value:+.2f} pp")

    fm, fsd = mean_sd(fold_deltas)
    print(
        f"Biological-fold mean ΔTop-1 = {100*fm:+.2f}±{100*fsd:.2f} pp; "
        f"Full wins {sum(x > 0 for x in fold_deltas)}/5 folds"
    )

    print("\\nSaved:", out_root / "summary.json")


if __name__ == "__main__":
    main()
