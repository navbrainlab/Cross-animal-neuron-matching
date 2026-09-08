from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from ..data import PairIndex, WormCache, WormSample, unique_identity_map
from ..evaluate import load_checkpoint
from ..model import MPRTNet, PopulationEncoding
from ..sinkhorn import SinkhornResult, augmented_sinkhorn, contracted_relation_cost, log_sinkhorn


MODES = ("ordinary", "uniform_dustbin", "capacity_dustbin")


@dataclass
class Conditionals:
    row: torch.Tensor
    column: torch.Tensor


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _augmented_logits(
    real_logits: torch.Tensor,
    deletion_logit: torch.Tensor,
    insertion_logit: torch.Tensor,
) -> torch.Tensor:
    num_a, num_b = real_logits.shape
    return torch.cat(
        [
            torch.cat([real_logits, deletion_logit.to(real_logits).expand(num_a, 1)], 1),
            torch.cat(
                [insertion_logit.to(real_logits).expand(1, num_b), real_logits.new_zeros(1, 1)],
                1,
            ),
        ],
        0,
    )


def solve_transport(
    real_logits: torch.Tensor,
    deletion_logit: torch.Tensor,
    insertion_logit: torch.Tensor,
    iterations: int,
    mode: str,
) -> SinkhornResult:
    if mode == "capacity_dustbin":
        return augmented_sinkhorn(
            real_logits, deletion_logit, insertion_logit, iterations=iterations
        )
    num_a, num_b = real_logits.shape
    if mode == "uniform_dustbin":
        augmented = _augmented_logits(real_logits, deletion_logit, insertion_logit)
        mu = real_logits.new_full((num_a + 1,), 1.0 / (num_a + 1))
        nu = real_logits.new_full((num_b + 1,), 1.0 / (num_b + 1))
        log_plan = log_sinkhorn(augmented, mu.log(), nu.log(), iterations)
        return SinkhornResult(log_plan.exp(), log_plan, augmented, mu, nu)
    if mode != "ordinary":
        raise ValueError(f"Unknown Sinkhorn mode: {mode}")
    mu_real = real_logits.new_full((num_a,), 1.0 / num_a)
    nu_real = real_logits.new_full((num_b,), 1.0 / num_b)
    log_real = log_sinkhorn(real_logits, mu_real.log(), nu_real.log(), iterations)
    plan = real_logits.new_zeros(num_a + 1, num_b + 1)
    plan[:num_a, :num_b] = log_real.exp()
    log_plan = real_logits.new_full((num_a + 1, num_b + 1), -torch.inf)
    log_plan[:num_a, :num_b] = log_real
    augmented = real_logits.new_full((num_a + 1, num_b + 1), -torch.inf)
    augmented[:num_a, :num_b] = real_logits
    mu = torch.cat([mu_real, mu_real.new_zeros(1)])
    nu = torch.cat([nu_real, nu_real.new_zeros(1)])
    return SinkhornResult(plan, log_plan, augmented, mu, nu)


@torch.no_grad()
def match_with_solver(
    model: MPRTNet,
    encoding_a: PopulationEncoding,
    encoding_b: PopulationEncoding,
    mode: str,
) -> Conditionals:
    nodes_a = F.normalize(encoding_a.nodes, dim=-1)
    nodes_b = F.normalize(encoding_b.nodes, dim=-1)
    unary = (nodes_a @ nodes_b.transpose(0, 1)) / model.unary_temperature
    logits = unary
    solved = solve_transport(
        logits,
        model.deletion_logit,
        model.insertion_logit,
        model.config.sinkhorn_iterations,
        mode,
    )
    if model.config.use_relation_transport:
        for _ in range(model.config.transport_steps):
            relation_cost = contracted_relation_cost(
                encoding_a.relations, encoding_b.relations, solved.plan[:-1, :-1]
            )
            logits = unary - model.structural_weight * relation_cost / model.relation_temperature
            solved = solve_transport(
                logits,
                model.deletion_logit,
                model.insertion_logit,
                model.config.sinkhorn_iterations,
                mode,
            )
    num_a, num_b = unary.shape
    row = solved.plan[:num_a, :] / solved.mu[:num_a, None].clamp_min(1e-12)
    column = (
        solved.plan[:, :num_b] / solved.nu[None, :num_b].clamp_min(1e-12)
    ).transpose(0, 1)
    return Conditionals(row=row, column=column)


def _append_spurious_duplicates(
    sample: WormSample,
    count: int,
    generator: torch.Generator,
    xyz_jitter: float,
    activity_jitter: float,
) -> tuple[WormSample, list[int]]:
    if count <= 0:
        return sample, []
    chosen = torch.randint(sample.num_nodes, (count,), generator=generator)
    xyz_scale = sample.xyz.std(dim=0, unbiased=False).clamp_min(1e-4)
    activity_scale = sample.activity.std().clamp_min(1e-4)
    xyz_noise = torch.randn(count, 3, generator=generator) * xyz_scale * xyz_jitter
    activity_noise = (
        torch.randn(count, sample.activity.shape[1], generator=generator)
        * activity_scale
        * activity_jitter
    )
    xyz = torch.cat([sample.xyz, sample.xyz[chosen] + xyz_noise], dim=0)
    activity = torch.cat(
        [sample.activity, sample.activity[chosen] + activity_noise], dim=0
    )
    first = sample.num_nodes
    spurious = list(range(first, first + count))
    return (
        WormSample(
            uid=sample.uid,
            xyz=xyz,
            activity=activity,
            cell_ids=sample.cell_ids
            + tuple(f"__SPURIOUS_{sample.uid}_{index}" for index in range(count)),
            supervised_mask=torch.cat(
                [sample.supervised_mask, torch.zeros(count, dtype=torch.bool)]
            ),
            source_path=sample.source_path,
        ),
        spurious,
    )


def perturb_target(
    source: WormSample,
    target: WormSample,
    rate: float,
    generator: torch.Generator,
    xyz_jitter: float,
    activity_jitter: float,
) -> tuple[WormSample, list[tuple[int, int | None]], list[int]]:
    source_map = unique_identity_map(source)
    target_map = unique_identity_map(target)
    common = sorted(set(source_map).intersection(target_map))
    if len(common) < 2:
        raise ValueError("A robustness pair needs at least two shared unique identities")
    drop_count = min(int(round(rate * len(common))), len(common) - 1)
    order = torch.randperm(len(common), generator=generator).tolist()
    dropped_ids = {common[index] for index in order[:drop_count]}
    drop_indices = {target_map[identity] for identity in dropped_ids}
    keep = torch.tensor(
        [index not in drop_indices for index in range(target.num_nodes)], dtype=torch.bool
    )
    if int(keep.sum()) < 2:
        raise RuntimeError("Perturbation left fewer than two target nodes")
    old_to_new = {
        old: new for new, old in enumerate(keep.nonzero(as_tuple=False).flatten().tolist())
    }
    kept_target = target.subset(keep)
    spurious_count = int(round(rate * len(common)))
    perturbed, spurious = _append_spurious_duplicates(
        kept_target, spurious_count, generator, xyz_jitter, activity_jitter
    )
    queries = [
        (
            source_map[identity],
            None if identity in dropped_ids else old_to_new[target_map[identity]],
        )
        for identity in common
    ]
    return perturbed, queries, spurious


def _scenario_records(
    *,
    conditionals: Conditionals,
    source_queries: list[tuple[int, int | None]],
    spurious_target_indices: list[int],
    mode: str,
    rate: float,
    scenario_id: str,
    pair_id: str,
) -> Iterable[dict[str, Any]]:
    for query_index, target_index in source_queries:
        probabilities = conditionals.row[query_index]
        predicted = int(probabilities.argmax())
        is_unmatched = target_index is None
        yield {
            "mode": mode,
            "rate": rate,
            "scenario_id": scenario_id,
            "pair_id": pair_id,
            "query_type": "missing" if is_unmatched else "matched",
            "is_unmatched": int(is_unmatched),
            "dustbin_score": float(probabilities[-1]),
            "rejected": int(predicted == probabilities.numel() - 1),
            "correct_real": (
                "" if is_unmatched else int(predicted == int(target_index))
            ),
        }
    for target_index in spurious_target_indices:
        probabilities = conditionals.column[target_index]
        predicted = int(probabilities.argmax())
        yield {
            "mode": mode,
            "rate": rate,
            "scenario_id": scenario_id,
            "pair_id": pair_id,
            "query_type": "spurious",
            "is_unmatched": 1,
            "dustbin_score": float(probabilities[-1]),
            "rejected": int(predicted == probabilities.numel() - 1),
            "correct_real": "",
        }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = labels == 1
    negatives = ~positives
    num_pos, num_neg = int(positives.sum()), int(negatives.sum())
    if num_pos == 0 or num_neg == 0:
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
    rank_sum = ranks[positives].sum()
    return float((rank_sum - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    num_pos = int(labels.sum())
    if num_pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / num_pos)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in rows if row["query_type"] == "matched"]
    unmatched = [row for row in rows if row["is_unmatched"] == 1]
    labels = np.asarray([int(row["is_unmatched"]) for row in rows])
    scores = np.asarray([float(row["dustbin_score"]) for row in rows])
    tp = sum(int(row["rejected"]) for row in unmatched)
    fp = sum(int(row["rejected"]) for row in matched)
    fn = len(unmatched) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1) if unmatched else None
    f1 = (
        2 * precision * recall / max(precision + recall, 1e-12)
        if recall is not None
        else None
    )
    return {
        "queries": len(rows),
        "matched_queries": len(matched),
        "unmatched_queries": len(unmatched),
        "matched_top1_real": (
            sum(int(row["correct_real"]) for row in matched) / len(matched)
            if matched
            else None
        ),
        "matched_false_reject_rate": fp / len(matched) if matched else None,
        "unmatched_recall": recall,
        "forced_match_error": 1.0 - recall if recall is not None else None,
        "dustbin_precision": precision if unmatched else None,
        "dustbin_f1": f1,
        "unmatched_auroc": _roc_auc(labels, scores),
        "unmatched_average_precision": _average_precision(labels, scores),
    }


def _cluster_bootstrap(
    rows: list[dict[str, Any]], iterations: int, seed: int
) -> dict[str, dict[str, float | None]]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["scenario_id"])].append(row)
    names = tuple(clusters)
    rng = np.random.default_rng(seed)
    metrics = ("matched_top1_real", "unmatched_recall", "forced_match_error", "unmatched_auroc")
    draws = {metric: [] for metric in metrics}
    for _ in range(iterations):
        sampled = rng.integers(0, len(names), size=len(names))
        replicate = [row for index in sampled for row in clusters[names[int(index)]]]
        values = summarize(replicate)
        for metric in metrics:
            if values[metric] is not None:
                draws[metric].append(float(values[metric]))
    output: dict[str, dict[str, float | None]] = {}
    for metric, values in draws.items():
        if not values:
            output[metric] = {"ci95_low": None, "ci95_high": None}
        else:
            array = np.asarray(values)
            output[metric] = {
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
            }
    return output


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dustbin robustness under missing neurons and spurious duplicates"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rates", default="0,0.1,0.2,0.3,0.4")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--xyz-jitter", type=float, default=0.03)
    parser.add_argument("--activity-jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-output", type=Path, required=True)
    args = parser.parse_args()
    rates = tuple(float(value) for value in args.rates.split(","))
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    if any(rate < 0 or rate >= 1 for rate in rates):
        parser.error("Every rate must be in [0, 1)")
    if not set(modes).issubset(MODES):
        parser.error(f"Modes must be selected from {MODES}")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    device = _device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if model.atlas_is_initialized:
        raise ValueError(
            "Dustbin ablation must use the atlas-free pairwise checkpoint so the "
            "transport solver is the only changed component"
        )
    model.eval()
    index = PairIndex(args.dataset_root, args.split, min_shared=args.min_shared)
    pairs = index.pairs[: args.max_pairs] if args.max_pairs else index.pairs
    cache = WormCache(activity_length=args.activity_length, max_items=max(48, len(index.files)))
    rows: list[dict[str, Any]] = []
    for pair_number, (path_a, path_b) in enumerate(pairs, start=1):
        original_a, original_b = cache.get(path_a), cache.get(path_b)
        pair_id = "|".join(sorted((original_a.uid, original_b.uid)))
        for orientation, (source, target) in enumerate(
            ((original_a, original_b), (original_b, original_a))
        ):
            for rate_index, rate in enumerate(rates):
                for repeat in range(args.repeats):
                    scenario_seed = (
                        args.seed
                        + pair_number * 1_000_003
                        + orientation * 100_003
                        + rate_index * 10_007
                        + repeat
                    )
                    generator = torch.Generator(device="cpu").manual_seed(scenario_seed)
                    perturbed, source_queries, spurious = perturb_target(
                        source,
                        target,
                        rate,
                        generator,
                        args.xyz_jitter,
                        args.activity_jitter,
                    )
                    encoding_source = model.encode_population(source.to(device))
                    encoding_target = model.encode_population(perturbed.to(device))
                    scenario_id = (
                        f"{pair_id}|orientation{orientation}|rate{rate:.4f}|rep{repeat}"
                    )
                    for mode in modes:
                        conditionals = match_with_solver(
                            model, encoding_source, encoding_target, mode
                        )
                        rows.extend(
                            _scenario_records(
                                conditionals=conditionals,
                                source_queries=source_queries,
                                spurious_target_indices=spurious,
                                mode=mode,
                                rate=rate,
                                scenario_id=scenario_id,
                                pair_id=pair_id,
                            )
                        )
        print(f"dustbin pair={pair_number:03d}/{len(pairs):03d} id={pair_id}", flush=True)
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), float(row["rate"]))].append(row)
    results = []
    for group_index, ((mode, rate), group) in enumerate(sorted(grouped.items())):
        result = {"mode": mode, "rate": rate, **summarize(group)}
        if args.bootstrap_iterations > 0:
            result["scenario_cluster_bootstrap"] = _cluster_bootstrap(
                group,
                args.bootstrap_iterations,
                args.seed + group_index + 500_000,
            )
        results.append(result)
    payload = {
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "pairs": len(pairs),
        "orientations_per_pair": 2,
        "repeats": args.repeats,
        "rates": rates,
        "modes": modes,
        "perturbation": {
            "missing": "drop rate × shared identities from target",
            "spurious": "append rate × shared identities as jittered duplicates",
            "xyz_jitter": args.xyz_jitter,
            "activity_jitter": args.activity_jitter,
        },
        "results": results,
    }
    args.query_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0])
    with args.query_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
