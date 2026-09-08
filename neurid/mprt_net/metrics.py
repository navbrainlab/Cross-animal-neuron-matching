from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import PairTargets
from .model import MPRTOutput


@dataclass
class MetricTotals:
    queries: int = 0
    real_top1_correct: int = 0
    real_top5_correct: int = 0
    real_reciprocal_rank_sum: float = 0.0
    partial_top1_correct: int = 0
    partial_top5_correct: int = 0
    partial_reciprocal_rank_sum: float = 0.0
    dustbin_top1: int = 0
    hungarian_queries: int = 0
    hungarian_correct: int = 0

    def update(self, other: "MetricTotals") -> None:
        self.queries += other.queries
        self.real_top1_correct += other.real_top1_correct
        self.real_top5_correct += other.real_top5_correct
        self.real_reciprocal_rank_sum += other.real_reciprocal_rank_sum
        self.partial_top1_correct += other.partial_top1_correct
        self.partial_top5_correct += other.partial_top5_correct
        self.partial_reciprocal_rank_sum += other.partial_reciprocal_rank_sum
        self.dustbin_top1 += other.dustbin_top1
        self.hungarian_queries += other.hungarian_queries
        self.hungarian_correct += other.hungarian_correct

    def compute(self) -> dict[str, float | int]:
        denominator = max(self.queries, 1)
        h_denominator = max(self.hungarian_queries, 1)
        return {
            "queries": self.queries,
            "top1_real": self.real_top1_correct / denominator,
            "top5_real": self.real_top5_correct / denominator,
            "mrr_real": self.real_reciprocal_rank_sum / denominator,
            "top1_with_dustbin": self.partial_top1_correct / denominator,
            "top5_with_dustbin": self.partial_top5_correct / denominator,
            "mrr_with_dustbin": self.partial_reciprocal_rank_sum / denominator,
            "dustbin_top1_rate": self.dustbin_top1 / denominator,
            "hungarian_queries": self.hungarian_queries,
            "hungarian_accuracy": self.hungarian_correct / h_denominator,
        }


def _directional_ranks(probabilities: torch.Tensor, targets: torch.Tensor) -> MetricTotals:
    """Rank known matches both with and without the dustbin candidate."""

    real_probabilities = probabilities[:, :-1]
    # Direct matches only; synthetic dustbin targets are training augmentation.
    valid = (targets >= 0) & (targets < real_probabilities.shape[1])
    if not bool(valid.any()):
        return MetricTotals()
    probabilities = probabilities[valid]
    real_probabilities = real_probabilities[valid]
    targets = targets[valid]
    target_score = real_probabilities.gather(1, targets[:, None])
    # Strictly greater scores define rank; ties receive the optimistic rank.
    real_rank = 1 + (real_probabilities > target_score).sum(dim=1)
    partial_rank = 1 + (probabilities > target_score).sum(dim=1)
    dustbin_is_top = probabilities[:, -1] > real_probabilities.amax(dim=1)
    return MetricTotals(
        queries=int(real_rank.numel()),
        real_top1_correct=int((real_rank <= 1).sum()),
        real_top5_correct=int((real_rank <= 5).sum()),
        real_reciprocal_rank_sum=float((1.0 / real_rank.float()).sum()),
        partial_top1_correct=int((partial_rank <= 1).sum()),
        partial_top5_correct=int((partial_rank <= 5).sum()),
        partial_reciprocal_rank_sum=float((1.0 / partial_rank.float()).sum()),
        dustbin_top1=int(dustbin_is_top.sum()),
    )


def _hungarian_counts(output: MPRTOutput, targets: PairTargets) -> tuple[int, int]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return 0, 0

    scores = output.plan[:-1, :-1].detach().cpu().numpy()
    row_index, col_index = linear_sum_assignment(-scores)
    assignment = {int(row): int(col) for row, col in zip(row_index, col_index)}
    inverse = {col: row for row, col in assignment.items()}

    row_target = targets.row_target.detach().cpu().numpy()
    col_target = targets.col_target.detach().cpu().numpy()
    num_a, num_b = scores.shape
    correct = total = 0
    for row, target in enumerate(row_target):
        if 0 <= target < num_b:
            total += 1
            correct += int(assignment.get(row, -1) == int(target))
    for col, target in enumerate(col_target):
        if 0 <= target < num_a:
            total += 1
            correct += int(inverse.get(col, -1) == int(target))
    return correct, total


def pair_metrics(output: MPRTOutput, targets: PairTargets) -> MetricTotals:
    totals = _directional_ranks(output.row_conditional.detach(), targets.row_target)
    totals.update(_directional_ranks(output.col_conditional.detach(), targets.col_target))
    h_correct, h_total = _hungarian_counts(output, targets)
    totals.hungarian_correct = h_correct
    totals.hungarian_queries = h_total
    return totals
