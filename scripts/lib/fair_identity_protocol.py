"""Identity-level evaluation for test-to-training-template matchers.

Pairwise methods predict neuron correspondences against every training animal.
This module maps those candidate scores to the union of training identities and
combines the repeated evidence without looking at test labels.  Test labels are
read only after the identity score matrix has been constructed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class IdentityAnimal:
    uid: str
    labels: tuple[str, ...]
    supervised_mask: np.ndarray


@dataclass(frozen=True)
class EnsembleResult:
    vocabulary: tuple[str, ...]
    scores: np.ndarray
    support_animals: np.ndarray
    votes: np.ndarray


@dataclass(frozen=True)
class GeometryMedoid:
    """Training-only medoid selected from normalized point-cloud geometry."""

    index: int
    uid: str
    mean_distance: float
    mean_distances: tuple[float, ...]
    pairwise_distances: np.ndarray


def _point_cloud(xyz: np.ndarray, uid: str) -> np.ndarray:
    value = np.asarray(xyz, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] < 3:
        raise ValueError(f"{uid}: xyz must be non-empty [N,>=3], got {value.shape}")
    value = value[:, :3]
    if not np.isfinite(value).all():
        raise ValueError(f"{uid}: xyz contains NaN/Inf")
    return value


def symmetric_chamfer_distance(xyz_a: np.ndarray, xyz_b: np.ndarray) -> float:
    """Mean bidirectional nearest-neighbour distance between two point clouds.

    Inputs are expected to have already undergone the fold's locked per-animal
    geometry normalization.  No identity labels or held-out data are used.
    """
    a = _point_cloud(xyz_a, "point_cloud_a")
    b = _point_cloud(xyz_b, "point_cloud_b")
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(0.5 * (distances.min(axis=1).mean() + distances.min(axis=0).mean()))


def select_geometry_medoid(
    uids: Sequence[str], point_clouds: Sequence[np.ndarray]
) -> GeometryMedoid:
    """Select the outer-training animal with minimum mean geometry distance.

    Ties are resolved deterministically by UID and then original list index.
    At least two training animals are required because a medoid cannot be
    defined from a self-distance alone.
    """
    if len(uids) != len(point_clouds):
        raise ValueError("uids/point_clouds length mismatch")
    if len(uids) < 2:
        raise ValueError("At least two outer-training animals are required")
    if len(set(uids)) != len(uids):
        raise ValueError("Outer-training animal UIDs must be unique")
    clouds = [_point_cloud(xyz, uid) for uid, xyz in zip(uids, point_clouds)]
    pairwise = np.zeros((len(clouds), len(clouds)), dtype=np.float64)
    for left in range(len(clouds)):
        for right in range(left + 1, len(clouds)):
            distance = symmetric_chamfer_distance(clouds[left], clouds[right])
            pairwise[left, right] = pairwise[right, left] = distance
    means = pairwise.sum(axis=1) / float(len(clouds) - 1)
    selected = min(range(len(clouds)), key=lambda i: (float(means[i]), uids[i], i))
    pairwise.setflags(write=False)
    return GeometryMedoid(
        index=selected,
        uid=uids[selected],
        mean_distance=float(means[selected]),
        mean_distances=tuple(float(value) for value in means),
        pairwise_distances=pairwise,
    )


def unique_identity_map(animal: IdentityAnimal) -> dict[str, int]:
    if len(animal.labels) != len(animal.supervised_mask):
        raise ValueError(f"{animal.uid}: labels/mask length mismatch")
    eligible = [
        label
        for label, keep in zip(animal.labels, animal.supervised_mask.tolist())
        if keep and label
    ]
    counts = Counter(eligible)
    return {
        label: index
        for index, (label, keep) in enumerate(
            zip(animal.labels, animal.supervised_mask.tolist())
        )
        if keep and label and counts[label] == 1
    }


def training_vocabulary(references: Sequence[IdentityAnimal]) -> tuple[str, ...]:
    """Sorted union of clean identities that are unique within a reference."""
    return tuple(sorted({key for animal in references for key in unique_identity_map(animal)}))


def row_softmax(scores: np.ndarray) -> np.ndarray:
    value = np.asarray(scores, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Pair scores must be a finite 2-D matrix")
    shifted = value - value.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def row_zscore(scores: np.ndarray) -> np.ndarray:
    value = np.asarray(scores, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Pair scores must be a finite 2-D matrix")
    return (value - value.mean(1, keepdims=True)) / np.maximum(
        value.std(1, keepdims=True), 1e-12
    )


def ensemble_pairwise_scores(
    query: IdentityAnimal,
    references: Sequence[IdentityAnimal],
    score_pair: Callable[[IdentityAnimal, IdentityAnimal], np.ndarray],
    *,
    normalization: str = "softmax",
) -> EnsembleResult:
    """Map every test-to-reference matrix into the training identity union.

    The primary score is the mean normalized score over training animals in
    which an identity has a unique clean representative.  Consequently an
    identity's frequency does not multiply its score.  ``votes`` is retained
    as a sensitivity-analysis output; it stores raw per-reference winner votes.
    """
    if not references:
        raise ValueError("At least one training reference is required")
    vocabulary = training_vocabulary(references)
    if not vocabulary:
        raise ValueError("Training references contain no evaluable identities")
    identity_column = {identity: index for index, identity in enumerate(vocabulary)}
    score_sum = np.zeros((len(query.labels), len(vocabulary)), dtype=np.float64)
    support = np.zeros(len(vocabulary), dtype=np.int64)
    votes = np.zeros_like(score_sum, dtype=np.float64)
    normalizer = {"softmax": row_softmax, "zscore": row_zscore}.get(normalization)
    if normalizer is None:
        raise ValueError("normalization must be 'softmax' or 'zscore'")

    for reference in references:
        raw = np.asarray(score_pair(query, reference), dtype=np.float64)
        expected = (len(query.labels), len(reference.labels))
        if raw.shape != expected:
            raise ValueError(
                f"{query.uid}->{reference.uid}: score shape {raw.shape}, expected {expected}"
            )
        normalized = normalizer(raw)
        reference_map = unique_identity_map(reference)
        if not reference_map:
            continue
        columns = np.fromiter(reference_map.values(), dtype=np.int64)
        identities = tuple(reference_map)
        global_columns = np.fromiter(
            (identity_column[identity] for identity in identities), dtype=np.int64
        )
        score_sum[:, global_columns] += normalized[:, columns]
        support[global_columns] += 1

        eligible = normalized[:, columns]
        maxima = eligible.max(axis=1, keepdims=True)
        tied_winners = eligible == maxima
        tied_winners = tied_winners / tied_winners.sum(axis=1, keepdims=True)
        votes[:, global_columns] += tied_winners

    if np.any(support == 0):
        raise AssertionError("Vocabulary contains an identity with zero support")
    mean_scores = score_sum / support[None, :]
    return EnsembleResult(vocabulary, mean_scores, support, votes)


def _tie_metrics(row: np.ndarray, target: int, k: int) -> tuple[float, float]:
    target_score = row[target]
    greater = int(np.count_nonzero(row > target_score))
    tied = int(np.count_nonzero(row == target_score))
    topk = max(0.0, min(1.0, (k - greater) / tied))
    reciprocal_rank = sum(
        1.0 / rank for rank in range(greater + 1, greater + tied + 1)
    ) / tied
    return topk, reciprocal_rank


def evaluate_identity_scores(
    query: IdentityAnimal,
    result: EnsembleResult,
    *,
    aggregation: str = "mean_score",
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    """Evaluate only clean test identities contained in the training vocabulary."""
    if aggregation == "mean_score":
        scores = result.scores
    elif aggregation == "vote":
        scores = result.votes.astype(np.float64)
    else:
        raise ValueError("aggregation must be 'mean_score' or 'vote'")
    vocabulary_index = {identity: index for index, identity in enumerate(result.vocabulary)}
    query_map = unique_identity_map(query)
    rows: list[dict[str, float | int | str]] = []
    valid_query_indices: list[int] = []
    valid_targets: list[int] = []
    for identity, query_index in query_map.items():
        if identity not in vocabulary_index:
            continue
        target = vocabulary_index[identity]
        top1, rr = _tie_metrics(scores[query_index], target, 1)
        top5, _ = _tie_metrics(scores[query_index], target, 5)
        greater = int(np.count_nonzero(scores[query_index] > scores[query_index, target]))
        tied = int(np.count_nonzero(scores[query_index] == scores[query_index, target]))
        rows.append(
            {
                "query_uid": query.uid,
                "query_index": query_index,
                "identity": identity,
                "candidate_count": len(result.vocabulary),
                "rank_min": greater + 1,
                "rank_max": greater + tied,
                "top1": top1,
                "top5": top5,
                "rr": rr,
                "predicted_identity": result.vocabulary[int(scores[query_index].argmax())],
            }
        )
        valid_query_indices.append(query_index)
        valid_targets.append(target)
    if not rows:
        raise ValueError(f"{query.uid}: no clean test identity occurs in the training vocabulary")

    assignment_rows, assignment_columns = linear_sum_assignment(
        -scores[np.asarray(valid_query_indices)]
    )
    assignment = {int(row): int(column) for row, column in zip(assignment_rows, assignment_columns)}
    hungarian_correct = sum(
        int(assignment.get(row, -1) == target)
        for row, target in enumerate(valid_targets)
    )
    metrics: dict[str, float | int] = {
        "queries": len(rows),
        "vocabulary_size": len(result.vocabulary),
        "top1": float(np.mean([float(row["top1"]) for row in rows])),
        "top5": float(np.mean([float(row["top5"]) for row in rows])),
        "mrr": float(np.mean([float(row["rr"]) for row in rows])),
        "hungarian_accuracy": hungarian_correct / len(rows),
    }
    return metrics, rows


def pool_metrics(metrics: Iterable[dict[str, float | int]]) -> dict[str, float | int]:
    values = list(metrics)
    queries = sum(int(item["queries"]) for item in values)
    if queries == 0:
        raise ValueError("Cannot pool zero queries")
    pooled: dict[str, float | int] = {"queries": queries}
    for key in ("top1", "top5", "mrr", "hungarian_accuracy"):
        pooled[key] = (
            sum(float(item[key]) * int(item["queries"]) for item in values) / queries
        )
    return pooled
