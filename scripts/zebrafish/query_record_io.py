"""Canonical per-query records for the zebrafish LOFO8 supplement.

The helpers in this module deliberately do not know about any model.  Native
evaluators pass one physical-pair score matrix and the canonical q/r candidate
metadata; this module applies the common rank, tie, and Hungarian conventions
and emits auditable rows.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


QUERY_COLUMNS = (
    "method", "fold", "seed", "pair_index", "pair_id", "direction",
    "query_uid", "reference_uid", "query_candidate_index", "query_id",
    "target_candidate_index", "target_id", "predicted_candidate_index",
    "predicted_id", "candidate_count", "candidate_order_sha256", "covered",
    "rank", "top1_correct", "top5_correct", "reciprocal_rank",
    "hungarian_candidate_index", "hungarian_id", "hungarian_correct",
)


def _strings(values: Iterable[Any]) -> list[str]:
    return [str(value).strip() for value in values]


def candidate_order_sha256(candidate_ids: Iterable[Any]) -> str:
    """Hash an ordered candidate list without delimiter ambiguity."""
    payload = json.dumps(
        _strings(candidate_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _direction_rows(
    *,
    method: str,
    fold: int,
    seed: int | None,
    pair_index: int,
    pair_id: str,
    direction: str,
    score: np.ndarray,
    query_uid: str,
    reference_uid: str,
    query_ids: Iterable[Any],
    reference_ids: Iterable[Any],
    targets: np.ndarray,
    assignment: dict[int, int],
    valid_score: np.ndarray | None,
    top1_from_prediction: bool,
) -> list[dict[str, Any]]:
    score = np.asarray(score, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    query_ids = _strings(query_ids)
    reference_ids = _strings(reference_ids)
    if score.shape != (len(query_ids), len(reference_ids)):
        raise ValueError(
            f"{pair_id}/{direction}: score shape {score.shape} != "
            f"{len(query_ids)}x{len(reference_ids)}"
        )
    if targets.shape != (len(query_ids),):
        raise ValueError(f"{pair_id}/{direction}: bad target shape {targets.shape}")
    if valid_score is None:
        valid_score = np.isfinite(score)
    else:
        valid_score = np.asarray(valid_score, dtype=bool)
        if valid_score.shape != score.shape:
            raise ValueError(f"{pair_id}/{direction}: bad valid-score shape")

    order_hash = candidate_order_sha256(reference_ids)
    rows: list[dict[str, Any]] = []
    for query_index in np.flatnonzero(
        (targets >= 0) & (targets < score.shape[1])
    ):
        target_index = int(targets[query_index])
        visible = bool(valid_score[query_index, target_index])
        usable = np.flatnonzero(valid_score[query_index])
        if len(usable):
            # np.argmax returns the first position, which is the declared
            # candidate-order tie break used for the archived prediction.
            predicted_index = int(usable[np.argmax(score[query_index, usable])])
            predicted_id = reference_ids[predicted_index]
        else:
            predicted_index = None
            predicted_id = ""

        if visible:
            target_score = score[query_index, target_index]
            rank = 1 + int(
                np.sum(
                    valid_score[query_index]
                    & (score[query_index] > target_score)
                )
            )
            rr = 1.0 / rank
        else:
            rank = None
            rr = 0.0

        hungarian_index = assignment.get(int(query_index))
        if hungarian_index is not None and not valid_score[query_index, hungarian_index]:
            hungarian_index = None
        rows.append({
            "method": method,
            "fold": int(fold),
            "seed": "" if seed is None else int(seed),
            "pair_index": int(pair_index),
            "pair_id": str(pair_id),
            "direction": direction,
            "query_uid": str(query_uid),
            "reference_uid": str(reference_uid),
            "query_candidate_index": int(query_index),
            "query_id": query_ids[query_index],
            "target_candidate_index": target_index,
            "target_id": reference_ids[target_index],
            "predicted_candidate_index": "" if predicted_index is None else predicted_index,
            "predicted_id": predicted_id,
            "candidate_count": len(reference_ids),
            "candidate_order_sha256": order_hash,
            "covered": int(visible),
            "rank": "" if rank is None else rank,
            "top1_correct": (
                int(predicted_index == target_index)
                if top1_from_prediction
                else (int(rank == 1) if rank is not None else 0)
            ),
            "top5_correct": int(rank <= min(5, score.shape[1])) if rank is not None else 0,
            "reciprocal_rank": rr,
            "hungarian_candidate_index": "" if hungarian_index is None else hungarian_index,
            "hungarian_id": "" if hungarian_index is None else reference_ids[hungarian_index],
            "hungarian_correct": int(hungarian_index == target_index),
        })
    return rows


def records_from_score_matrix(
    *,
    method: str,
    fold: int,
    seed: int | None,
    pair_index: int,
    pair_id: str,
    score: np.ndarray,
    q_uid: str,
    r_uid: str,
    q_ids: Iterable[Any],
    r_ids: Iterable[Any],
    row_target: np.ndarray,
    col_target: np.ndarray,
    valid_score: np.ndarray | None = None,
    reverse_score: np.ndarray | None = None,
    reverse_valid_score: np.ndarray | None = None,
    hungarian_score: np.ndarray | None = None,
    reverse_hungarian_score: np.ndarray | None = None,
    top1_from_prediction: bool = False,
) -> list[dict[str, Any]]:
    """Create both directed query sets from one physical-pair score matrix."""
    score = np.asarray(score, dtype=np.float64)
    if valid_score is None:
        valid_score = np.isfinite(score)
    else:
        valid_score = np.asarray(valid_score, dtype=bool)

    # Invalid entries receive a finite penalty only for scipy.  They remain
    # invalid in the emitted row and can never count as a correct assignment.
    reverse_score = score.T if reverse_score is None else np.asarray(reverse_score, dtype=np.float64)
    reverse_valid_score = (
        valid_score.T
        if reverse_valid_score is None
        else np.asarray(reverse_valid_score, dtype=bool)
    )
    hungarian_score = score if hungarian_score is None else np.asarray(hungarian_score, dtype=np.float64)
    if reverse_score.shape != score.T.shape or reverse_valid_score.shape != score.T.shape:
        raise ValueError(f"{pair_id}: reverse score shape mismatch")
    if hungarian_score.shape != score.shape:
        raise ValueError(f"{pair_id}: Hungarian score shape mismatch")

    finite = hungarian_score[valid_score]
    floor = (float(finite.min()) - max(1.0, float(np.ptp(finite)))) if len(finite) else -1e12
    scipy_score = np.where(valid_score, hungarian_score, floor)
    rr, cc = linear_sum_assignment(-scipy_score)
    assignment = {int(i): int(j) for i, j in zip(rr, cc)}
    inverse = {int(j): int(i) for i, j in zip(rr, cc)}
    if reverse_hungarian_score is not None:
        reverse_hungarian_score = np.asarray(reverse_hungarian_score, dtype=np.float64)
        if reverse_hungarian_score.shape != score.T.shape:
            raise ValueError(f"{pair_id}: reverse Hungarian score shape mismatch")
        finite_reverse = reverse_hungarian_score[reverse_valid_score]
        reverse_floor = (
            float(finite_reverse.min()) - max(1.0, float(np.ptp(finite_reverse)))
            if len(finite_reverse) else -1e12
        )
        reverse_scipy = np.where(reverse_valid_score, reverse_hungarian_score, reverse_floor)
        reverse_rows, reverse_cols = linear_sum_assignment(-reverse_scipy)
        inverse = {int(i): int(j) for i, j in zip(reverse_rows, reverse_cols)}

    rows = _direction_rows(
        method=method, fold=fold, seed=seed, pair_index=pair_index,
        pair_id=pair_id, direction="q_to_r", score=score, query_uid=q_uid,
        reference_uid=r_uid, query_ids=q_ids, reference_ids=r_ids,
        targets=row_target, assignment=assignment, valid_score=valid_score,
        top1_from_prediction=top1_from_prediction,
    )
    rows.extend(_direction_rows(
        method=method, fold=fold, seed=seed, pair_index=pair_index,
        pair_id=pair_id, direction="r_to_q", score=reverse_score, query_uid=r_uid,
        reference_uid=q_uid, query_ids=r_ids, reference_ids=q_ids,
        targets=col_target, assignment=inverse, valid_score=reverse_valid_score,
        top1_from_prediction=top1_from_prediction,
    ))
    return rows
