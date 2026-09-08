#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-protocol majority vote for the 21 freely moving Atanas worms.

This script reproduces Appendix F of:
"Unsupervised Neuronal Matching with Spontaneous Neuronal Activity".

For each of 1,000 UNIQUE random 9-teacher combinations:
  1. For every h in {0,5,...,50}, perform leave-one-out identification inside
     the 9 teachers: one worm is validation and the other 8 are teachers.
  2. Select h using the pooled validation accuracy at selection_v=5,
     selection_k=5 (smallest h breaks an exact tie).
  3. With the selected h, identify the remaining 12 test worms using all
     9 teachers.
  4. Each teacher casts v votes using its Top-v matched neurons. Votes from
     every candidate neuron are counted; noID occupies a Top-v position but is
     excluded from the final Top-k candidate labels.
  5. Report per-individual accuracies and their median, matching the paper's
     reporting language and boxplots.

Input
-----
One completed GWOT/GWOT-MD cache directory for every h. Each directory should
contain:
    pairs/<target>__to__<reference>/matching.npz

The cache loader is compatible with both the h=0 and GWOT-MD schemas used in
this project. It accepts gamma/conditional/coupling/transport matrices and can
infer worm names from NPZ keys, JSON sidecars, or directory names.

Exact-paper cache requirements (default)
----------------------------------------
Every h cache must have config.json showing:
  paper_mode=true
  21 epsilon values
  50 initializations
  lag_step=1
  normalize_lag_weights=false
  max_lag=h

Use --allow-non-paper-cache only for debugging reduced-search caches. Results
from that flag are NOT the paper-exact solver setting.

Example
-------
python -u paper_exact_h_selection_majority_vote.py \
  --run-template 'runs/atanas21_gwot_md_paper_h{h}' \
  --h-values 0,5,10,15,20,25,30,35,40,45,50 \
  --teacher-count 9 \
  --num-splits 1000 \
  --selection-v 5 \
  --selection-k 5 \
  --v-values 1,5,10 \
  --k-values 1,3,5 \
  --seed 42 \
  --output-dir runs/atanas21_paper_exact_majority_vote

Individual paths can override the template:
  --cache 0=runs/custom_h0 --cache 10=runs/custom_h10

Python 3.8 compatible.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


INVALID_LABELS = {
    "", "nan", "none", "null", "na", "n/a", "unknown",
    "unlabeled", "unlabelled", "-1", "-1.0", "noid",
}


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def decode(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value).strip()


def clean_labels(values: np.ndarray, exclude_uncertain: bool) -> np.ndarray:
    labels: List[str] = []
    for value in np.asarray(values).reshape(-1):
        text = decode(value)
        if text.lower() in INVALID_LABELS:
            text = ""
        if exclude_uncertain and "?" in text:
            text = ""
        labels.append(text)
    return np.asarray(labels, dtype=object)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=sums > 0,
    )


def parse_int_list(text: str, allow_zero: bool = False) -> List[int]:
    values = sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))
    lower = 0 if allow_zero else 1
    if not values or any(x < lower for x in values):
        requirement = "non-negative" if allow_zero else "positive"
        raise ValueError("Expected comma-separated {} integers.".format(requirement))
    return values


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("Could not read JSON {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object in {}".format(path))
    return value


# -----------------------------------------------------------------------------
# Flexible matching-cache loader
# -----------------------------------------------------------------------------


def first_npz_key(data: Any, aliases: Sequence[str]) -> Optional[str]:
    lower_to_real = {str(key).lower(): str(key) for key in data.files}
    for alias in aliases:
        real = lower_to_real.get(alias.lower())
        if real is not None:
            return real
    return None


def recursive_json_value(obj: Any, aliases: Sequence[str]) -> Optional[Any]:
    wanted = {x.lower() for x in aliases}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in wanted:
                return value
        for value in obj.values():
            found = recursive_json_value(value, aliases)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = recursive_json_value(value, aliases)
            if found is not None:
                return found
    return None


def canonical_worm_name(value: Any) -> str:
    text = decode(value)
    match = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}", text)
    return match.group(0) if match else text


def infer_worm_names(path: Path, data: Any) -> Tuple[str, str]:
    row_aliases = (
        "target_worm", "source_worm", "query_worm", "row_worm",
        "worm_a", "worm1", "target_name", "source_name", "query_name",
        "target_recording", "source_recording", "recording_a",
    )
    col_aliases = (
        "reference_worm", "ref_worm", "candidate_worm", "column_worm",
        "worm_b", "worm2", "reference_name", "ref_name", "candidate_name",
        "reference_recording", "recording_b",
    )

    row_key = first_npz_key(data, row_aliases)
    col_key = first_npz_key(data, col_aliases)
    if row_key is not None and col_key is not None:
        return canonical_worm_name(data[row_key]), canonical_worm_name(data[col_key])

    for json_path in sorted(path.parent.glob("*.json")):
        obj = read_json(json_path)
        if obj is None:
            continue
        row = recursive_json_value(obj, row_aliases)
        col = recursive_json_value(obj, col_aliases)
        if row is not None and col is not None:
            return canonical_worm_name(row), canonical_worm_name(col)

    recordings = re.findall(r"\d{4}-\d{2}-\d{2}-\d{2}", str(path))
    unique: List[str] = []
    for recording in recordings:
        if recording not in unique:
            unique.append(recording)
    if len(unique) >= 2:
        return unique[-2], unique[-1]

    raise KeyError(
        "Cannot infer target/reference worms for {}. NPZ keys: {}".format(
            path, sorted(data.files)
        )
    )


def load_score_matrix(path: Path, data: Any) -> np.ndarray:
    aliases = (
        "conditional", "gamma", "coupling", "transport", "transport_matrix",
        "transport_plan", "matching", "matching_matrix", "score_matrix",
        "probability", "probabilities", "best_gamma", "best_transport", "P",
    )
    key = first_npz_key(data, aliases)
    if key is None:
        candidates: List[str] = []
        for name in data.files:
            array = np.asarray(data[name])
            if array.ndim == 2 and np.issubdtype(array.dtype, np.number):
                candidates.append(name)
        if len(candidates) != 1:
            raise KeyError(
                "{}: cannot identify score matrix; 2-D numeric candidates={}".format(
                    path, candidates
                )
            )
        key = candidates[0]
    return np.asarray(data[key], dtype=np.float64)


def load_label_arrays(
    path: Path,
    data: Any,
    matrix_shape: Tuple[int, int],
    exclude_uncertain: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    row_aliases = (
        "target_labels", "source_labels", "query_labels", "row_labels",
        "labels_a", "labels1", "target_ids", "source_ids", "query_ids",
    )
    col_aliases = (
        "reference_labels", "ref_labels", "candidate_labels", "column_labels",
        "labels_b", "labels2", "reference_ids", "ref_ids", "candidate_ids",
    )
    row_key = first_npz_key(data, row_aliases)
    col_key = first_npz_key(data, col_aliases)
    if row_key is not None and col_key is not None:
        return (
            clean_labels(data[row_key], exclude_uncertain),
            clean_labels(data[col_key], exclude_uncertain),
        )

    vectors: List[Tuple[str, np.ndarray]] = []
    for name in data.files:
        array = np.asarray(data[name])
        if array.ndim == 1 and array.size > 1 and array.dtype.kind in {"U", "S", "O"}:
            vectors.append((name, clean_labels(array, exclude_uncertain)))

    rows = [(name, arr) for name, arr in vectors if len(arr) == matrix_shape[0]]
    cols = [(name, arr) for name, arr in vectors if len(arr) == matrix_shape[1]]

    def row_priority(name: str) -> Tuple[int, str]:
        low = name.lower()
        return (
            0 if any(x in low for x in ("target", "source", "query", "row", "_a")) else 1,
            name,
        )

    def col_priority(name: str) -> Tuple[int, str]:
        low = name.lower()
        return (
            0 if any(x in low for x in ("reference", "ref", "candidate", "column", "_b")) else 1,
            name,
        )

    rows.sort(key=lambda item: row_priority(item[0]))
    cols.sort(key=lambda item: col_priority(item[0]))
    if rows and cols and rows[0][0] != cols[0][0]:
        return rows[0][1], cols[0][1]

    raise KeyError(
        "{}: cannot identify target/reference labels. Matrix shape={}, vectors={}".format(
            path, matrix_shape, [(name, len(arr)) for name, arr in vectors]
        )
    )


def validate_paper_cache(run_dir: Path, h: int, allow_non_paper: bool) -> None:
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "summary.json")

    if allow_non_paper:
        return
    if config is None:
        raise ValueError(
            "{} has no config.json. Exact mode requires proof that --paper_mode was used. "
            "Use --allow-non-paper-cache only for debugging.".format(run_dir)
        )

    problems: List[str] = []
    if not bool(config.get("paper_mode", False)):
        problems.append("paper_mode is not true")

    resolved_eps = config.get("resolved_epsilons")
    if not isinstance(resolved_eps, list) or len(resolved_eps) != 21:
        problems.append("resolved_epsilons does not contain 21 values")

    if int(config.get("resolved_n_inits", -1)) != 50:
        problems.append("resolved_n_inits is not 50")

    max_lag = config.get("max_lag", config.get("max_lag_frames"))
    if max_lag is None or int(max_lag) != int(h):
        problems.append("max_lag={} but expected {}".format(max_lag, h))

    if int(config.get("lag_step", -1)) != 1:
        problems.append("lag_step is not 1")
    if bool(config.get("normalize_lag_weights", False)):
        problems.append("normalize_lag_weights must be false")

    cutoff = config.get("highpass_cutoff")
    if cutoff is None or not math.isclose(float(cutoff), 0.01, rel_tol=0.0, abs_tol=1e-12):
        problems.append("highpass_cutoff is not 0.01 Hz")
    if int(config.get("highpass_order", -1)) != 1:
        problems.append("highpass_order is not 1")
    if str(config.get("trace_normalization", "")).lower() != "none":
        problems.append("trace_normalization is not none")

    if summary is not None and not bool(summary.get("paper_mode", True)):
        problems.append("summary.json records paper_mode=false")

    if problems:
        raise ValueError(
            "Cache {} is not paper-exact:\n  - {}".format(
                run_dir, "\n  - ".join(problems)
            )
        )


def load_cache(
    run_dir: Path,
    exclude_uncertain: bool,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[str, np.ndarray], List[str]]:
    paths = sorted((run_dir / "pairs").glob("*/matching.npz"))
    if not paths:
        paths = sorted(run_dir.glob("**/matching.npz"))
    if not paths:
        raise FileNotFoundError("No matching.npz found under {}".format(run_dir))

    explicit_scores: Dict[Tuple[str, str], np.ndarray] = {}
    labels_by_worm: Dict[str, np.ndarray] = {}

    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            raw = load_score_matrix(path, data)
            target_labels, reference_labels = load_label_arrays(
                path, data, tuple(raw.shape), exclude_uncertain
            )
            target, reference = infer_worm_names(path, data)

        if raw.shape == (len(target_labels), len(reference_labels)):
            forward = row_normalize(raw)
        elif raw.T.shape == (len(target_labels), len(reference_labels)):
            forward = row_normalize(raw.T)
        else:
            raise ValueError(
                "{} matrix shape {} incompatible with label lengths ({}, {})".format(
                    path, raw.shape, len(target_labels), len(reference_labels)
                )
            )

        key = (target, reference)
        if key in explicit_scores:
            old = explicit_scores[key]
            if old.shape != forward.shape or not np.allclose(old, forward, rtol=1e-6, atol=1e-9):
                raise ValueError("Conflicting duplicate direction {} -> {}".format(*key))
        else:
            explicit_scores[key] = forward

        for worm, labels in ((target, target_labels), (reference, reference_labels)):
            if worm in labels_by_worm:
                old = labels_by_worm[worm]
                if len(old) != len(labels) or not np.array_equal(old, labels):
                    raise ValueError("Inconsistent labels/neuron ordering for {}".format(worm))
            else:
                labels_by_worm[worm] = labels

    scores = dict(explicit_scores)
    for (target, reference), matrix in list(explicit_scores.items()):
        reverse = (reference, target)
        if reverse not in scores:
            scores[reverse] = row_normalize(matrix.T)

    worms = sorted(labels_by_worm)
    expected = len(worms) * (len(worms) - 1)
    missing = [
        (a, b) for a in worms for b in worms
        if a != b and (a, b) not in scores
    ]
    if missing:
        raise ValueError(
            "{} is incomplete: {} of {} directed matchings are missing.".format(
                run_dir, len(missing), expected
            )
        )

    print(
        "  loaded {} files, {} explicit and {} usable directions".format(
            len(paths), len(explicit_scores), len(scores)
        ),
        flush=True,
    )
    return scores, labels_by_worm, worms


# -----------------------------------------------------------------------------
# Compact Top-v cache
# -----------------------------------------------------------------------------


def build_label_vocabulary(labels_by_worm: Mapping[str, np.ndarray]) -> Tuple[List[str], Dict[str, int]]:
    labels = sorted({str(x) for values in labels_by_worm.values() for x in values if str(x)})
    return labels, {label: index for index, label in enumerate(labels)}


def labels_to_ids(labels: np.ndarray, label_to_id: Mapping[str, int]) -> np.ndarray:
    return np.asarray(
        [label_to_id.get(str(label), -1) if str(label) else -1 for label in labels],
        dtype=np.int16,
    )


def precompute_top_label_ids(
    cache_dirs: Mapping[int, Path],
    max_v: int,
    exclude_uncertain: bool,
    allow_non_paper: bool,
) -> Tuple[
    Dict[int, Dict[Tuple[str, str], np.ndarray]],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    List[str],
    List[str],
]:
    top_by_h: Dict[int, Dict[Tuple[str, str], np.ndarray]] = {}
    canonical_labels: Optional[Dict[str, np.ndarray]] = None
    canonical_worms: Optional[List[str]] = None
    vocabulary: Optional[List[str]] = None
    label_to_id: Optional[Dict[str, int]] = None
    label_ids_by_worm: Optional[Dict[str, np.ndarray]] = None

    for h in sorted(cache_dirs):
        run_dir = cache_dirs[h]
        print("Loading h={} cache: {}".format(h, run_dir), flush=True)
        validate_paper_cache(run_dir, h, allow_non_paper)
        scores, labels_by_worm, worms = load_cache(run_dir, exclude_uncertain)

        if canonical_labels is None:
            canonical_labels = {worm: labels.copy() for worm, labels in labels_by_worm.items()}
            canonical_worms = list(worms)
            vocabulary, label_to_id = build_label_vocabulary(canonical_labels)
            label_ids_by_worm = {
                worm: labels_to_ids(labels, label_to_id)
                for worm, labels in canonical_labels.items()
            }
        else:
            if worms != canonical_worms:
                raise ValueError("Worm list differs at h={}".format(h))
            for worm in worms:
                if not np.array_equal(labels_by_worm[worm], canonical_labels[worm]):
                    raise ValueError("Labels/neuron order differs for {} at h={}".format(worm, h))

        assert label_ids_by_worm is not None
        compact: Dict[Tuple[str, str], np.ndarray] = {}
        for (test_worm, teacher_worm), matrix in scores.items():
            v_eff = min(max_v, matrix.shape[1])
            # Stable full sorting avoids argpartition ambiguity at equal scores.
            indices = np.argsort(-matrix, axis=1, kind="mergesort")[:, :v_eff]
            teacher_ids = label_ids_by_worm[teacher_worm]
            compact[(test_worm, teacher_worm)] = teacher_ids[indices]
        top_by_h[h] = compact
        del scores

    assert canonical_labels is not None
    assert canonical_worms is not None
    assert vocabulary is not None
    assert label_ids_by_worm is not None
    return top_by_h, canonical_labels, label_ids_by_worm, canonical_worms, vocabulary


# -----------------------------------------------------------------------------
# Paper majority vote and evaluation
# -----------------------------------------------------------------------------


def query_ranks_from_votes(
    candidate_ids: np.ndarray,
    true_ids: np.ndarray,
    num_labels: int,
    tie_break: str,
) -> np.ndarray:
    """
    candidate_ids: [N_query, M*v], -1 denotes noID.
    true_ids:      [N_query], -1 denotes an unlabeled query.

    Every candidate occurrence casts one vote, exactly as Appendix F. noID has
    already consumed a Top-v slot but is excluded from the final candidate list.
    Labels with zero votes are excluded. Equal vote counts use first appearance
    order by default, matching Python Counter.most_common stable behavior.
    """
    n_query, width = candidate_ids.shape
    counts = np.zeros((n_query, num_labels), dtype=np.int16)
    rows = np.broadcast_to(np.arange(n_query)[:, None], candidate_ids.shape)
    valid = candidate_ids >= 0
    np.add.at(counts, (rows[valid], candidate_ids[valid]), 1)

    ranks = np.full(n_query, np.inf, dtype=np.float64)
    labeled_rows = np.flatnonzero(true_ids >= 0)
    if labeled_rows.size == 0:
        return ranks

    true = true_ids[labeled_rows].astype(np.int64)
    true_counts = counts[labeled_rows, true]
    present = true_counts > 0
    if not np.any(present):
        return ranks

    active_rows = labeled_rows[present]
    active_true = true[present]
    active_counts = counts[active_rows]
    active_true_counts = true_counts[present]

    greater = np.sum(active_counts > active_true_counts[:, None], axis=1)

    if tie_break == "label":
        label_ids = np.arange(num_labels)[None, :]
        before = np.sum(
            (active_counts == active_true_counts[:, None])
            & (label_ids < active_true[:, None]),
            axis=1,
        )
    elif tie_break == "first":
        sentinel = width + 1
        first = np.full((n_query, num_labels), sentinel, dtype=np.int16)
        positions = np.broadcast_to(np.arange(width, dtype=np.int16)[None, :], candidate_ids.shape)
        np.minimum.at(first, (rows[valid], candidate_ids[valid]), positions[valid])
        active_first = first[active_rows]
        true_first = active_first[np.arange(len(active_rows)), active_true]
        before = np.sum(
            (active_counts == active_true_counts[:, None])
            & (active_first < true_first[:, None]),
            axis=1,
        )
    else:
        raise ValueError("Unsupported tie_break {}".format(tie_break))

    ranks[active_rows] = 1.0 + greater + before
    return ranks


def evaluate_one_worm(
    h_top: Mapping[Tuple[str, str], np.ndarray],
    test_worm: str,
    teachers: Sequence[str],
    true_ids: np.ndarray,
    label_presence: np.ndarray,
    worm_to_index: Mapping[str, int],
    v: int,
    k_values: Sequence[int],
    num_labels: int,
    tie_break: str,
) -> Dict[str, Any]:
    candidate_blocks = [h_top[(test_worm, teacher)][:, :v] for teacher in teachers]
    # Paper order: teacher 1 Top-v, teacher 2 Top-v, ...
    candidates = np.concatenate(candidate_blocks, axis=1)
    ranks = query_ranks_from_votes(candidates, true_ids, num_labels, tie_break)

    labeled = true_ids >= 0
    query_indices = np.flatnonzero(labeled)
    valid_true = true_ids[labeled].astype(np.int64)
    teacher_indices = np.asarray([worm_to_index[x] for x in teachers], dtype=np.int64)
    coverage_counts = label_presence[teacher_indices][:, valid_true].sum(axis=0)
    covered = coverage_counts > 0

    result: Dict[str, Any] = {
        "test_worm": test_worm,
        "num_queries": int(np.sum(labeled)),
        "num_covered_queries": int(np.sum(covered)),
        "coverage": float(np.mean(covered)) if covered.size else math.nan,
        "mean_teacher_coverage_count": float(np.mean(coverage_counts)) if coverage_counts.size else math.nan,
    }

    labeled_ranks = ranks[labeled]
    for k in k_values:
        hit = labeled_ranks <= k
        result["top{}_correct".format(k)] = int(np.sum(hit))
        result["top{}_accuracy".format(k)] = float(np.mean(hit)) if hit.size else math.nan
        result["top{}_covered_correct".format(k)] = int(np.sum(hit & covered))
        result["top{}_covered_accuracy".format(k)] = (
            float(np.mean(hit[covered])) if np.any(covered) else math.nan
        )
    return result


def evaluate_worm_group(
    h_top: Mapping[Tuple[str, str], np.ndarray],
    test_worms: Sequence[str],
    teachers: Sequence[str],
    label_ids_by_worm: Mapping[str, np.ndarray],
    label_presence: np.ndarray,
    worm_to_index: Mapping[str, int],
    v: int,
    k_values: Sequence[int],
    num_labels: int,
    tie_break: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    individual: List[Dict[str, Any]] = []
    for test_worm in test_worms:
        individual.append(
            evaluate_one_worm(
                h_top=h_top,
                test_worm=test_worm,
                teachers=teachers,
                true_ids=label_ids_by_worm[test_worm],
                label_presence=label_presence,
                worm_to_index=worm_to_index,
                v=v,
                k_values=k_values,
                num_labels=num_labels,
                tie_break=tie_break,
            )
        )

    total = sum(int(row["num_queries"]) for row in individual)
    covered_total = sum(int(row["num_covered_queries"]) for row in individual)
    group: Dict[str, Any] = {
        "num_test_worms": len(test_worms),
        "num_queries": total,
        "num_covered_queries": covered_total,
        "coverage": float(covered_total / total) if total else math.nan,
    }
    for k in k_values:
        correct = sum(int(row["top{}_correct".format(k)]) for row in individual)
        covered_correct = sum(int(row["top{}_covered_correct".format(k)]) for row in individual)
        individual_values = np.asarray(
            [float(row["top{}_accuracy".format(k)]) for row in individual],
            dtype=np.float64,
        )
        group["top{}_correct".format(k)] = correct
        group["top{}_micro".format(k)] = float(correct / total) if total else math.nan
        group["top{}_individual_mean".format(k)] = float(np.mean(individual_values))
        group["top{}_individual_median".format(k)] = float(np.median(individual_values))
        group["top{}_covered".format(k)] = (
            float(covered_correct / covered_total) if covered_total else math.nan
        )
    return individual, group


def sample_unique_teacher_sets(
    worms: Sequence[str],
    teacher_count: int,
    num_splits: int,
    seed: int,
) -> List[Tuple[str, ...]]:
    total_combinations = math.comb(len(worms), teacher_count)
    if num_splits > total_combinations:
        raise ValueError(
            "Requested {} splits but only {} unique teacher combinations exist.".format(
                num_splits, total_combinations
            )
        )

    rng = np.random.default_rng(seed)
    selected: Dict[Tuple[int, ...], None] = {}
    while len(selected) < num_splits:
        indices = tuple(sorted(int(x) for x in rng.choice(
            len(worms), size=teacher_count, replace=False
        )))
        selected.setdefault(indices, None)
    return [tuple(worms[i] for i in indices) for indices in selected.keys()]


def aggregate_outer_results(
    individual_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    v_values: Sequence[int],
    k_values: Sequence[int],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for v in v_values:
        individuals = [row for row in individual_rows if int(row["v"]) == v]
        splits = [row for row in split_rows if int(row["v"]) == v]
        for k in k_values:
            indiv_values = np.asarray(
                [float(row["top{}_accuracy".format(k)]) for row in individuals],
                dtype=np.float64,
            )
            split_micro = np.asarray(
                [float(row["top{}_micro".format(k)]) for row in splits],
                dtype=np.float64,
            )
            split_indiv_mean = np.asarray(
                [float(row["top{}_individual_mean".format(k)]) for row in splits],
                dtype=np.float64,
            )
            row = {
                "v": v,
                "k": k,
                "num_individual_evaluations": len(indiv_values),
                # Closest to the paper's Table 2 / Appendix F reporting.
                "paper_median_individual_accuracy": float(np.median(indiv_values)),
                "individual_mean": float(np.mean(indiv_values)),
                "individual_std": float(np.std(indiv_values, ddof=1)),
                "individual_q025": float(np.quantile(indiv_values, 0.025)),
                "individual_q25": float(np.quantile(indiv_values, 0.25)),
                "individual_q75": float(np.quantile(indiv_values, 0.75)),
                "individual_q975": float(np.quantile(indiv_values, 0.975)),
                "split_micro_mean": float(np.mean(split_micro)),
                "split_micro_std": float(np.std(split_micro, ddof=1)),
                "split_micro_median": float(np.median(split_micro)),
                "split_individual_mean_mean": float(np.mean(split_indiv_mean)),
                "split_individual_mean_median": float(np.median(split_indiv_mean)),
            }
            output.append(row)
    return output


# -----------------------------------------------------------------------------
# CLI and main protocol
# -----------------------------------------------------------------------------


def parse_cache_overrides(items: Sequence[str]) -> Dict[int, Path]:
    output: Dict[int, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--cache expects H=PATH, got {!r}".format(item))
        h_text, path_text = item.split("=", 1)
        h = int(h_text.strip())
        output[h] = Path(path_text.strip()).expanduser().resolve()
    return output


def build_cache_dirs(
    h_values: Sequence[int],
    run_template: str,
    overrides: Mapping[int, Path],
) -> Dict[int, Path]:
    cache_dirs: Dict[int, Path] = {}
    for h in h_values:
        if h in overrides:
            cache_dirs[h] = overrides[h]
        else:
            cache_dirs[h] = Path(run_template.format(h=h)).expanduser().resolve()
        if not cache_dirs[h].exists():
            raise FileNotFoundError("Missing h={} cache directory: {}".format(h, cache_dirs[h]))
    return cache_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-exact h-selection and majority vote for Atanas21 GWOT-MD caches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-template",
        default="runs/atanas21_gwot_md_paper_h{h}",
        help="Cache path template. {h} is replaced by each h value.",
    )
    parser.add_argument(
        "--cache",
        action="append",
        default=[],
        help="Override a cache path as H=PATH; may be repeated.",
    )
    parser.add_argument("--h-values", default="0,5,10,15,20,25,30,35,40,45,50")
    parser.add_argument("--teacher-count", type=int, default=9)
    parser.add_argument("--num-splits", type=int, default=1000)
    parser.add_argument("--selection-v", type=int, default=5)
    parser.add_argument("--selection-k", type=int, default=5)
    parser.add_argument("--v-values", default="1,5,10")
    parser.add_argument("--k-values", default="1,3,5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tie-break", choices=["first", "label"], default="first")
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument(
        "--allow-non-paper-cache",
        action="store_true",
        help="Permit reduced-search caches for debugging; output is not paper-exact.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/atanas21_paper_exact_majority_vote",
    )
    args = parser.parse_args()

    h_values = parse_int_list(args.h_values, allow_zero=True)
    v_values = parse_int_list(args.v_values)
    k_values = parse_int_list(args.k_values)
    if args.selection_v <= 0 or args.selection_k <= 0:
        raise ValueError("selection-v and selection-k must be positive.")
    max_v = max(max(v_values), args.selection_v)

    overrides = parse_cache_overrides(args.cache)
    cache_dirs = build_cache_dirs(h_values, args.run_template, overrides)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    top_by_h, labels_by_worm, label_ids_by_worm, worms, vocabulary = precompute_top_label_ids(
        cache_dirs=cache_dirs,
        max_v=max_v,
        exclude_uncertain=not args.include_uncertain,
        allow_non_paper=args.allow_non_paper_cache,
    )

    if args.teacher_count != 9:
        print("WARNING: the paper uses exactly 9 teacher worms.", flush=True)
    if len(worms) != 21:
        print("WARNING: the freely moving paper experiment uses exactly 21 worms.", flush=True)
    if not (1 <= args.teacher_count < len(worms)):
        raise ValueError("teacher-count must be in [1, num_worms-1].")

    worm_to_index = {worm: index for index, worm in enumerate(worms)}
    label_presence = np.zeros((len(worms), len(vocabulary)), dtype=bool)
    for worm, ids in label_ids_by_worm.items():
        valid = ids[ids >= 0]
        label_presence[worm_to_index[worm], valid] = True

    teacher_sets = sample_unique_teacher_sets(
        worms=worms,
        teacher_count=args.teacher_count,
        num_splits=args.num_splits,
        seed=args.seed,
    )

    inner_rows: List[Dict[str, Any]] = []
    selected_h_rows: List[Dict[str, Any]] = []
    outer_individual_rows: List[Dict[str, Any]] = []
    outer_split_rows: List[Dict[str, Any]] = []

    all_worms_set = set(worms)

    for split_index, teachers_tuple in enumerate(teacher_sets):
        teachers = list(teachers_tuple)
        tests = sorted(all_worms_set.difference(teachers))

        best_h: Optional[int] = None
        best_accuracy = -math.inf

        for h in h_values:
            total_correct = 0
            total_queries = 0
            validation_accuracies: List[float] = []

            for validation_worm in teachers:
                inner_teachers = [worm for worm in teachers if worm != validation_worm]
                individual, group = evaluate_worm_group(
                    h_top=top_by_h[h],
                    test_worms=[validation_worm],
                    teachers=inner_teachers,
                    label_ids_by_worm=label_ids_by_worm,
                    label_presence=label_presence,
                    worm_to_index=worm_to_index,
                    v=args.selection_v,
                    k_values=[args.selection_k],
                    num_labels=len(vocabulary),
                    tie_break=args.tie_break,
                )
                row = individual[0]
                total_correct += int(row["top{}_correct".format(args.selection_k)])
                total_queries += int(row["num_queries"])
                validation_accuracies.append(float(row["top{}_accuracy".format(args.selection_k)]))

            accuracy = float(total_correct / total_queries) if total_queries else math.nan
            inner_rows.append({
                "split": split_index,
                "h": h,
                "selection_v": args.selection_v,
                "selection_k": args.selection_k,
                "num_validation_worms": len(teachers),
                "num_queries": total_queries,
                "correct": total_correct,
                "micro_accuracy": accuracy,
                "individual_mean_accuracy": float(np.mean(validation_accuracies)),
                "individual_median_accuracy": float(np.median(validation_accuracies)),
                "teachers": "|".join(teachers),
            })

            # Paper does not specify tie handling for h; iterating in ascending h
            # and updating only on strict improvement selects the smallest tied h.
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_h = h

        assert best_h is not None
        selected_h_rows.append({
            "split": split_index,
            "selected_h": best_h,
            "inner_micro_accuracy": best_accuracy,
            "selection_v": args.selection_v,
            "selection_k": args.selection_k,
            "teachers": "|".join(teachers),
            "tests": "|".join(tests),
        })

        for v in v_values:
            individual, group = evaluate_worm_group(
                h_top=top_by_h[best_h],
                test_worms=tests,
                teachers=teachers,
                label_ids_by_worm=label_ids_by_worm,
                label_presence=label_presence,
                worm_to_index=worm_to_index,
                v=v,
                k_values=k_values,
                num_labels=len(vocabulary),
                tie_break=args.tie_break,
            )

            for row in individual:
                row.update({
                    "split": split_index,
                    "selected_h": best_h,
                    "v": v,
                    "teacher_count": len(teachers),
                    "teachers": "|".join(teachers),
                })
                outer_individual_rows.append(row)

            group.update({
                "split": split_index,
                "selected_h": best_h,
                "v": v,
                "teacher_count": len(teachers),
                "test_count": len(tests),
                "teachers": "|".join(teachers),
                "tests": "|".join(tests),
            })
            outer_split_rows.append(group)

        if (split_index + 1) % max(1, args.num_splits // 20) == 0 or split_index + 1 == args.num_splits:
            print(
                "Completed {}/{} splits; latest selected h={}, inner Top-{}={:.4f}".format(
                    split_index + 1,
                    args.num_splits,
                    best_h,
                    args.selection_k,
                    best_accuracy,
                ),
                flush=True,
            )
            # Checkpoint after every progress interval.
            write_csv(output_dir / "inner_validation.csv", inner_rows)
            write_csv(output_dir / "selected_h.csv", selected_h_rows)
            write_csv(output_dir / "individual_results.csv", outer_individual_rows)
            write_csv(output_dir / "split_results.csv", outer_split_rows)

    aggregate_rows = aggregate_outer_results(
        individual_rows=outer_individual_rows,
        split_rows=outer_split_rows,
        v_values=v_values,
        k_values=k_values,
    )

    h_counter = Counter(int(row["selected_h"]) for row in selected_h_rows)
    h_distribution = [
        {
            "h": h,
            "count": int(h_counter.get(h, 0)),
            "fraction": float(h_counter.get(h, 0) / args.num_splits),
        }
        for h in h_values
    ]

    write_csv(output_dir / "aggregate.csv", aggregate_rows)
    write_csv(output_dir / "selected_h_distribution.csv", h_distribution)

    summary = {
        "protocol": "paper Appendix F: 9-teacher inner LOO h selection then outer majority vote",
        "paper_exact_cache_validation": not args.allow_non_paper_cache,
        "num_worms": len(worms),
        "teacher_count": args.teacher_count,
        "test_count": len(worms) - args.teacher_count,
        "num_unique_splits": args.num_splits,
        "h_values": h_values,
        "selection_v": args.selection_v,
        "selection_k": args.selection_k,
        "v_values": v_values,
        "k_values": k_values,
        "tie_break": args.tie_break,
        "seed": args.seed,
        "cache_dirs": {str(h): str(path) for h, path in cache_dirs.items()},
        "selected_h_distribution": h_distribution,
        "aggregate": aggregate_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print("\nSelected-h distribution:")
    for row in h_distribution:
        print("h={:2d}: {:4d} ({:.1%})".format(row["h"], row["count"], row["fraction"]))

    print("\nPaper-style aggregate results (median per-individual accuracy):")
    for row in aggregate_rows:
        print(
            "v={} k={} median={:.4f} mean={:.4f} split_micro={:.4f}±{:.4f}".format(
                row["v"],
                row["k"],
                row["paper_median_individual_accuracy"],
                row["individual_mean"],
                row["split_micro_mean"],
                row["split_micro_std"],
            )
        )

    paper_rows = [row for row in aggregate_rows if int(row["v"]) == 5 and int(row["k"]) == 5]
    if paper_rows:
        row = paper_rows[0]
        print("\nPaper headline setting: v=5, k=5")
        print(
            "Median per-individual Top-5: {:.4f} ({:.2f}%)".format(
                row["paper_median_individual_accuracy"],
                100.0 * row["paper_median_individual_accuracy"],
            )
        )
        print(
            "Mean per-individual Top-5:   {:.4f} ± {:.4f}".format(
                row["individual_mean"], row["individual_std"]
            )
        )

    print("\nSaved to: {}".format(output_dir))


if __name__ == "__main__":
    main()
