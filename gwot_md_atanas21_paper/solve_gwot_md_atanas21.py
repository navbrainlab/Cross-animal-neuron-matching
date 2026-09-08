#!/usr/bin/env python3
"""Generate paper-protocol GWOT-MD pair caches for the Atanas-21 cohort.

The script implements the pairwise stage described in Algorithms 1--2 and
Appendix B of "Unsupervised Neuronal Matching with Spontaneous Neuronal
Activity".  It deliberately does not reuse the 38-worm CV5 baseline: that
runner uses a different cohort, distance, and optimizer.

Strict-mode invariants
----------------------
* the 21 baseline+NeuroPAL Atanas recordings listed in ``PAPER_COHORT``;
* all recorded neurons (no geometry/coordinate validity filtering);
* first-order 0.01-Hz Butterworth high-pass filtering;
* delayed cosine relations at every integer lag in [-h, h], w_tau=1;
* h in {0,5,...,50}; uniform neuron masses;
* epsilon in {10^-4, 10^-3.8, ..., 10^-0.2, 1};
* 50 random feasible initial couplings per epsilon;
* entropic GWOT-MD fixed-point updates with log-domain Sinkhorn;
* selection among 1,050 converged couplings by the unregularized objective.

The paper does not publish its random seeds, random-plan construction,
Sinkhorn/outer stopping settings, filtering direction, or source code.  Those
otherwise-unrecoverable choices are explicitly locked here to the conventions
used by the Oizumi-lab GWTune implementation (seeds 0..49, random-uniform
matrix followed by alternating marginal normalization, 1e-9 outer tolerance,
and 1,000 iterations), plus zero-phase filtering and log Sinkhorn.  They are
written to every config.json; changing one makes a run non-paper-mode.

Input formats
-------------
1. NPZ (recommended for already prepared data): one file per recording.  It
   must contain a 2-D activity array and a same-length label vector.  Accepted
   activity keys include ``activity_raw``, ``traces_array_F_F20``, and
   ``trace_array_original``.  Accepted label keys include ``cell_id`` and
   ``labels``.  No xyz field is read.
2. Official Atanas HDF5 files plus a label JSON/JSON.bz2.  HDF5 loading needs
   h5py.  The official activity key is ``gcamp/traces_array_F_F20``.  The label
   file may be the WormWideWeb ``neuropal_label.json.bz2`` schema.

Examples
--------
  python solve_gwot_md_atanas21.py validate-data --data-dir data/atanas_h5 \
      --labels neuropal_label.json.bz2

  # Distribute the 4,620 directed pair/h jobs across an array of 64 processes.
  python solve_gwot_md_atanas21.py solve --data-dir data/atanas_h5 \
      --labels neuropal_label.json.bz2 --num-shards 64 --shard-index 0 \
      --output-template 'runs/atanas21_gwot_md_paper_h{h}'

  python solve_gwot_md_atanas21.py audit \
      --output-template 'runs/atanas21_gwot_md_paper_h{h}'
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.special import logsumexp


CODE_VERSION = "atanas21-paper-lock-v1"
PAPER_URL = "https://openreview.net/forum?id=qAgQqVVwq9"
ATANAS_DOI = "https://doi.org/10.1016/j.cell.2023.07.035"
ZENODO_RECORD = "https://zenodo.org/records/19388374"
H_VALUES = tuple(range(0, 51, 5))
EPSILONS = tuple(float(10.0 ** exponent) for exponent in np.linspace(-4.0, 0.0, 21))
INIT_SEEDS = tuple(range(50))

# Exact baseline+NeuroPAL cohort.  The diagonal is Figure C.1's labeled-neuron
# count and is a useful guard against accidentally using the 38/40-worm cohort.
PAPER_COHORT: tuple[tuple[str, str, str, int], ...] = (
    ("2022-06-14-01", "2022-06-14-01-data.h5", "aabca135ef82e4be6821bf515cd2e936ae2f78014bf2979ed8a10aeaa7c4ac11", 78),
    ("2022-06-14-07", "2022-06-14-07-data.h5", "e509f47041d9faf86057832ffe5ab623ffaf2611419b90322bf8d634cfc6c2ec", 87),
    ("2022-06-14-13", "2022-06-14-13-data.h5", "a87a19529ef8d647d4801abef3d0716d90a9f8294067db63c4a8894893fa7252", 95),
    ("2022-06-28-01", "2022-06-28-01-data.h5", "df97fa9d2576f340d874d4ae40d76ed345d1a96cd7e9300a5c844f1584496842", 74),
    ("2022-06-28-07", "2022-06-28-07-data.h5", "74bad5e04bbd72c59eb45975c07bb02a066b33db172ad3478a46a85913d31ab4", 84),
    ("2022-07-15-06", "2022-07-15-06-data.h5", "fd98a50596e86b06c4c86e56ee9569e74d1bcc4311ee9eec5be9acc9bf85f448", 69),
    ("2022-07-15-12", "2022-07-15-12-data.h5", "a151870788b0f1b5e2833594f882f013d5393726415a18a3cd3206a52163038f", 91),
    ("2022-07-20-01", "2022-07-20-01-data.h5", "9bc1e197491e1e267f2a76dc21670598d7a257ece91fb8214dfdef9c1aa27ca6", 95),
    ("2022-07-26-01", "2022-07-26-01-data.h5", "84ef44b1b5bb6d5a74ba694b0133838ae379f54cbe86fc693264881e61c1347c", 67),
    ("2022-08-02-01", "2022-08-02-01-data.h5", "fdc30fb834fbbc9fc12bac4c36bd122597bcef398b94189cf2a818d363d7ee6a", 111),
    ("2023-01-09-28", "2023-01-09-28-data.h5", "811ab2a3fc3bd792967ad677c7ac6b8f92c0e29c31f021cfa84db34cae96e2a5", 91),
    ("2023-01-17-01", "2023-01-17-01-data.h5", "edb3df7ee253d58501f6645e9cf50a80ee37ad68aa33478b60514a13a2cc3d51", 90),
    ("2023-01-19-01", "2023-01-19-01-data.h5", "fc3207951da52fa6186b83259b0c6cccffb1803513eeb5ee6efa7b2212ab7f12", 71),
    ("2023-01-19-08", "2023-01-19-08-data.h5", "80b0922aca67c77b48d3d05116efff308060f877b31c1471fab8120fb2c03d17", 85),
    ("2023-01-19-15", "2023-01-19-15-data.h5", "8b55cdaea4e02b1287ab194a0e790ddb48c1ca6be8cf0342e128f7d0aa7dbdae", 60),
    ("2023-01-19-22", "2023-01-19-22-data.h5", "727ed035454485d9728fa4f946259454c979534543188358caf63de55b5a84bc", 92),
    ("2023-01-23-01", "2023-01-23-01-data.h5", "6a415a6a18123cc160d2841c4aada05cc9afd0fea5c86138ae68764367ac9861", 87),
    ("2023-01-23-08", "2023-01-23-08-data.h5", "ad514f3e768f908c41ea416f68f73c585e81db63dd2b4b324b440949d368edd9", 87),
    ("2023-01-23-15", "2023-01-23-15-data.h5", "090a6cd463c64074100d099bd2d098e852a389b656f9ad46120689db0d16d3ef", 96),
    ("2023-01-23-21", "2023-01-23-21-data.h5", "e124994f82c6ffd05671f53762bce165f04238dcab46c976ed28a857ee7a47b4", 95),
    ("2023-03-07-01", "2023-03-07-01-data.h5", "e4e9786717c0963e919f67b1c02f91cf435c249cff6af217c25192346a1fbf54", 82),
)
COHORT_UIDS = tuple(row[0] for row in PAPER_COHORT)
COHORT_BY_UID = {row[0]: row for row in PAPER_COHORT}

INVALID_LABELS = {
    "", "nan", "none", "null", "na", "n/a", "unknown", "unk",
    "unlabeled", "unlabelled", "-1", "-1.0", "noid",
}
ACTIVITY_KEYS = (
    "activity_raw", "activity", "traces_array_F_F20", "trace_array_F20",
    "trace_array_original", "trace_original", "gcamp/traces_array_F_F20",
)
LABEL_KEYS = (
    "cell_id", "cell_ids", "labels", "neuron_labels", "neuron_ids",
)
TIMESTAMP_KEYS = (
    "timestamps", "timestamp", "time", "t", "timing/timestamp_confocal",
)


@dataclass(frozen=True)
class Worm:
    uid: str
    source_path: str
    activity: np.ndarray  # [neuron, time]
    labels: np.ndarray    # [neuron], empty string means noID
    sample_rate_hz: float
    activity_key: str
    timestamp_key: str | None

    @property
    def n(self) -> int:
        return int(self.activity.shape[0])


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Every Slurm shard writes the same semantic config at startup.  A
    # process-unique temporary name prevents concurrent os.replace calls from
    # racing over one shared ``config.json.tmp`` file.
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical_uid(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    match = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}", str(value))
    return match.group(0) if match else str(value).strip()


def clean_label(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return "" if text.lower() in INVALID_LABELS else text


def first_key(keys: Iterable[str], aliases: Sequence[str]) -> str | None:
    actual = {str(key).lower().strip("/"): str(key) for key in keys}
    for alias in aliases:
        if alias.lower().strip("/") in actual:
            return actual[alias.lower().strip("/")]
    return None


def load_label_json(path: Path | None) -> dict[str, dict[int, str]]:
    if path is None:
        return {}
    opener = bz2.open if path.suffix.lower() == ".bz2" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        root = json.load(handle)
    data = root.get("data", root)
    output: dict[str, dict[int, str]] = {}
    for raw_uid, raw_entry in data.items():
        uid = canonical_uid(raw_uid)
        entries = raw_entry.get("idx_neuron-label", raw_entry.get("label", raw_entry))
        if not isinstance(entries, Mapping):
            continue
        parsed: dict[int, str] = {}
        numeric = [int(key) for key in entries if str(key).lstrip("-").isdigit()]
        # WormWideWeb's summarized labels use one-based neuron indices.
        offset = 1 if numeric and min(numeric) >= 1 and 0 not in numeric else 0
        for raw_index, raw_label in entries.items():
            if not str(raw_index).lstrip("-").isdigit():
                continue
            if isinstance(raw_label, Mapping):
                raw_label = raw_label.get("label", "")
            parsed[int(raw_index) - offset] = clean_label(raw_label)
        output[uid] = parsed
    return output


def orient_activity(array: np.ndarray, n_labels: int | None = None) -> np.ndarray:
    activity = np.asarray(array, dtype=np.float64)
    if activity.ndim != 2:
        raise ValueError(f"Activity must be 2-D, got {activity.shape}")
    if n_labels is not None:
        if activity.shape[0] == n_labels:
            return np.ascontiguousarray(activity)
        if activity.shape[1] == n_labels:
            return np.ascontiguousarray(activity.T)
        raise ValueError(
            f"Neither activity axis matches {n_labels} labels: {activity.shape}"
        )
    # The paper recordings have 109--153 neurons and roughly 1,600 frames.
    if activity.shape[0] <= 400 and activity.shape[1] > activity.shape[0]:
        return np.ascontiguousarray(activity)
    if activity.shape[1] <= 400 and activity.shape[0] > activity.shape[1]:
        return np.ascontiguousarray(activity.T)
    raise ValueError(f"Cannot infer neuron axis for activity shape {activity.shape}")


def infer_sample_rate(timestamps: np.ndarray | None, fallback: float) -> float:
    if timestamps is not None:
        values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        delta = np.diff(values)
        delta = delta[np.isfinite(delta) & (delta > 0)]
        if delta.size:
            return float(1.0 / np.median(delta))
    if not math.isfinite(fallback) or fallback <= 0:
        raise ValueError("No usable timestamps and invalid fallback sample rate")
    return float(fallback)


def load_npz_worm(path: Path, fallback_hz: float) -> Worm:
    with np.load(path, allow_pickle=False) as data:
        uid = canonical_uid(data["recording_uid"] if "recording_uid" in data.files else path.stem)
        activity_key = first_key(data.files, ACTIVITY_KEYS)
        label_key = first_key(data.files, LABEL_KEYS)
        if activity_key is None or label_key is None:
            raise KeyError(
                f"{path}: need activity and labels; keys={sorted(data.files)}"
            )
        labels = np.asarray([clean_label(item) for item in np.asarray(data[label_key]).reshape(-1)])
        activity = orient_activity(data[activity_key], len(labels))
        timestamp_key = first_key(data.files, TIMESTAMP_KEYS)
        timestamps = data[timestamp_key] if timestamp_key else None
        sample_rate = infer_sample_rate(timestamps, fallback_hz)
    return finish_worm(uid, path, activity, labels, sample_rate, activity_key, timestamp_key)


def h5_dataset_names(group: Any) -> list[str]:
    names: list[str] = []
    def visitor(name: str, value: Any) -> None:
        if hasattr(value, "shape"):
            names.append(name)
    group.visititems(visitor)
    return names


def load_h5_worm(
    path: Path,
    uid: str,
    label_map: Mapping[str, Mapping[int, str]],
    fallback_hz: float,
) -> Worm:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("HDF5 input needs h5py: python -m pip install h5py") from exc
    with h5py.File(path, "r") as handle:
        names = h5_dataset_names(handle)
        activity_key = first_key(names, ACTIVITY_KEYS)
        if activity_key is None:
            raise KeyError(f"{path}: no accepted activity dataset; datasets={names}")
        raw_activity = np.asarray(handle[activity_key], dtype=np.float64)
        timestamp_key = first_key(names, TIMESTAMP_KEYS)
        timestamps = np.asarray(handle[timestamp_key]) if timestamp_key else None
    activity = orient_activity(raw_activity)
    if uid not in label_map:
        raise KeyError(f"No label entry for {uid}; pass --labels JSON/JSON.bz2")
    labels = np.full(activity.shape[0], "", dtype=object)
    for index, label in label_map[uid].items():
        if not 0 <= index < len(labels):
            raise IndexError(f"{uid}: label index {index} outside 0..{len(labels)-1}")
        labels[index] = clean_label(label)
    sample_rate = infer_sample_rate(timestamps, fallback_hz)
    return finish_worm(uid, path, activity, labels, sample_rate, activity_key, timestamp_key)


def finish_worm(
    uid: str,
    path: Path,
    activity: np.ndarray,
    labels: np.ndarray,
    sample_rate: float,
    activity_key: str,
    timestamp_key: str | None,
) -> Worm:
    if uid not in COHORT_BY_UID:
        raise ValueError(f"{uid} is not in the paper's 21-recording cohort")
    if activity.shape[0] != len(labels):
        raise ValueError(f"{uid}: activity/label length mismatch")
    if not 109 <= activity.shape[0] <= 153:
        raise ValueError(f"{uid}: paper reports 109--153 neurons, got {activity.shape[0]}")
    if activity.shape[1] <= 2 * max(H_VALUES) + 10:
        raise ValueError(f"{uid}: too few time points: {activity.shape[1]}")
    if not np.isfinite(activity).all():
        raise ValueError(f"{uid}: activity contains NaN/Inf; strict mode does not impute")
    return Worm(
        uid=uid,
        source_path=str(path.resolve()),
        activity=np.ascontiguousarray(activity, dtype=np.float64),
        labels=np.asarray(labels, dtype=str),
        sample_rate_hz=float(sample_rate),
        activity_key=activity_key,
        timestamp_key=timestamp_key,
    )


def find_input_file(data_dir: Path, uid: str, official_name: str) -> Path:
    preferred = [
        data_dir / official_name,
        data_dir / f"{uid}.npz",
        data_dir / f"{uid}-data.npz",
        data_dir / f"{uid}.h5",
        data_dir / f"{uid}-data.h5",
    ]
    for path in preferred:
        if path.exists():
            return path
    candidates = [path for path in data_dir.rglob("*") if path.is_file() and uid in path.name]
    candidates = [path for path in candidates if path.suffix.lower() in {".npz", ".h5", ".hdf5"}]
    if len(candidates) != 1:
        raise FileNotFoundError(f"{uid}: expected one NPZ/H5 under {data_dir}, found {candidates}")
    return candidates[0]


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_cohort(
    data_dir: Path,
    labels_path: Path | None,
    fallback_hz: float,
    verify_h5_sha256: bool,
) -> list[Worm]:
    label_map = load_label_json(labels_path)
    worms: list[Worm] = []
    for uid, official_name, expected_sha, _ in PAPER_COHORT:
        path = find_input_file(data_dir, uid, official_name)
        if path.suffix.lower() == ".npz":
            worm = load_npz_worm(path, fallback_hz)
        else:
            if verify_h5_sha256:
                actual = sha256_file(path)
                if actual != expected_sha:
                    raise ValueError(f"{uid}: SHA-256 {actual} != official {expected_sha}")
            worm = load_h5_worm(path, uid, label_map, fallback_hz)
        if worm.uid != uid:
            raise ValueError(f"Expected {uid}, loaded {worm.uid} from {path}")
        worms.append(worm)
    return worms


def highpass(activity: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    sos = butter(1, 0.01, btype="highpass", fs=sample_rate_hz, output="sos")
    return np.ascontiguousarray(sosfiltfilt(sos, activity, axis=1), dtype=np.float64)


def delayed_cosine_relations(filtered: np.ndarray, max_h: int) -> np.ndarray:
    n, length = filtered.shape
    output = np.empty((2 * max_h + 1, n, n), dtype=np.float64)
    for lag in range(0, max_h + 1):
        if lag == 0:
            left = right = filtered
        else:
            left, right = filtered[:, :-lag], filtered[:, lag:]
        numerator = left @ right.T
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        denominator = left_norm[:, None] * right_norm[None, :]
        if np.any(denominator <= np.finfo(np.float64).tiny):
            raise ValueError(f"Zero-norm activity at lag {lag}")
        cosine = np.clip(numerator / denominator, -1.0, 1.0)
        distance = 1.0 - cosine
        output[max_h + lag] = distance
        output[max_h - lag] = distance.T
    if output.shape != (2 * max_h + 1, n, n) or length <= max_h:
        raise AssertionError("Relation construction failed")
    return output


def select_relations(all_relations: np.ndarray, all_h: int, h: int) -> np.ndarray:
    return all_relations[all_h - h : all_h + h + 1]


def random_feasible_plan(m: int, n: int, seed: int) -> np.ndarray:
    # Match GWTune's legacy RandomState/np.random.rand behavior.
    plan = np.random.RandomState(seed).rand(m, n)
    p = np.full((m, 1), 1.0 / m)
    q = np.full((1, n), 1.0 / n)
    for _ in range(1000):
        plan = plan * p / plan.sum(axis=1, keepdims=True)
        plan = plan * q / plan.sum(axis=0, keepdims=True)
        if (
            np.linalg.norm(plan.sum(axis=1, keepdims=True) - p) < 1e-3
            and np.linalg.norm(plan.sum(axis=0, keepdims=True) - q) < 1e-3
        ):
            break
    return plan


def sinkhorn_log(
    cost: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float,
    max_iter: int = 1000,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, int, float]:
    log_p = np.log(p)
    log_q = np.log(q)
    log_kernel = -np.asarray(cost, dtype=np.float64) / float(epsilon)
    log_u = np.zeros_like(p)
    log_v = np.zeros_like(q)
    error = math.inf
    for iteration in range(max_iter):
        log_u = log_p - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_q - logsumexp(log_kernel + log_u[:, None], axis=0)
        if iteration % 10 == 0 or iteration + 1 == max_iter:
            plan = np.exp(log_kernel + log_u[:, None] + log_v[None, :])
            error = max(
                float(np.max(np.abs(plan.sum(axis=1) - p))),
                float(np.max(np.abs(plan.sum(axis=0) - q))),
            )
            if error <= tolerance:
                return plan, iteration + 1, error
    return plan, max_iter, error


def gw_gradient_cost(
    dx: np.ndarray,
    dy: np.ndarray,
    gamma: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    # POT/GWTune differentiates the unhalved quadratic GW objective, so its
    # linearized cost is twice the tensor product below.  Keeping that factor
    # is essential: dropping it silently replaces epsilon by 2*epsilon.
    left = np.einsum("tij,j->ti", dx * dx, p).sum(axis=0)
    right = np.einsum("tij,j->ti", dy * dy, q).sum(axis=0)
    cross = np.zeros_like(gamma)
    for lag_index in range(dx.shape[0]):
        cross += dx[lag_index] @ gamma @ dy[lag_index].T
    return 2.0 * (left[:, None] + right[None, :] - 2.0 * cross)


def gw_objective(
    dx: np.ndarray,
    dy: np.ndarray,
    gamma: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    gradient = gw_gradient_cost(dx, dy, gamma, p, q)
    return float(0.5 * np.sum(gradient * gamma))


def solve_one_start(
    dx: np.ndarray,
    dy: np.ndarray,
    epsilon: float,
    initial: np.ndarray,
    outer_max_iter: int,
    sinkhorn_max_iter: int,
    outer_tolerance: float,
    sinkhorn_tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    m, n = initial.shape
    p = np.full(m, 1.0 / m)
    q = np.full(n, 1.0 / n)
    gamma = initial.copy()
    outer_error = math.inf
    total_sinkhorn_iterations = 0
    marginal_error = math.inf
    for outer_iteration in range(outer_max_iter):
        previous = gamma
        cost = gw_gradient_cost(dx, dy, previous, p, q)
        gamma, inner_iterations, marginal_error = sinkhorn_log(
            cost,
            p,
            q,
            epsilon,
            max_iter=sinkhorn_max_iter,
            tolerance=sinkhorn_tolerance,
        )
        total_sinkhorn_iterations += inner_iterations
        if outer_iteration % 10 == 0 or outer_iteration + 1 == outer_max_iter:
            outer_error = float(np.linalg.norm(gamma - previous))
            if outer_error <= outer_tolerance:
                break
    stats = {
        "outer_iterations": outer_iteration + 1,
        "outer_error": outer_error,
        "sinkhorn_iterations_total": total_sinkhorn_iterations,
        "marginal_error": marginal_error,
        "converged": bool(outer_error <= outer_tolerance),
    }
    return gamma, stats


def solve_pair(
    dx: np.ndarray,
    dy: np.ndarray,
    epsilons: Sequence[float] = EPSILONS,
    seeds: Sequence[int] = INIT_SEEDS,
    outer_max_iter: int = 1000,
    sinkhorn_max_iter: int = 1000,
    outer_tolerance: float = 1e-9,
    sinkhorn_tolerance: float = 1e-9,
) -> tuple[np.ndarray, dict[str, Any]]:
    if dx.shape[0] != dy.shape[0]:
        raise ValueError("Source/target relation counts differ")
    m, n = dx.shape[1], dy.shape[1]
    initials = [random_feasible_plan(m, n, seed) for seed in seeds]
    best_gamma: np.ndarray | None = None
    best: dict[str, Any] | None = None
    failures = 0
    starts = 0
    started = time.time()
    for epsilon in epsilons:
        for seed, initial in zip(seeds, initials):
            starts += 1
            gamma, stats = solve_one_start(
                dx,
                dy,
                float(epsilon),
                initial,
                outer_max_iter,
                sinkhorn_max_iter,
                outer_tolerance,
                sinkhorn_tolerance,
            )
            if not np.isfinite(gamma).all():
                failures += 1
                continue
            p = np.full(m, 1.0 / m)
            q = np.full(n, 1.0 / n)
            objective = gw_objective(dx, dy, gamma, p, q)
            record = {
                "epsilon": float(epsilon),
                "init_seed": int(seed),
                "unregularized_objective": objective,
                **stats,
            }
            if best is None or objective < float(best["unregularized_objective"]):
                best_gamma, best = gamma.copy(), record
    if best_gamma is None or best is None:
        raise RuntimeError("Every epsilon/initialization failed")
    best.update({
        "num_starts": starts,
        "num_failed_starts": failures,
        "elapsed_seconds": time.time() - started,
    })
    return best_gamma, best


def solve_pair_torch(
    dx_numpy: np.ndarray,
    dy_numpy: np.ndarray,
    device: str,
    init_batch_size: int,
    epsilons: Sequence[float] = EPSILONS,
    seeds: Sequence[int] = INIT_SEEDS,
    outer_max_iter: int = 1000,
    sinkhorn_max_iter: int = 1000,
    outer_tolerance: float = 1e-9,
    sinkhorn_tolerance: float = 1e-9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Torch implementation batching random starts for practical GPU runs.

    Every batch member follows the same equations as ``solve_pair``.  The
    random feasible plans are still built by NumPy RandomState so changing the
    backend does not change initialization.  A converged member may receive
    additional fixed-point updates while another member in its batch finishes;
    this only tightens convergence and is recorded as an implementation choice.
    """
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "--device cpu/cuda needs PyTorch; install the build appropriate for your machine"
        ) from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    if init_batch_size <= 0:
        raise ValueError("init-batch-size must be positive")
    torch_device = torch.device(device)
    dtype = torch.float64
    dx = torch.as_tensor(dx_numpy, dtype=dtype, device=torch_device)
    dy = torch.as_tensor(dy_numpy, dtype=dtype, device=torch_device)
    if dx.shape[0] != dy.shape[0]:
        raise ValueError("Source/target relation counts differ")
    m, n = int(dx.shape[1]), int(dy.shape[1])
    p = torch.full((m,), 1.0 / m, dtype=dtype, device=torch_device)
    q = torch.full((n,), 1.0 / n, dtype=dtype, device=torch_device)
    log_p, log_q = torch.log(p), torch.log(q)
    left = torch.einsum("rij,j->ri", dx * dx, p).sum(dim=0)
    right = torch.einsum("rij,j->ri", dy * dy, q).sum(dim=0)
    constant = left[:, None] + right[None, :]

    def gradient(batch_gamma: Any) -> Any:
        # dx[r,i,j] G[b,j,k] dy[r,l,k] = (dx @ G @ dy.T)[b,i,l]
        cross = torch.einsum("rij,bjk,rlk->bil", dx, batch_gamma, dy)
        return constant[None, :, :] - 2.0 * cross

    def batched_sinkhorn(cost: Any, epsilon: float) -> tuple[Any, int, float]:
        log_kernel = -cost / float(epsilon)
        batch = int(cost.shape[0])
        log_u = torch.zeros((batch, m), dtype=dtype, device=torch_device)
        log_v = torch.zeros((batch, n), dtype=dtype, device=torch_device)
        error = math.inf
        plan = None
        for iteration in range(sinkhorn_max_iter):
            log_u = log_p[None, :] - torch.logsumexp(
                log_kernel + log_v[:, None, :], dim=2
            )
            log_v = log_q[None, :] - torch.logsumexp(
                log_kernel + log_u[:, :, None], dim=1
            )
            if iteration % 10 == 0 or iteration + 1 == sinkhorn_max_iter:
                plan = torch.exp(log_kernel + log_u[:, :, None] + log_v[:, None, :])
                row_error = torch.amax(torch.abs(plan.sum(dim=2) - p[None, :]))
                col_error = torch.amax(torch.abs(plan.sum(dim=1) - q[None, :]))
                error = float(torch.maximum(row_error, col_error).item())
                if error <= sinkhorn_tolerance:
                    return plan, iteration + 1, error
        assert plan is not None
        return plan, sinkhorn_max_iter, error

    best_gamma: np.ndarray | None = None
    best: dict[str, Any] | None = None
    failures = starts = 0
    started = time.time()
    with torch.no_grad():
        for epsilon in epsilons:
            for offset in range(0, len(seeds), init_batch_size):
                seed_batch = list(seeds[offset : offset + init_batch_size])
                initial_numpy = np.stack(
                    [random_feasible_plan(m, n, int(seed)) for seed in seed_batch]
                )
                gamma = torch.as_tensor(initial_numpy, dtype=dtype, device=torch_device)
                starts += len(seed_batch)
                total_sinkhorn_iterations = 0
                marginal_error = math.inf
                outer_errors = torch.full(
                    (len(seed_batch),), math.inf, dtype=dtype, device=torch_device
                )
                for outer_iteration in range(outer_max_iter):
                    previous = gamma
                    gamma, inner_iterations, marginal_error = batched_sinkhorn(
                        gradient(previous), float(epsilon)
                    )
                    total_sinkhorn_iterations += inner_iterations
                    if outer_iteration % 10 == 0 or outer_iteration + 1 == outer_max_iter:
                        outer_errors = torch.linalg.vector_norm(
                            (gamma - previous).reshape(len(seed_batch), -1), dim=1
                        )
                        if bool(torch.all(outer_errors <= outer_tolerance).item()):
                            break
                objectives = 0.5 * torch.sum(gradient(gamma) * gamma, dim=(1, 2))
                finite = torch.isfinite(objectives)
                failures += int((~finite).sum().item())
                if bool(torch.any(finite).item()):
                    masked = torch.where(
                        finite,
                        objectives,
                        torch.full_like(objectives, math.inf),
                    )
                    local_index = int(torch.argmin(masked).item())
                    objective = float(objectives[local_index].item())
                    record = {
                        "epsilon": float(epsilon),
                        "init_seed": int(seed_batch[local_index]),
                        "unregularized_objective": objective,
                        "outer_iterations": outer_iteration + 1,
                        "outer_error": float(outer_errors[local_index].item()),
                        "sinkhorn_iterations_total": total_sinkhorn_iterations,
                        "marginal_error": marginal_error,
                        "converged": bool(
                            float(outer_errors[local_index].item()) <= outer_tolerance
                        ),
                        "backend": f"torch-{device}",
                        "init_batch_size": len(seed_batch),
                    }
                    if best is None or objective < float(best["unregularized_objective"]):
                        best_gamma = gamma[local_index].detach().cpu().numpy().copy()
                        best = record
    if best_gamma is None or best is None:
        raise RuntimeError("Every epsilon/initialization failed")
    best.update({
        "num_starts": starts,
        "num_failed_starts": failures,
        "elapsed_seconds": time.time() - started,
    })
    return best_gamma, best


def run_config(h: int, args: argparse.Namespace) -> dict[str, Any]:
    paper_lock = (
        args.outer_max_iter == 1000
        and args.sinkhorn_max_iter == 1000
        and math.isclose(args.outer_tolerance, 1e-9, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(args.sinkhorn_tolerance, 1e-9, rel_tol=0.0, abs_tol=0.0)
    )
    return {
        "code_version": CODE_VERSION,
        "paper_mode": paper_lock,
        "paper_url": PAPER_URL,
        "atanas_dataset_doi": ATANAS_DOI,
        "source_archive": ZENODO_RECORD,
        "cohort_rule": "label=true AND type contains baseline,neuropal",
        "cohort_uids": list(COHORT_UIDS),
        "geometry_used": False,
        "all_recorded_neurons_used": True,
        "distance": "time-delayed cosine",
        "max_lag": int(h),
        "max_lag_frames": int(h),
        "lag_step": 1,
        "lag_weights": "w_tau=1",
        "normalize_lag_weights": False,
        "highpass_cutoff": 0.01,
        "highpass_order": 1,
        "highpass_direction": "zero-phase scipy.signal.sosfiltfilt (paper unspecified)",
        "trace_normalization": "none",
        "neuron_masses": "uniform",
        "solver": "entropic GWOT-MD fixed point with log-domain Sinkhorn",
        "compute_backend": args.device,
        "init_batch_size": args.init_batch_size,
        "gradient_factor": 2,
        "resolved_epsilons": list(EPSILONS),
        "resolved_n_inits": len(INIT_SEEDS),
        "initialization": "RandomState(seed).rand then alternating marginal normalization",
        "initialization_seeds": list(INIT_SEEDS),
        "outer_max_iter": args.outer_max_iter,
        "sinkhorn_max_iter": args.sinkhorn_max_iter,
        "outer_tolerance": args.outer_tolerance,
        "sinkhorn_tolerance": args.sinkhorn_tolerance,
        "paper_unspecified_choices": [
            "random seeds and exact initial-plan sampler",
            "outer/Sinkhorn stopping rules",
            "causal versus zero-phase Butterworth filtering",
            "factor-of-two convention in the entropic GW linearization",
        ],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "command": sys.argv,
    }


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def pair_path(output_template: str, h: int, target: str, reference: str) -> Path:
    return Path(output_template.format(h=h)).expanduser().resolve() / "pairs" / f"{target}__to__{reference}" / "matching.npz"


def all_tasks() -> list[tuple[int, str, str]]:
    return [
        (h, target, reference)
        for h in H_VALUES
        for target in COHORT_UIDS
        for reference in COHORT_UIDS
        if target != reference
    ]


def validate_data_command(args: argparse.Namespace) -> None:
    worms = load_cohort(
        Path(args.data_dir).expanduser().resolve(),
        Path(args.labels).expanduser().resolve() if args.labels else None,
        args.fallback_sample_rate_hz,
        args.verify_h5_sha256,
    )
    rows = []
    for worm in worms:
        rows.append({
            "uid": worm.uid,
            "neurons": worm.n,
            "timepoints": worm.activity.shape[1],
            "labeled_entries": int(np.sum(worm.labels != "")),
            "unique_nonempty_labels": len(set(worm.labels).difference({""})),
            "paper_figure_c1_labeled": COHORT_BY_UID[worm.uid][3],
            "sample_rate_hz": worm.sample_rate_hz,
            "activity_key": worm.activity_key,
            "path": worm.source_path,
        })
    print(json.dumps(rows, indent=2), flush=True)
    log("Dataset structure passed. Run the evaluator's full Figure-C.1 label-overlap audit before claiming exact reproduction.")


def solve_command(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    data_dir = Path(args.data_dir).expanduser().resolve()
    labels = Path(args.labels).expanduser().resolve() if args.labels else None
    worms = load_cohort(
        data_dir,
        labels,
        args.fallback_sample_rate_hz,
        args.verify_h5_sha256,
    )
    by_uid = {worm.uid: worm for worm in worms}
    log("High-pass filtering and building all lag -50..50 cosine relations")
    relations: dict[str, np.ndarray] = {}
    for worm in worms:
        filtered = highpass(worm.activity, worm.sample_rate_hz)
        relations[worm.uid] = delayed_cosine_relations(filtered, max(H_VALUES))
        log(f"relations {worm.uid}: {relations[worm.uid].shape}")

    for h in H_VALUES:
        run_dir = Path(args.output_template.format(h=h)).expanduser().resolve()
        write_json_atomic(run_dir / "config.json", run_config(h, args))

    tasks = all_tasks()
    selected = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_index]
    log(f"Shard {args.shard_index}/{args.num_shards}: {len(selected)} of {len(tasks)} directed pair/h jobs")
    completed = skipped = 0
    for task_index, (h, target, reference) in enumerate(selected, start=1):
        output = pair_path(args.output_template, h, target, reference)
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        log(f"[{task_index}/{len(selected)}] h={h} {target} -> {reference}")
        dx = select_relations(relations[target], max(H_VALUES), h)
        dy = select_relations(relations[reference], max(H_VALUES), h)
        solve_kwargs = dict(
            outer_max_iter=args.outer_max_iter,
            sinkhorn_max_iter=args.sinkhorn_max_iter,
            outer_tolerance=args.outer_tolerance,
            sinkhorn_tolerance=args.sinkhorn_tolerance,
        )
        if args.device == "numpy":
            gamma, stats = solve_pair(dx, dy, **solve_kwargs)
            stats["backend"] = "numpy-cpu"
        else:
            gamma, stats = solve_pair_torch(
                dx,
                dy,
                device=args.device,
                init_batch_size=args.init_batch_size,
                **solve_kwargs,
            )
        conditional = gamma / gamma.sum(axis=1, keepdims=True)
        atomic_save_npz(
            output,
            gamma=gamma,
            conditional=conditional,
            target_worm=np.asarray(target),
            reference_worm=np.asarray(reference),
            target_labels=by_uid[target].labels,
            reference_labels=by_uid[reference].labels,
            h=np.asarray(h),
            best_epsilon=np.asarray(stats["epsilon"]),
            best_init_seed=np.asarray(stats["init_seed"]),
            unregularized_objective=np.asarray(stats["unregularized_objective"]),
        )
        write_json_atomic(output.parent / "matching.json", stats)
        completed += 1
    log(f"Shard finished: completed={completed}, resumed/skipped={skipped}")


def audit_command(args: argparse.Namespace) -> None:
    missing: list[str] = []
    malformed: list[str] = []
    count = 0
    for h, target, reference in all_tasks():
        path = pair_path(args.output_template, h, target, reference)
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                gamma = np.asarray(data["gamma"], dtype=np.float64)
                conditional = np.asarray(data["conditional"], dtype=np.float64)
                if gamma.ndim != 2 or conditional.shape != gamma.shape:
                    raise ValueError("bad matrix shape")
                if not np.isfinite(gamma).all() or not np.allclose(conditional.sum(axis=1), 1.0, atol=1e-7):
                    raise ValueError("nonfinite or non-stochastic")
            count += 1
        except Exception as exc:
            malformed.append(f"{path}: {exc}")
    result = {
        "expected_directed_pair_h_caches": len(all_tasks()),
        "valid": count,
        "missing": len(missing),
        "malformed": len(malformed),
        "first_missing": missing[:20],
        "first_malformed": malformed[:20],
    }
    print(json.dumps(result, indent=2), flush=True)
    if missing or malformed:
        raise SystemExit(2)


def self_check() -> None:
    rng = np.random.RandomState(7)
    trace = rng.normal(size=(5, 300))
    trace = np.cumsum(trace, axis=1)
    filtered = highpass(trace, 1.67)
    relations = delayed_cosine_relations(filtered, 3)
    if not np.allclose(relations[2], relations[4].T, atol=1e-12):
        raise AssertionError("negative-lag transpose identity failed")
    # Verify the decomposed (standard, unhalved) GW objective against the
    # literal six-loop definition used by POT/GWTune.
    tiny_dx = relations[:2, :3, :3]
    tiny_dy = relations[1:3, :4, :4]
    tiny_p, tiny_q = np.full(3, 1.0 / 3), np.full(4, 1.0 / 4)
    tiny_gamma, _, _ = sinkhorn_log(
        np.arange(12, dtype=np.float64).reshape(3, 4) / 20.0,
        tiny_p,
        tiny_q,
        epsilon=0.5,
        max_iter=1000,
        tolerance=1e-13,
    )
    fast_objective = gw_objective(tiny_dx, tiny_dy, tiny_gamma, tiny_p, tiny_q)
    literal_objective = sum(
        (tiny_dx[r, i, k] - tiny_dy[r, j, ell]) ** 2
        * tiny_gamma[i, j]
        * tiny_gamma[k, ell]
        for r in range(2)
        for i in range(3)
        for j in range(4)
        for k in range(3)
        for ell in range(4)
    )
    if not math.isclose(fast_objective, literal_objective, rel_tol=0.0, abs_tol=1e-11):
        raise AssertionError("decomposed GWOT-MD objective does not match definition")
    permutation = np.asarray([2, 4, 0, 3, 1])
    target_relations = relations[:, permutation][:, :, permutation]
    gamma, stats = solve_pair(
        relations,
        target_relations,
        epsilons=[0.001],
        seeds=list(range(8)),
        outer_max_iter=100,
        sinkhorn_max_iter=300,
        outer_tolerance=1e-8,
        sinkhorn_tolerance=1e-9,
    )
    if not np.allclose(gamma.sum(axis=1), 0.2, atol=1e-7):
        raise AssertionError("row marginals failed")
    if not np.allclose(gamma.sum(axis=0), 0.2, atol=1e-7):
        raise AssertionError("column marginals failed")
    expected = np.argsort(permutation)
    accuracy = float(np.mean(np.argmax(gamma, axis=1) == expected))
    if accuracy < 0.8:
        raise AssertionError(f"permutation recovery only {accuracy:.1%}")
    print(json.dumps({"status": "ok", "top1": accuracy, "best": stats}, indent=2))


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--labels", default=None, help="Label JSON or JSON.bz2 for HDF5 input")
    parser.add_argument("--fallback-sample-rate-hz", type=float, default=1.67)
    parser.add_argument("--verify-h5-sha256", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-protocol GWOT-MD pair solver for Atanas-21",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data")
    add_data_arguments(validate)

    solve = subparsers.add_parser("solve")
    add_data_arguments(solve)
    solve.add_argument("--output-template", default="runs/atanas21_gwot_md_paper_h{h}")
    solve.add_argument("--num-shards", type=int, default=1)
    solve.add_argument("--shard-index", type=int, default=0)
    solve.add_argument("--overwrite", action="store_true")
    solve.add_argument(
        "--device",
        choices=["numpy", "cpu", "cuda"],
        default="numpy",
        help="NumPy serial starts, or PyTorch batched starts on CPU/CUDA",
    )
    solve.add_argument(
        "--init-batch-size",
        type=int,
        default=50,
        help="Number of the 50 initializations solved together by PyTorch",
    )
    solve.add_argument("--outer-max-iter", type=int, default=1000)
    solve.add_argument("--sinkhorn-max-iter", type=int, default=1000)
    solve.add_argument("--outer-tolerance", type=float, default=1e-9)
    solve.add_argument("--sinkhorn-tolerance", type=float, default=1e-9)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--output-template", default="runs/atanas21_gwot_md_paper_h{h}")

    subparsers.add_parser("self-check")
    args = parser.parse_args()

    if args.command == "validate-data":
        validate_data_command(args)
    elif args.command == "solve":
        solve_command(args)
    elif args.command == "audit":
        audit_command(args)
    elif args.command == "self-check":
        self_check()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
