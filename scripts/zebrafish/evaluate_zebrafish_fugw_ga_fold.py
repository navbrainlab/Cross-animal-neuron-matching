#!/usr/bin/env python3
"""Official FUGW (geometry + activity) on one Zebrafish LOFO8 fold.

The outer-validation grid is completed and durably locked before this program
opens the outer-test directory.  One FUGW coupling is fitted per physical q/r
pair and is scored in both directions with the benchmark's canonical rules.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from NeuRID_reproducibility.scripts.zebrafish.query_record_io import (
    QUERY_COLUMNS,
    records_from_score_matrix,
)


DEFAULT_DATA = REPO / "Data/Zebrafish_MPRT_LOFO8_60m"
DEFAULT_OUTPUT = REPO / "runs/zebrafish_fugw_official_ga_lofo8"
METHOD_KEY = "fugw_ga_official"
METHOD_NAME = "FUGW (G+A; official)"
EXPECTED_QUERIES = {
    1: 3536, 2: 3954, 3: 768, 4: 1820,
    5: 1464, 6: 938, 7: 2492, 8: 2184,
}
INVALID_IDS = {
    "", "-1", "nan", "none", "null", "na", "n/a", "unknown",
    "unlabeled", "unlabelled", "invalid",
}


@dataclass(frozen=True)
class Side:
    path: Path
    pair_id: str
    side: str
    uid: str
    xyz_raw: np.ndarray
    activity: np.ndarray
    ids: np.ndarray
    supervised: np.ndarray


@dataclass(frozen=True)
class Parameters:
    alpha: float
    rho: float
    eps: float

    def key(self) -> str:
        return f"a{self.alpha:.12g}_r{self.rho:.12g}_e{self.eps:.12g}"

    def as_dict(self) -> dict[str, float]:
        return {"alpha_activity": self.alpha, "rho": self.rho, "eps": self.eps}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_query_gzip(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUERY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def scalar_text(value: Any) -> str:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected scalar, got {array.shape}")
    item = array[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def mask(data: np.lib.npyio.NpzFile, key: str, n: int) -> np.ndarray:
    if key not in data.files:
        return np.ones(n, dtype=bool)
    value = np.asarray(data[key], dtype=bool).reshape(-1)
    if value.shape != (n,):
        raise ValueError(f"{key}: expected {(n,)}, got {value.shape}")
    return value


def clean_id(value: Any) -> str:
    result = str(value).strip()
    return "" if result.lower() in INVALID_IDS else result


def load_side(path: Path) -> Side:
    with np.load(path, allow_pickle=True) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        activity = np.asarray(data["activity_raw"], dtype=np.float64)
        ids = np.asarray(data["cell_id"]).astype(str).reshape(-1)
        n = len(ids)
        if xyz.shape != (n, 3):
            raise ValueError(f"{path}: xyz={xyz.shape}, expected={(n, 3)}")
        if activity.ndim != 2:
            raise ValueError(f"{path}: activity={activity.shape}")
        if activity.shape[0] != n and activity.shape[1] == n:
            activity = activity.T
        if activity.shape[0] != n:
            raise ValueError(f"{path}: activity={activity.shape}, N={n}")
        candidate = np.isfinite(xyz).all(axis=1) & mask(data, "valid_xyz_mask", n)
        supervised = candidate.copy()
        for name in ("labeled_mask", "certain_mask", "clean_mask"):
            supervised &= mask(data, name, n)
        supervised &= np.asarray([bool(clean_id(value)) for value in ids])
        pair_id = scalar_text(data["pair_id"])
        side = scalar_text(data["side"]).lower()
        uid = (
            scalar_text(data["recording_uid"])
            if "recording_uid" in data.files
            else path.stem
        )
    if side not in {"q", "r"}:
        raise ValueError(f"{path}: side={side!r}")
    activity = activity[candidate]
    if not np.isfinite(activity).all():
        raise ValueError(f"{path}: non-finite candidate activity")
    return Side(
        path=path.resolve(), pair_id=pair_id, side=side, uid=uid,
        xyz_raw=np.ascontiguousarray(xyz[candidate]),
        activity=np.ascontiguousarray(activity), ids=ids[candidate],
        supervised=supervised[candidate],
    )


def load_pairs(fold_root: Path, split: str) -> list[dict[str, Any]]:
    paths = sorted((fold_root / split).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ under {fold_root / split}")
    grouped: dict[str, dict[str, Side]] = {}
    for path in paths:
        side = load_side(path)
        if side.side in grouped.setdefault(side.pair_id, {}):
            raise ValueError(f"Duplicate {side.pair_id}/{side.side}")
        grouped[side.pair_id][side.side] = side
    pairs = []
    for pair_id in sorted(grouped):
        if set(grouped[pair_id]) != {"q", "r"}:
            raise ValueError(f"Incomplete pair {pair_id}: {sorted(grouped[pair_id])}")
        pairs.append({"pair_id": pair_id, **grouped[pair_id]})
    return pairs


def fit_xyz_scaler(pairs: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.concatenate([pair[side].xyz_raw for pair in pairs for side in ("q", "r")])
    mean, std = xyz.mean(axis=0), xyz.std(axis=0)
    return mean, np.where(std > 1e-8, std, 1.0)


def unique_supervised_map(side: Side) -> dict[str, int]:
    eligible = [clean_id(value) for value, keep in zip(side.ids, side.supervised) if keep]
    counts = Counter(eligible)
    return {
        clean_id(value): index
        for index, (value, keep) in enumerate(zip(side.ids, side.supervised))
        if keep and clean_id(value) and counts[clean_id(value)] == 1
    }


def target_vectors(q: Side, r: Side) -> tuple[np.ndarray, np.ndarray]:
    qmap, rmap = unique_supervised_map(q), unique_supervised_map(r)
    common = sorted(set(qmap) & set(rmap))
    qtargets = np.full(len(q.ids), -1, dtype=np.int64)
    rtargets = np.full(len(r.ids), -1, dtype=np.int64)
    for identity in common:
        qtargets[qmap[identity]] = rmap[identity]
        rtargets[rmap[identity]] = qmap[identity]
    return qtargets, rtargets


def activity_geometry(activity: np.ndarray) -> np.ndarray:
    """Within-side RMS activity-trace distance, normalized to [0, 1]."""
    length = activity.shape[1]
    norms = np.einsum("it,it->i", activity, activity) / length
    squared = norms[:, None] + norms[None, :] - 2.0 * (activity @ activity.T) / length
    distances = np.sqrt(np.maximum(squared, 0.0))
    np.fill_diagonal(distances, 0.0)
    maximum = float(distances.max(initial=0.0))
    if maximum > 1e-12:
        distances /= maximum
    return distances.astype(np.float32)


def geometry_features(
    q: Side, r: Side, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = (q.xyz_raw - mean) / std
    target = (r.xyz_raw - mean) / std
    squared = np.sum((source[:, None] - target[None, :]) ** 2, axis=2)
    scale = np.sqrt(max(float(squared.max(initial=0.0)), 1e-12))
    return (source / scale).T.astype(np.float32), (target / scale).T.astype(np.float32)


def official_fugw() -> tuple[type, str, Path]:
    try:
        import fugw
        from fugw.mappings import FUGW
        import fugw.mappings.dense as dense
    except ImportError as exc:
        raise RuntimeError("Official fugw==0.1.1 is required") from exc
    version = str(getattr(fugw, "__version__", "unknown"))
    if version != "0.1.1":
        raise RuntimeError(f"Expected official fugw==0.1.1, found {version}")
    return FUGW, version, Path(dense.__file__).resolve()


def solve_plan(
    q: Side, r: Side, mean: np.ndarray, std: np.ndarray, params: Parameters,
    *, device: str, nits_bcd: int, nits_uot: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    FUGW, _, _ = official_fugw()
    qfeatures, rfeatures = geometry_features(q, r, mean, std)
    mapping = FUGW(
        alpha=params.alpha, rho=params.rho, eps=params.eps,
        reg_mode="joint", divergence="kl",
    )
    mapping.fit(
        source_features=qfeatures, target_features=rfeatures,
        source_geometry=activity_geometry(q.activity),
        target_geometry=activity_geometry(r.activity),
        solver="mm", solver_params={"nits_bcd": nits_bcd, "nits_uot": nits_uot},
        device=device, verbose=False,
    )
    plan = mapping.pi.detach().cpu().numpy().astype(np.float64, copy=False)
    if plan.shape != (len(q.ids), len(r.ids)) or not np.isfinite(plan).all():
        raise RuntimeError(f"Invalid FUGW plan {plan.shape}")
    if np.any(plan < 0):
        raise RuntimeError("Official FUGW returned negative mass")
    return plan, {
        "transport_mass": float(plan.sum()),
        "nonzero_entries": int(np.count_nonzero(plan)),
    }


def cached_plan(
    pair: dict[str, Any], mean: np.ndarray, std: np.ndarray, params: Parameters,
    cache_path: Path, *, device: str, nits_bcd: int, nits_uot: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    q, r = pair["q"], pair["r"]
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            expected = (
                scalar_text(data["pair_id"]) == pair["pair_id"]
                and float(data["alpha"].item()) == params.alpha
                and float(data["rho"].item()) == params.rho
                and float(data["eps"].item()) == params.eps
                and int(data["nits_bcd"].item()) == nits_bcd
                and int(data["nits_uot"].item()) == nits_uot
            )
            if not expected:
                raise RuntimeError(f"Cache metadata mismatch: {cache_path}")
            plan = np.asarray(data["plan"], dtype=np.float64)
            if plan.shape != (len(q.ids), len(r.ids)):
                raise RuntimeError(f"Cache shape mismatch: {cache_path}")
            return plan, {"cache_hit": True, "transport_mass": float(plan.sum())}
    plan, diagnostics = solve_plan(
        q, r, mean, std, params,
        device=device, nits_bcd=nits_bcd, nits_uot=nits_uot,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(
        temporary, plan=plan.astype(np.float32), pair_id=np.asarray(pair["pair_id"]),
        alpha=np.asarray(params.alpha), rho=np.asarray(params.rho), eps=np.asarray(params.eps),
        nits_bcd=np.asarray(nits_bcd), nits_uot=np.asarray(nits_uot),
        q_uid=np.asarray(q.uid), r_uid=np.asarray(r.uid), q_ids=q.ids, r_ids=r.ids,
    )
    os.replace(temporary, cache_path)
    return plan, {**diagnostics, "cache_hit": False}


def metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    q = len(rows)
    if q == 0:
        raise RuntimeError("No scored queries")
    return {
        "queries": q,
        "top1_correct": sum(int(row["top1_correct"]) for row in rows),
        "top5_correct": sum(int(row["top5_correct"]) for row in rows),
        "rr_sum": sum(float(row["reciprocal_rank"]) for row in rows),
        "hungarian_correct": sum(int(row["hungarian_correct"]) for row in rows),
        "top1": sum(int(row["top1_correct"]) for row in rows) / q,
        "top5": sum(int(row["top5_correct"]) for row in rows) / q,
        "mrr": sum(float(row["reciprocal_rank"]) for row in rows) / q,
        "hungarian": sum(int(row["hungarian_correct"]) for row in rows) / q,
        "coverage": sum(int(row["covered"]) for row in rows) / q,
    }


def evaluate_pairs(
    *, fold: int, split: str, pairs: Sequence[dict[str, Any]], mean: np.ndarray,
    std: np.ndarray, params: Parameters, output_dir: Path, device: str,
    nits_bcd: int, nits_uot: int, save_test: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs):
        cache_path = output_dir / "couplings" / split / params.key() / f"pair_{pair_index:03d}.npz"
        plan, diagnostics = cached_plan(
            pair, mean, std, params, cache_path,
            device=device, nits_bcd=nits_bcd, nits_uot=nits_uot,
        )
        q, r = pair["q"], pair["r"]
        row_target, col_target = target_vectors(q, r)
        rows = records_from_score_matrix(
            method=METHOD_KEY, fold=fold, seed=None, pair_index=pair_index,
            pair_id=pair["pair_id"], score=plan, q_uid=q.uid, r_uid=r.uid,
            q_ids=q.ids, r_ids=r.ids, row_target=row_target, col_target=col_target,
        )
        all_rows.extend(rows)
        value = metrics(rows)
        pair_rows.append({
            "pair_index": pair_index, "pair_id": pair["pair_id"],
            "q_path": str(q.path), "r_path": str(r.path),
            "q_candidates": len(q.ids), "r_candidates": len(r.ids),
            "coupling": str(cache_path.resolve()),
            "coupling_sha256": sha256(cache_path),
            "transport_mass": diagnostics["transport_mass"], **value,
        })
    value = {"pairs": len(pairs), **metrics(all_rows), **params.as_dict()}
    if save_test:
        write_query_gzip(output_dir / "query_predictions.csv.gz", all_rows)
        write_csv(output_dir / "pair_manifest.csv", pair_rows)
    return value, all_rows, pair_rows


def parse_grid(raw: str, name: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name}: grid must be nonempty and unique")
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{name}: values must be finite and positive")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alphas", default="0.25,0.5,0.75")
    parser.add_argument("--rhos", default="0.1,1,10")
    parser.add_argument("--epsilons", default="0.01")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nits-bcd", type=int, default=10)
    parser.add_argument("--nits-uot", type=int, default=1000)
    parser.add_argument("--reuse-lock", action="store_true")
    args = parser.parse_args()
    alphas = parse_grid(args.alphas, "alphas")
    if any(alpha >= 1 for alpha in alphas):
        raise ValueError("G+A alpha must be strictly between zero and one")
    rhos = parse_grid(args.rhos, "rhos")
    epsilons = parse_grid(args.epsilons, "epsilons")
    if args.nits_bcd <= 0 or args.nits_uot <= 0:
        raise ValueError("Solver iteration counts must be positive")
    grid = [Parameters(*values) for values in itertools.product(alphas, rhos, epsilons)]
    fold_root = args.data_root.resolve() / f"fold_{args.fold}"
    output_dir = args.output_root.resolve() / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: test remains unopened until the lock is durable.
    train_pairs = load_pairs(fold_root, "train")
    validation_pairs = load_pairs(fold_root, "val")
    mean, std = fit_xyz_scaler(train_pairs)
    FUGW, version, official_source = official_fugw()
    del FUGW
    lock_path = output_dir / "LOCKED_BEFORE_TEST.json"
    if args.reuse_lock:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for key, expected in {
            "status": "LOCKED_BEFORE_TEST", "fold": args.fold,
            "fugw_version": version, "device": args.device,
            "nits_bcd": args.nits_bcd, "nits_uot": args.nits_uot,
            "hyperparameter_grid": [value.as_dict() for value in grid],
        }.items():
            if lock.get(key) != expected:
                raise RuntimeError(f"Lock mismatch for {key}: {lock.get(key)!r} != {expected!r}")
        if not np.array_equal(np.asarray(lock["train_xyz_mean"]), mean):
            raise RuntimeError("Locked train mean changed")
        if not np.array_equal(np.asarray(lock["train_xyz_std"]), std):
            raise RuntimeError("Locked train std changed")
        chosen = lock["selected"]
        selected = Parameters(chosen["alpha_activity"], chosen["rho"], chosen["eps"])
    else:
        if lock_path.exists():
            raise FileExistsError(f"Use --reuse-lock or another output: {lock_path}")
        sweep = []
        for grid_index, params in enumerate(grid):
            value, _, _ = evaluate_pairs(
                fold=args.fold, split="validation", pairs=validation_pairs,
                mean=mean, std=std, params=params, output_dir=output_dir,
                device=args.device, nits_bcd=args.nits_bcd,
                nits_uot=args.nits_uot, save_test=False,
            )
            sweep.append({"grid_index": grid_index, **params.as_dict(), **value})
            print(
                f"[VAL] fold{args.fold} {params.key()} Q={value['queries']} "
                f"Top1={100*value['top1']:.2f}% MRR={value['mrr']:.4f} "
                f"Hung={100*value['hungarian']:.2f}%", flush=True,
            )
        best = sorted(
            sweep,
            key=lambda row: (
                -row["top1"], -row["mrr"], -row["hungarian"], row["grid_index"]
            ),
        )[0]
        selected = Parameters(best["alpha_activity"], best["rho"], best["eps"])
        lock = {
            "status": "LOCKED_BEFORE_TEST", "method": METHOD_NAME,
            "method_key": METHOD_KEY, "fold": args.fold, "deterministic": True,
            "implementation": "official fugw.mappings.FUGW dense solver",
            "fugw_version": version, "official_source": str(official_source),
            "official_source_sha256": sha256(official_source),
            "data_root": str(args.data_root.resolve()),
            "split_counts_before_test": {
                "train_pairs": len(train_pairs), "validation_pairs": len(validation_pairs)
            },
            "train_xyz_mean": mean.tolist(), "train_xyz_std": std.tolist(),
            "hyperparameter_grid": [value.as_dict() for value in grid],
            "selected": selected.as_dict(), "validation_sweep": sweep,
            "selection_split": "outer validation only",
            "selection_criterion": "Top-1, then MRR, Hungarian, declared grid order",
            "linear_feature_term": "pair-normalized squared Euclidean of train-z-scored XYZ",
            "gw_structure_term": "within-side normalized RMS distance of archived 128-point activity",
            "alpha_semantics": "alpha*activity_GW + (1-alpha)*geometry_Wasserstein",
            "solver": "mm", "reg_mode": "joint", "divergence": "kl",
            "device": args.device, "nits_bcd": args.nits_bcd,
            "nits_uot": args.nits_uot,
            "ranking_tie_policy": "optimistic: 1 + count(score > target)",
            "test_opened_before_lock": False,
        }
        write_json(lock_path, lock)
        write_csv(output_dir / "validation_grid.csv", sweep)
        print(f"[LOCK] {lock_path}; test has not been read", flush=True)

    # Phase 2: first test access.
    test_pairs = load_pairs(fold_root, "test")
    test, rows, pair_rows = evaluate_pairs(
        fold=args.fold, split="test", pairs=test_pairs, mean=mean, std=std,
        params=selected, output_dir=output_dir, device=args.device,
        nits_bcd=args.nits_bcd, nits_uot=args.nits_uot, save_test=True,
    )
    if test["queries"] != EXPECTED_QUERIES[args.fold]:
        raise RuntimeError(
            f"fold{args.fold}: query audit {test['queries']} != {EXPECTED_QUERIES[args.fold]}"
        )
    report = {
        "status": "PASS", "method": METHOD_NAME, "method_key": METHOD_KEY,
        "implementation": "official fugw==0.1.1; fugw.mappings.FUGW",
        "fold": args.fold, "model_seed": None, "deterministic": True,
        "selected": selected.as_dict(), "test": test,
        "expected_queries": EXPECTED_QUERIES[args.fold], "query_audit_passed": True,
        "query_protocol": "physical q/r pairs, both directions, common strict unique identities",
        "candidate_protocol": "full finite-XYZ and valid_xyz_mask population in original order",
        "ranking_tie_policy": "optimistic: 1 + count(score > target)",
        "hungarian_protocol": "one assignment per physical coupling, inverted for reverse direction",
        "lock": str(lock_path), "lock_sha256": sha256(lock_path),
        "query_predictions": str((output_dir / "query_predictions.csv.gz").resolve()),
        "query_predictions_sha256": sha256(output_dir / "query_predictions.csv.gz"),
        "pair_manifest": str((output_dir / "pair_manifest.csv").resolve()),
        "pair_manifest_sha256": sha256(output_dir / "pair_manifest.csv"),
    }
    write_json(output_dir / "test_metrics.json", report)
    print(
        f"[TEST] fold{args.fold} {selected.key()} pairs={len(pair_rows)} Q={len(rows)} "
        f"Top1={100*test['top1']:.2f}% Top5={100*test['top5']:.2f}% "
        f"MRR={test['mrr']:.4f} Hung={100*test['hungarian']:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
