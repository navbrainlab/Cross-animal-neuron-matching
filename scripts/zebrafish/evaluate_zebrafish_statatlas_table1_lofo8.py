#!/usr/bin/env python3
"""Table-1-equivalent StatAtlas evaluation for zebrafish LOFO8.

For each physical q/r pair, both endpoint time points are excluded and the
atlas is built from all remaining unique time points of that fish.  The atlas
trainer, label-free alignment, identity posterior, posterior-overlap score and
Hungarian convention are the same functions used by the Table 1 worm adapter.
Only the source of repeated reference observations differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.zebrafish import evaluate_zebrafish_statatlas_multireference_lofo8 as multi
from scripts.zebrafish.query_record_io import QUERY_COLUMNS, records_from_score_matrix


SCHEMA = "zebrafish-statatlas-table1-core-lofo8-v1"
METHOD = "statatlas"
DISPLAY = "StatAtlas"
OFFICIAL_COMMIT = multi.OFFICIAL_COMMIT
ATLAS_PARAMS = dict(multi.ATLAS_PARAMS)
CANDIDATE_CONFIGS = [dict(config) for config in multi.CANDIDATE_CONFIGS]
PAIR_RE = re.compile(r"^(?P<pair>.+)__(?P<side>[qr])\.npz$")


@dataclass(frozen=True)
class PairEpisode:
    pair_id: str
    q: multi.Timepoint
    r: multi.Timepoint
    references: tuple[multi.Timepoint, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {
        "evaluator": sha256_file(Path(__file__).resolve()),
        "multireference_loader": sha256_file(Path(multi.__file__).resolve()),
        "table1_statatlas_adapter": sha256_file(
            ROOT / "baselines/official/adapters/stat_atlas/evaluate_statatlas_official.py"
        ),
        "query_record_io": sha256_file(
            ROOT / "scripts/zebrafish/query_record_io.py"
        ),
    }


def ordered_hash(values: Iterable[Any]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fieldnames = list(columns) if columns is not None else list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_manifest(split_dir: Path) -> dict[str, Any]:
    entries = []
    digest = hashlib.sha256()
    for path in sorted(split_dir.glob("*.npz")):
        value = sha256_file(path)
        entries.append({"name": path.name, "sha256": value})
        digest.update(path.name.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return {
        "split": split_dir.name,
        "source_files": len(entries),
        "sha256": digest.hexdigest(),
        "entries": entries,
    }


def load_pair_episodes(split_dir: Path) -> list[PairEpisode]:
    by_fish = multi.load_unique_timepoints(split_dir)
    alias_to_record = {
        alias: record
        for records in by_fish.values()
        for record in records
        for alias in record.aliases
    }
    sides: dict[str, dict[str, str]] = {}
    for path in sorted(split_dir.glob("*.npz")):
        match = PAIR_RE.match(path.name)
        if match is None:
            raise RuntimeError(f"unexpected pair filename: {path.name}")
        sides.setdefault(match.group("pair"), {})[match.group("side")] = path.name

    episodes = []
    for pair_id, side_names in sorted(sides.items()):
        if set(side_names) != {"q", "r"}:
            raise RuntimeError(f"{pair_id}: incomplete q/r pair")
        q, r = alias_to_record[side_names["q"]], alias_to_record[side_names["r"]]
        if q.specimen_id != r.specimen_id or q.start_frame == r.start_frame:
            raise RuntimeError(f"{pair_id}: invalid endpoint time points")
        excluded = {q.start_frame, r.start_frame}
        references = tuple(
            record for record in by_fish[q.specimen_id]
            if record.start_frame not in excluded
        )
        if len(references) < 3:
            raise RuntimeError(f"{pair_id}: fewer than three leave-two-out references")
        if any(record.start_frame in excluded for record in references):
            raise AssertionError(f"{pair_id}: endpoint leaked into atlas references")
        episodes.append(PairEpisode(pair_id, q, r, references))
    if not episodes:
        raise RuntimeError(f"no physical pairs under {split_dir}")
    return episodes


def target_arrays(pair: PairEpisode) -> tuple[np.ndarray, np.ndarray]:
    r_lookup = {int(value): index for index, value in enumerate(pair.r.tracking_ids)}
    q_lookup = {int(value): index for index, value in enumerate(pair.q.tracking_ids)}
    row_target = np.asarray(
        [r_lookup.get(int(value), -1) for value in pair.q.tracking_ids], dtype=np.int64
    )
    col_target = np.asarray(
        [q_lookup.get(int(value), -1) for value in pair.r.tracking_ids], dtype=np.int64
    )
    return row_target, col_target


def endpoint_ids(pair_id: str, values: np.ndarray) -> list[str]:
    return [f"{pair_id}::cell{int(value)}" for value in values]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("zero eligible directed queries")
    n = len(rows)
    return {
        "queries": n,
        "top1": sum(int(row["top1_correct"]) for row in rows) / n,
        "top5": sum(int(row["top5_correct"]) for row in rows) / n,
        "mrr": sum(float(row["reciprocal_rank"]) for row in rows) / n,
        "hungarian": sum(int(row["hungarian_correct"]) for row in rows) / n,
    }


def evaluate_pairs(
    episodes: list[PairEpisode], configs: list[dict[str, Any]], split: str, fold: int,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[dict[str, Any]]]:
    all_rows = [[] for _ in configs]
    diagnostics = []
    adapter = multi.official_adapter()
    for pair_index, pair in enumerate(episodes):
        atlas, train_diagnostics = adapter.train_official_atlas(
            pair.references,
            min_counts=ATLAS_PARAMS["min_counts"],
            epsilon_pos=ATLAS_PARAMS["epsilon_pos"],
            n_iter=ATLAS_PARAMS["atlas_iterations"],
            min_train_labels_per_worm=ATLAS_PARAMS["min_train_labels_per_worm"],
        )
        mu_xyz = np.asarray(atlas["mu"], dtype=np.float64)[:, :3]
        alignment_cache: dict[tuple[str, float], Any] = {}
        posterior_cache: dict[tuple[str, float, float], np.ndarray] = {}
        for config_index, config in enumerate(configs):
            trim = float(config["icp_trim"])
            ridge = float(config["covariance_ridge_fraction"])
            for side, endpoint in (("q", pair.q), ("r", pair.r)):
                alignment_key = (side, trim)
                if alignment_key not in alignment_cache:
                    alignment_cache[alignment_key] = adapter.label_free_align(
                        endpoint.xyz, mu_xyz,
                        iterations=ATLAS_PARAMS["icp_iterations"],
                        trim_fraction=trim,
                    )
                posterior_key = (side, trim, ridge)
                if posterior_key not in posterior_cache:
                    posterior_cache[posterior_key] = adapter.atlas_identity_posterior(
                        alignment_cache[alignment_key].aligned_xyz,
                        atlas,
                        covariance_ridge_fraction=ridge,
                    )
            # Exact Table 1 score_fn: identity-posterior overlap.
            score = posterior_cache[("q", trim, ridge)] @ posterior_cache[("r", trim, ridge)].T
            row_target, col_target = target_arrays(pair)
            rows = records_from_score_matrix(
                method=METHOD,
                fold=fold,
                seed=None,
                pair_index=pair_index,
                pair_id=pair.pair_id,
                score=score,
                q_uid=f"{pair.pair_id}__q",
                r_uid=f"{pair.pair_id}__r",
                q_ids=endpoint_ids(pair.pair_id, pair.q.tracking_ids),
                r_ids=endpoint_ids(pair.pair_id, pair.r.tracking_ids),
                row_target=row_target,
                col_target=col_target,
            )
            all_rows[config_index].extend(rows)
        diagnostics.append({
            "pair_index": pair_index,
            "pair_id": pair.pair_id,
            "specimen_id": pair.q.specimen_id,
            "q_start_frame": pair.q.start_frame,
            "r_start_frame": pair.r.start_frame,
            "reference_count": len(pair.references),
            "reference_start_frames": [record.start_frame for record in pair.references],
            "endpoints_used_in_atlas": False,
            "endpoint_truth_used_in_scoring": False,
            "atlas_identities": len(atlas["names"]),
            "atlas_training": train_diagnostics,
        })
        print(
            f"  {split} pair={pair_index + 1}/{len(episodes)} "
            f"{pair.pair_id} refs={len(pair.references)} atlas_ids={len(atlas['names'])}",
            flush=True,
        )
    return [summarize(rows) for rows in all_rows], all_rows, diagnostics


def pair_manifest_rows(episodes: list[PairEpisode], split: str) -> list[dict[str, Any]]:
    rows = []
    for pair_index, pair in enumerate(episodes):
        rows.append({
            "split": split,
            "pair_index": pair_index,
            "pair_id": pair.pair_id,
            "specimen_id": pair.q.specimen_id,
            "q_start_frame": pair.q.start_frame,
            "r_start_frame": pair.r.start_frame,
            "reference_count": len(pair.references),
            "reference_uids_json": json.dumps([record.worm_id for record in pair.references]),
            "reference_start_frames_json": json.dumps([record.start_frame for record in pair.references]),
            "q_and_r_excluded_from_atlas": True,
        })
    return rows


def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = row["validation_metrics"]
    return (
        metrics["top1"], metrics["mrr"], metrics["hungarian"], metrics["top5"],
        -int(row["candidate_index"]),
    )


def select_fold(args: argparse.Namespace, fold: int) -> None:
    fold_root = args.data_root / f"fold_{fold}" / "matching"
    print(f"[SELECT] fold={fold}: opening VAL only", flush=True)
    val = load_pair_episodes(fold_root / "val")
    metrics, _, diagnostics = evaluate_pairs(val, CANDIDATE_CONFIGS, "val", fold)
    candidates = [
        {"candidate_index": index, "config": config, "validation_metrics": metrics[index]}
        for index, config in enumerate(CANDIDATE_CONFIGS)
    ]
    winner = max(candidates, key=selection_key)
    lock = {
        "schema": SCHEMA,
        "status": "LOCKED_BEFORE_TEST",
        "fold": fold,
        "method": METHOD,
        "display_name": DISPLAY,
        "seed": None,
        "deterministic": True,
        "official_repository": "https://github.com/amin-nejat/stat-atlas",
        "official_commit": OFFICIAL_COMMIT,
        "source_hashes": source_hashes(),
        "selection_split": "val",
        "test_opened_during_selection": False,
        "reference_protocol": "same-fish leave-two-timepoints-out per physical q/r pair",
        "endpoint_truth_used_in_atlas_or_scoring": False,
        "table1_core": {
            "atlas": "train_official_atlas",
            "alignment": "label_free_align",
            "posterior": "atlas_identity_posterior",
            "score_fn": "P_q @ P_r.T",
            "hungarian": "linear_sum_assignment(-(P_q @ P_r.T))",
        },
        "atlas_params": ATLAS_PARAMS,
        "candidate_order": CANDIDATE_CONFIGS,
        "selected_candidate_index": winner["candidate_index"],
        "selected_config": winner["config"],
        "candidate_results": candidates,
        "train_manifest": split_manifest(fold_root / "train"),
        "val_manifest": split_manifest(fold_root / "val"),
        "val_pair_count": len(val),
        "val_diagnostics_hash": ordered_hash(diagnostics),
    }
    path = args.run_root / "locks" / f"fold_{fold}.json"
    write_json(path, lock)
    print(f"[LOCKED] {path}: {winner['config']}", flush=True)


def verify_lock(lock: dict[str, Any], fold_root: Path, fold: int) -> None:
    if lock.get("schema") != SCHEMA or int(lock.get("fold", -1)) != fold:
        raise RuntimeError("lock schema/fold mismatch")
    if lock.get("status") != "LOCKED_BEFORE_TEST":
        raise RuntimeError("lock was not completed before test")
    if lock.get("test_opened_during_selection") is not False:
        raise RuntimeError("selection leakage declaration failed")
    if lock.get("source_hashes") != source_hashes():
        raise RuntimeError("implementation changed after selection; rerun select")
    for split in ("train", "val"):
        current = split_manifest(fold_root / split)["sha256"]
        if current != lock[f"{split}_manifest"]["sha256"]:
            raise RuntimeError(f"{split} data changed after selection")


def test_fold(args: argparse.Namespace, fold: int) -> dict[str, Any]:
    fold_root = args.data_root / f"fold_{fold}" / "matching"
    lock_path = args.run_root / "locks" / f"fold_{fold}.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_lock(lock, fold_root, fold)
    print(f"[TEST] fold={fold}: lock verified; opening TEST", flush=True)
    episodes = load_pair_episodes(fold_root / "test")
    metrics, rows_by_config, diagnostics = evaluate_pairs(
        episodes, [dict(lock["selected_config"])], "test", fold
    )
    output = args.run_root / "test" / f"fold_{fold}"
    write_csv(output / "query_predictions.csv", rows_by_config[0], QUERY_COLUMNS)
    write_csv(output / "pair_manifest.csv", pair_manifest_rows(episodes, "test"))
    write_json(output / "pair_diagnostics.json", diagnostics)
    result = {
        "schema": SCHEMA,
        "fold": fold,
        "method": METHOD,
        "display_name": DISPLAY,
        "seed": None,
        "deterministic": True,
        "reference_protocol": "same-fish leave-two-timepoints-out per physical q/r pair",
        "selected_config": lock["selected_config"],
        "metrics": metrics[0],
        "physical_pairs": len(episodes),
        "q_and_r_timepoints_used_in_atlas": False,
        "endpoint_truth_used_in_scoring": False,
        "endpoint_truth_used_for_metrics_only": True,
        "test_manifest": split_manifest(fold_root / "test"),
        "lock_sha256": sha256_file(lock_path),
        "source_hashes": source_hashes(),
    }
    write_json(output / "metrics.json", result)
    return result


def aggregate(args: argparse.Namespace) -> None:
    results = [
        json.loads(
            (args.run_root / "test" / f"fold_{fold}" / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for fold in args.folds
    ]
    output = {
        "schema": SCHEMA,
        "method": METHOD,
        "display_name": DISPLAY,
        "seed": None,
        "deterministic": True,
        "folds": len(results),
        "aggregation": "unweighted mean and sample SD across held-out fish",
        "sample_sd_ddof": 1,
        "reference_protocol": "same-fish leave-two-timepoints-out per physical q/r pair",
        "metrics": {},
    }
    for name in ("top1", "top5", "mrr", "hungarian"):
        values = np.asarray([row["metrics"][name] for row in results], dtype=np.float64)
        output["metrics"][name] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "fold_values": values.tolist(),
        }
    write_json(args.run_root / "aggregate.json", output)
    print(json.dumps(output, indent=2), flush=True)


def audit(args: argparse.Namespace) -> None:
    rows = []
    for fold in args.folds:
        fold_root = args.data_root / f"fold_{fold}" / "matching"
        split_rows = {}
        for split in ("train", "val", "test"):
            episodes = load_pair_episodes(fold_root / split)
            split_rows[split] = {
                "physical_pairs": len(episodes),
                "min_references": min(len(pair.references) for pair in episodes),
                "max_references": max(len(pair.references) for pair in episodes),
                "endpoint_exclusion_pass": all(
                    pair.q.start_frame not in {r.start_frame for r in pair.references}
                    and pair.r.start_frame not in {r.start_frame for r in pair.references}
                    for pair in episodes
                ),
            }
        rows.append({"fold": fold, "splits": split_rows})
    report = {
        "schema": SCHEMA,
        "status": "PASS",
        "reference_protocol": "same-fish leave-two-timepoints-out per physical q/r pair",
        "stable_tracking_id": "original_row",
        "folds": rows,
    }
    write_json(args.run_root / "data_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    global multi
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "select", "test", "aggregate"))
    parser.add_argument("--data-root", type=Path, default=Path("../Data/Zebrafish_LOFO8_60m"))
    parser.add_argument(
        "--run-root", type=Path,
        default=Path("../runs/zebrafish_statatlas_table1_core_lofo8_v1"),
    )
    parser.add_argument("--statatlas-root", type=Path, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    args = parser.parse_args()
    if any(fold not in range(1, 9) for fold in args.folds):
        raise ValueError("--folds must contain values 1..8")
    args.data_root = args.data_root.resolve()
    args.run_root = args.run_root.resolve()
    multi.OFFICIAL_ROOT_OVERRIDE = (
        None if args.statatlas_root is None else args.statatlas_root.resolve()
    )
    if args.stage == "audit":
        audit(args)
    elif args.stage == "select":
        for fold in args.folds:
            select_fold(args, fold)
    elif args.stage == "test":
        for fold in args.folds:
            test_fold(args, fold)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
