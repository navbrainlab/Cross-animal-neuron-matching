#!/usr/bin/env python3
"""Export exact query/atlas relation tensors from the locked NeurID models.

The export is fold-pure.  ``R_atlas`` comes from the static, training-only
identity-anchored atlas checkpoint.  ``R_Q`` is produced by the same frozen
population encoder for each held-out test animal.  Atlas pair support is
reconstructed with the exact unique-supervised-identity rule used when the
atlas was built.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO / "mprt_net_v1_1"
DATA_ROOTS = {
    "atanas": REPO / "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
    "rld": REPO / "Data/Dunn_001623/cv5_grouped_v1",
}
DEFAULT_CHECKPOINT_ROOT = REPO / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1"
DEFAULT_OUTPUT_ROOT = REPO / "runs/neurid_query_atlas_relation_export_cv5_seed42_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def source_row_indices(path: Path) -> np.ndarray:
    """Return original NPZ rows retained by ``mprt_net.data.load_worm``."""

    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"])
        keep = np.isfinite(xyz).all(axis=1)
        if "valid_xyz_mask" in data.files:
            valid = np.asarray(data["valid_xyz_mask"], dtype=bool)
            if valid.shape != keep.shape:
                raise ValueError(f"valid_xyz_mask shape mismatch in {path}")
            keep &= valid
    return np.flatnonzero(keep).astype(np.int32)


def ordered_identities(raw_mapping: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    identity_to_slot = {str(key): int(value) for key, value in raw_mapping.items()}
    expected = set(range(len(identity_to_slot)))
    if set(identity_to_slot.values()) != expected:
        raise ValueError("Atlas slots are not a contiguous zero-based permutation")
    identities = [""] * len(identity_to_slot)
    for identity, slot in identity_to_slot.items():
        identities[slot] = identity
    return identities, identity_to_slot


def reconstruct_observation_support(
    train_files: list[Path],
    cache: Any,
    identity_to_slot: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Count training animals that jointly observe each identity pair."""

    from mprt_net.data import unique_identity_map

    size = len(identity_to_slot)
    identity_count = np.zeros(size, dtype=np.int32)
    pair_count = np.zeros((size, size), dtype=np.int32)
    for path in train_files:
        identity_map = unique_identity_map(cache.get(path))
        slots = np.asarray(
            sorted(identity_to_slot[x] for x in identity_map if x in identity_to_slot),
            dtype=np.int64,
        )
        if slots.size == 0:
            continue
        identity_count[slots] += 1
        pair_count[np.ix_(slots, slots)] += 1
    return identity_count, pair_count


def unique_supervised_mask(sample: Any) -> np.ndarray:
    labels = list(map(str, sample.cell_ids))
    supervised = sample.supervised_mask.detach().cpu().numpy().astype(bool)
    counts = Counter(label for label, valid in zip(labels, supervised) if valid)
    return np.asarray(
        [valid and counts[label] == 1 for label, valid in zip(labels, supervised)],
        dtype=bool,
    )


def validate_atlas_support(
    *,
    atlas: Any,
    checkpoint: dict[str, Any],
    identity_count: np.ndarray,
    pair_count: np.ndarray,
    dataset: str,
    fold: int,
) -> dict[str, Any]:
    relation = atlas.relations.detach().cpu().numpy()
    pair_valid = pair_count > 0
    coverage = float(pair_valid.mean())
    recorded = float(checkpoint.get("atlas_build", {}).get("relation_coverage", coverage))
    if not np.isclose(coverage, recorded, rtol=0.0, atol=1e-7):
        raise RuntimeError(
            f"{dataset}/fold{fold}: reconstructed relation coverage {coverage} "
            f"does not match checkpoint {recorded}"
        )
    unsupported = relation[~pair_valid]
    if unsupported.size and not np.allclose(unsupported, 0.0, rtol=0.0, atol=1e-7):
        raise RuntimeError(f"{dataset}/fold{fold}: unsupported atlas pairs are nonzero")

    stored_support = atlas.support.detach().cpu().numpy()
    expected_support = identity_count / max(int(identity_count.max()), 1)
    if not np.allclose(stored_support, expected_support, rtol=1e-6, atol=1e-7):
        raise RuntimeError(f"{dataset}/fold{fold}: atlas identity support mismatch")

    if atlas.relation_count is not None:
        stored_count = atlas.relation_count.detach().cpu().numpy()
        if not np.array_equal(stored_count.astype(np.int64), pair_count.astype(np.int64)):
            raise RuntimeError(f"{dataset}/fold{fold}: stored relation counts mismatch")
    return {
        "relation_coverage": coverage,
        "valid_pair_entries": int(pair_valid.sum()),
        "unobserved_pair_entries": int((~pair_valid).sum()),
        "total_pair_entries": int(pair_valid.size),
        "unobserved_pair_fraction": float((~pair_valid).mean()),
        "minimum_identity_observations": int(identity_count.min()),
        "maximum_identity_observations": int(identity_count.max()),
    }


def export_fold(
    *,
    dataset: str,
    fold: int,
    seed: int,
    checkpoint_root: Path,
    output_root: Path,
    device: torch.device,
    activity_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from mprt_net.data import WormCache, split_files
    from mprt_net.evaluate import load_checkpoint

    dataset_root = DATA_ROOTS[dataset] / f"fold_{fold}"
    checkpoint_path = (
        checkpoint_root
        / dataset
        / f"fold{fold}"
        / f"seed{seed}"
        / "static_atlas/anchored_pure.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_hash = sha256(checkpoint_path)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    model.eval()
    if not model.atlas_is_initialized:
        raise RuntimeError(f"Atlas is not initialized in {checkpoint_path}")
    raw_mapping = checkpoint.get("atlas_identity_to_slot")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise RuntimeError(f"Missing atlas_identity_to_slot in {checkpoint_path}")
    atlas_identities, identity_to_slot = ordered_identities(raw_mapping)
    atlas = model.atlas_encoding()
    if atlas.relations.shape[:2] != (len(atlas_identities), len(atlas_identities)):
        raise RuntimeError(f"Atlas relation shape/mapping mismatch in {checkpoint_path}")

    train_files = split_files(dataset_root, "train")
    test_files = split_files(dataset_root, "test")
    cache = WormCache(
        activity_length=activity_length,
        max_items=max(8, len(train_files)),
    )
    identity_count, pair_count = reconstruct_observation_support(
        train_files, cache, identity_to_slot
    )
    support_audit = validate_atlas_support(
        atlas=atlas,
        checkpoint=checkpoint,
        identity_count=identity_count,
        pair_count=pair_count,
        dataset=dataset,
        fold=fold,
    )

    fold_dir = output_root / dataset / f"fold_{fold}"
    atlas_path = fold_dir / "atlas.npz"
    atlas_relations = atlas.relations.detach().cpu().numpy().astype(np.float32)
    relation_valid = pair_count > 0
    atomic_npz(
        atlas_path,
        dataset=np.asarray(dataset),
        fold=np.asarray(fold, dtype=np.int16),
        seed=np.asarray(seed, dtype=np.int32),
        checkpoint_epoch=np.asarray(int(checkpoint.get("epoch", -1)), dtype=np.int32),
        checkpoint_sha256=np.asarray(checkpoint_hash),
        atlas_identities=np.asarray(atlas_identities, dtype=np.str_),
        atlas_slots=np.arange(len(atlas_identities), dtype=np.int32),
        R_atlas=atlas_relations,
        identity_observation_count=identity_count,
        pair_observation_count=pair_count,
        pair_valid=relation_valid,
        atlas_support=atlas.support.detach().cpu().numpy().astype(np.float32),
        relation_dimension=np.asarray(atlas_relations.shape[-1], dtype=np.int16),
        relation_definition=np.asarray(
            "exact learned relation_field from frozen training-only identity-anchored atlas"
        ),
    )

    query_rows: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    with torch.inference_mode():
        for number, path in enumerate(test_files, start=1):
            sample_cpu = cache.get(path)
            uid = str(sample_cpu.uid)
            if uid in seen_uids:
                raise RuntimeError(f"Duplicate test animal UID in {dataset}/fold{fold}: {uid}")
            seen_uids.add(uid)
            encoding = model.encode_population(sample_cpu.to(device))
            relations = encoding.relations.detach().cpu().numpy().astype(np.float32)
            labels = np.asarray(sample_cpu.cell_ids, dtype=np.str_)
            supervised = sample_cpu.supervised_mask.detach().cpu().numpy().astype(bool)
            unique_mask = unique_supervised_mask(sample_cpu)
            original_rows = source_row_indices(path)
            if relations.shape[:2] != (len(labels), len(labels)):
                raise RuntimeError(f"Query relation/identity shape mismatch in {path}")
            if len(original_rows) != len(labels):
                raise RuntimeError(f"Query source-row mapping mismatch in {path}")
            identity_in_atlas = np.asarray(
                [label in identity_to_slot for label in labels], dtype=bool
            )
            query_path = fold_dir / "queries" / f"{uid}.npz"
            atomic_npz(
                query_path,
                dataset=np.asarray(dataset),
                fold=np.asarray(fold, dtype=np.int16),
                seed=np.asarray(seed, dtype=np.int32),
                test_animal_id=np.asarray(uid),
                source_path=np.asarray(str(path.resolve())),
                node_index=np.arange(len(labels), dtype=np.int32),
                source_row_index=original_rows,
                query_identities=labels,
                supervised_mask=supervised,
                unique_supervised_mask=unique_mask,
                identity_in_atlas=identity_in_atlas,
                R_Q=relations,
                relation_dimension=np.asarray(relations.shape[-1], dtype=np.int16),
                relation_definition=np.asarray(
                    "exact learned relation_field from the frozen checkpoint population encoder"
                ),
            )
            query_rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "seed": seed,
                    "test_animal_id": uid,
                    "query_nodes": len(labels),
                    "supervised_nodes": int(supervised.sum()),
                    "unique_supervised_nodes": int(unique_mask.sum()),
                    "unique_supervised_in_atlas": int((unique_mask & identity_in_atlas).sum()),
                    "relation_dim": int(relations.shape[-1]),
                    "query_file": str(query_path.relative_to(output_root)),
                    "query_file_sha256": sha256(query_path),
                    "source_path": str(path.resolve()),
                }
            )
            print(
                f"export dataset={dataset} fold={fold} "
                f"test={number:02d}/{len(test_files):02d} uid={uid} nodes={len(labels)}",
                flush=True,
            )

    fold_row = {
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "test_animals": len(query_rows),
        "atlas_identities": len(atlas_identities),
        "relation_dim": int(atlas_relations.shape[-1]),
        "train_animals": len(train_files),
        "relation_coverage": support_audit["relation_coverage"],
        "valid_pair_entries": support_audit["valid_pair_entries"],
        "unobserved_pair_entries": support_audit["unobserved_pair_entries"],
        "total_pair_entries": support_audit["total_pair_entries"],
        "unobserved_pair_fraction": support_audit["unobserved_pair_fraction"],
        "min_identity_observations": support_audit["minimum_identity_observations"],
        "max_identity_observations": support_audit["maximum_identity_observations"],
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "atlas_file": str(atlas_path.relative_to(output_root)),
        "atlas_file_sha256": sha256(atlas_path),
    }
    return fold_row, query_rows


def schema_text() -> str:
    return """# NeurID query/atlas relation export — CV5 seed42

This directory contains exact relation tensors from the frozen static
population-atlas checkpoints used by the main-table NeurID evaluation.

## Files

- `fold_manifest.csv`: fold, checkpoint, atlas and support provenance.
- `query_manifest.csv`: one row per held-out test animal.
- `{dataset}/fold_{f}/atlas.npz`: fold-specific training-only atlas.
- `{dataset}/fold_{f}/queries/{test_animal_id}.npz`: held-out query animal.
- `AUDIT.json`: shape, support and provenance validation summary.

## Array schema

`atlas.npz` contains `atlas_identities[A]`, `R_atlas[A,A,D]`,
`identity_observation_count[A]`, `pair_observation_count[A,A]` and
`pair_valid[A,A] = pair_observation_count > 0`. Counts are numbers of distinct
outer-training animals in which the identity or identity pair is jointly
unique and supervised.

Each query NPZ contains `query_identities[N]`, `R_Q[N,N,D]`, `node_index[N]`,
`source_row_index[N]`, `supervised_mask[N]`, `unique_supervised_mask[N]` and
`identity_in_atlas[N]`. `test_animal_id`, `dataset`, `fold` and `seed` are also
stored inside every file.

The exact model relation is vector-valued (`D=8` in these checkpoints). No
scalar reduction is applied. Both tensor axes follow the accompanying identity
order exactly. All finite-coordinate population-context neurons are retained;
the masks indicate which labels are valid for supervised evaluation.
"""


def parse_csv_choices(value: str, allowed: Iterable[str]) -> list[str]:
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected).difference(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown choices: {unknown}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="atanas,rld")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    datasets = parse_csv_choices(args.datasets, DATA_ROOTS)
    folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    if not datasets or not folds or any(fold not in range(5) for fold in folds):
        raise ValueError("Select at least one dataset and folds from 0 through 4")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty export: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(PACKAGE_ROOT))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    fold_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for fold in folds:
            fold_row, fold_queries = export_fold(
                dataset=dataset,
                fold=fold,
                seed=args.seed,
                checkpoint_root=args.checkpoint_root.resolve(),
                output_root=output_root,
                device=device,
                activity_length=args.activity_length,
            )
            fold_rows.append(fold_row)
            query_rows.extend(fold_queries)

    atomic_csv(output_root / "fold_manifest.csv", fold_rows, list(fold_rows[0]))
    atomic_csv(output_root / "query_manifest.csv", query_rows, list(query_rows[0]))
    atomic_text(output_root / "README.md", schema_text())
    audit = {
        "protocol": "neurid_static_population_atlas_query_relation_export_cv5_seed42_v1",
        "datasets": datasets,
        "folds": folds,
        "seed": args.seed,
        "device": str(device),
        "relation_tensor_semantics": "exact vector-valued learned relation_field; no scalar reduction",
        "atlas_scope": "outer-train only, fold-specific, static identity-anchored atlas",
        "query_scope": "all finite-coordinate nodes in each held-out test animal",
        "support_definition": "number of distinct outer-train animals jointly containing each pair as unique supervised identities",
        "folds_exported": len(fold_rows),
        "test_animals_exported": len(query_rows),
        "query_nodes_exported": sum(int(row["query_nodes"]) for row in query_rows),
        "support_checks_passed": len(fold_rows),
        "export_script": str(Path(__file__).resolve()),
        "export_script_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(output_root / "AUDIT.json", audit)

    checksum_targets = sorted(
        path for path in output_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(output_root)}"
        for path in checksum_targets
    ]
    atomic_text(output_root / "SHA256SUMS.txt", "\n".join(checksum_lines) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
