#!/usr/bin/env python3
"""Export the formal Atanas seed-42 Full/w-o-Activity predictions for plotting.

The formal evaluator reports only eligible labelled queries.  This exporter
replays the same static-atlas forward pass for every spatially valid test
neuron so that unlabelled neurons can be retained as plot context.  For the
official evaluation subset, the archived evaluator prediction slots are used
as the authority and replayed slots must agree exactly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch


FIELDS = (
    "fold",
    "animal_id",
    "neuron_row",
    "x",
    "y",
    "z",
    "gt_identity",
    "pred_geometry",
    "pred_full",
    "is_evaluated",
)
INVALID_IDS = {"", "nan", "none", "null", "unknown", "unk"}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo
            / "runs/visualization_export_atanas_seed42"
            / "neurid_query_neurons_atanas_cv5_seed42.csv"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    return parser.parse_args()


def read_formal_rows(path: Path) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["uid"], int(row["node_index"]))
            if key in result:
                raise RuntimeError(f"duplicate formal query in {path}: {key}")
            result[key] = int(row["prediction_slot"])
    if not result:
        raise RuntimeError(f"no formal query rows in {path}")
    return result


def raw_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        xyz = np.asarray(z["xyz"], dtype=np.float32)
        ids = np.asarray(z["cell_id"]).astype(str)
        valid = np.isfinite(xyz).all(axis=1)
        if "valid_xyz_mask" in z.files:
            valid &= np.asarray(z["valid_xyz_mask"], dtype=bool)
    if xyz.shape != (len(ids), 3):
        raise RuntimeError(f"row mismatch in {path}: xyz={xyz.shape}, ids={ids.shape}")
    return xyz, ids, valid


@torch.inference_mode()
def predict_slots(model, atlas, sample, device: torch.device) -> list[int]:
    query = model.encode_population(sample.to(device))
    output = model.match_encodings(query, atlas)
    real = output.row_conditional[:, :-1]
    return real.argmax(dim=1).detach().cpu().tolist()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    package_root = repo / "neurid"
    sys.path.insert(0, str(package_root))

    from mprt_net.data import WormCache, split_files
    from mprt_net.evaluate import load_checkpoint

    torch.set_num_threads(max(1, args.num_threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset_cv = repo / "data/atanas"
    component = repo / "runs/mprt_v1_1_component_ablation_cv5x3_v1/atanas"
    full_runs = repo / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1/atanas"
    output_rows: list[dict[str, object]] = []
    evaluated_total = 0

    for fold in range(5):
        cell = component / f"fold{fold}/seed42"
        full_checkpoint = full_runs / f"fold{fold}/seed42/static_atlas/anchored_pure.pt"
        geometry_checkpoint = cell / "geometry_only/static_atlas/anchored_pure.pt"
        full_formal = read_formal_rows(cell / "metrics/test/full_queries.csv")
        geometry_formal = read_formal_rows(cell / "metrics/test/geometry_only_queries.csv")
        if set(full_formal) != set(geometry_formal):
            raise RuntimeError(f"formal Full/geometry cohorts differ in fold {fold}")

        full_model, full_state = load_checkpoint(full_checkpoint, device)
        geometry_model, geometry_state = load_checkpoint(geometry_checkpoint, device)
        full_model.eval()
        geometry_model.eval()
        full_mapping = {str(k): int(v) for k, v in full_state["atlas_identity_to_slot"].items()}
        geometry_mapping = {
            str(k): int(v) for k, v in geometry_state["atlas_identity_to_slot"].items()
        }
        if full_mapping != geometry_mapping:
            raise RuntimeError(f"Full/geometry candidate mappings differ in fold {fold}")
        slot_to_identity = {slot: identity for identity, slot in full_mapping.items()}
        if len(slot_to_identity) != len(full_mapping):
            raise RuntimeError(f"non-unique atlas slots in fold {fold}")

        full_build = full_state.get("atlas_build", {})
        geometry_build = geometry_state.get("atlas_build", {})
        for key in ("recordings", "atlas_size", "node_observations"):
            if full_build.get(key) != geometry_build.get(key):
                raise RuntimeError(f"Full/geometry atlas-build field {key!r} differs in fold {fold}")

        full_atlas = full_model.atlas_encoding()
        geometry_atlas = geometry_model.atlas_encoding()
        cache = WormCache(activity_length=512, max_items=8)
        fold_seen: set[tuple[str, int]] = set()

        for path in split_files(dataset_cv / f"fold_{fold}", "test"):
            sample = cache.get(path)
            xyz, raw_ids, valid = raw_arrays(path)
            retained_rows = np.flatnonzero(valid).tolist()
            if len(retained_rows) != sample.num_nodes:
                raise RuntimeError(f"loader/raw valid-row mismatch in {path}")
            if tuple(raw_ids[valid].astype(str)) != sample.cell_ids:
                raise RuntimeError(f"loader/raw identity-order mismatch in {path}")

            full_slots = predict_slots(full_model, full_atlas, sample, device)
            geometry_slots = predict_slots(geometry_model, geometry_atlas, sample, device)
            if len(full_slots) != sample.num_nodes or len(geometry_slots) != sample.num_nodes:
                raise RuntimeError(f"prediction length mismatch in {path}")

            full_by_raw = dict(zip(retained_rows, full_slots))
            geometry_by_raw = dict(zip(retained_rows, geometry_slots))
            for sample_row, raw_row in enumerate(retained_rows):
                key = (sample.uid, sample_row)
                if key in full_formal:
                    # The archived formal files are authoritative.  An exact
                    # replay check protects against accidentally changing the
                    # model, atlas, candidate order, or Top-1 definition.
                    if full_slots[sample_row] != full_formal[key]:
                        raise RuntimeError(
                            f"Full replay differs from formal output in fold {fold}: {key}"
                        )
                    if geometry_slots[sample_row] != geometry_formal[key]:
                        raise RuntimeError(
                            f"geometry replay differs from formal output in fold {fold}: {key}"
                        )
                    fold_seen.add(key)

            for raw_row, (x, y, z) in enumerate(xyz.tolist()):
                raw_identity = str(raw_ids[raw_row]).strip()
                gt_identity = (
                    raw_identity if raw_identity.lower() not in INVALID_IDS else ""
                )
                sample_row = retained_rows.index(raw_row) if bool(valid[raw_row]) else None
                key = (sample.uid, sample_row) if sample_row is not None else None
                is_evaluated = key in full_formal if key is not None else False
                output_rows.append(
                    {
                        "fold": fold,
                        "animal_id": sample.uid,
                        "neuron_row": raw_row,
                        "x": format(float(x), ".9g"),
                        "y": format(float(y), ".9g"),
                        "z": format(float(z), ".9g"),
                        "gt_identity": gt_identity,
                        "pred_geometry": (
                            slot_to_identity[geometry_by_raw[raw_row]]
                            if raw_row in geometry_by_raw
                            else ""
                        ),
                        "pred_full": (
                            slot_to_identity[full_by_raw[raw_row]]
                            if raw_row in full_by_raw
                            else ""
                        ),
                        "is_evaluated": str(is_evaluated).lower(),
                    }
                )

        if fold_seen != set(full_formal):
            missing = sorted(set(full_formal) - fold_seen)[:10]
            raise RuntimeError(f"formal rows not found in fold {fold}: {missing}")
        evaluated_total += len(fold_seen)
        print(
            f"fold={fold} animals={len(split_files(dataset_cv / f'fold_{fold}', 'test'))} "
            f"evaluated={len(fold_seen)} candidates={len(full_mapping)}",
            flush=True,
        )

    keys = [(row["fold"], row["animal_id"], row["neuron_row"]) for row in output_rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate output neuron keys")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"wrote {len(output_rows)} neurons ({evaluated_total} evaluated) to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
