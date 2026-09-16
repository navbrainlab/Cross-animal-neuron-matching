#!/usr/bin/env python3
"""Materialize the canonical LOFO8 test-pair and candidate-order record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from scripts.zebrafish.query_record_io import candidate_order_sha256
except ModuleNotFoundError:  # Direct execution from the repository root.
    from query_record_io import candidate_order_sha256


INVALID = {"", "nan", "none", "null", "unknown", "unk"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value) -> str:
    value = np.asarray(value).reshape(-1)[0]
    return value.decode() if isinstance(value, bytes) else str(value)


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"])
        ids = np.asarray(data["cell_id"]).astype(str)
        n = len(ids)
        mask = lambda key: np.asarray(data[key], dtype=bool) if key in data else np.ones(n, dtype=bool)
        candidate = np.isfinite(xyz).all(1) & mask("valid_xyz_mask")
        supervised = candidate & mask("labeled_mask") & mask("certain_mask") & mask("clean_mask")
        supervised &= np.asarray([x.strip().lower() not in INVALID for x in ids])
        pair_id, side = scalar(data["pair_id"]), scalar(data["side"]).lower()
    filtered_ids = [x.strip() for x in ids[candidate]]
    return {
        "path": path, "pair_id": pair_id, "side": side,
        "ids": filtered_ids, "supervised": supervised[candidate],
    }


def unique_ids(record: dict) -> set[str]:
    values = [identity for identity, ok in zip(record["ids"], record["supervised"]) if ok]
    counts = Counter(values)
    return {identity for identity, count in counts.items() if count == 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for fold in range(1, 9):
        groups = {}
        for path in sorted((args.data_root / f"fold_{fold}" / "test").glob("*.npz")):
            record = load(path)
            groups.setdefault(record["pair_id"], {})[record["side"]] = record
        for pair_index, pair_id in enumerate(sorted(groups)):
            if set(groups[pair_id]) != {"q", "r"}:
                raise RuntimeError(f"fold {fold}/{pair_id}: incomplete pair")
            q, r = groups[pair_id]["q"], groups[pair_id]["r"]
            shared = sorted(unique_ids(q) & unique_ids(r))
            rows.append({
                "fold": fold, "pair_index": pair_index, "pair_id": pair_id,
                "q_file": q["path"].name, "r_file": r["path"].name,
                "q_file_sha256": file_hash(q["path"]), "r_file_sha256": file_hash(r["path"]),
                "q_candidates": len(q["ids"]), "r_candidates": len(r["ids"]),
                "direct_matches": len(shared), "bidirectional_queries": 2 * len(shared),
                "q_candidate_order_sha256": candidate_order_sha256(q["ids"]),
                "r_candidate_order_sha256": candidate_order_sha256(r["ids"]),
                "q_candidate_order_json": json.dumps(q["ids"], ensure_ascii=False, separators=(",", ":")),
                "r_candidate_order_json": json.dumps(r["ids"], ensure_ascii=False, separators=(",", ":")),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} physical pairs and {sum(r['bidirectional_queries'] for r in rows)} queries")


if __name__ == "__main__":
    main()
