#!/usr/bin/env python3
"""Replay locked Vanilla-FGW test folds and export per-query predictions."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.zebrafish import evaluate_zebrafish_vanilla_fgw_lofo8 as native
from scripts.zebrafish.query_record_io import QUERY_COLUMNS, records_from_score_matrix


def target_arrays(q: dict, r: dict) -> tuple[np.ndarray, np.ndarray]:
    qm, rm = native.unique_supervised_map(q), native.unique_supervised_map(r)
    row = np.full(len(q["ids"]), -1, dtype=np.int64)
    col = np.full(len(r["ids"]), -1, dtype=np.int64)
    for identity in sorted(set(qm) & set(rm)):
        row[qm[identity]] = rm[identity]
        col[rm[identity]] = qm[identity]
    return row, col


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    args = parser.parse_args()

    for fold in args.folds:
        lock = json.loads((args.run_root / f"fold_{fold}" / "LOCKED_BEFORE_TEST.json").read_text())
        mean = np.asarray(lock["train_mean"], dtype=np.float64)
        std = np.asarray(lock["train_std"], dtype=np.float64)
        alpha = float(lock["selected_alpha"])
        pairs = native.normalize_pairs(
            native.load_pairs(args.data_root / f"fold_{fold}", "test"), mean, std
        )
        rows = []
        for pair_index, pair in enumerate(pairs):
            q, r = pair["q"], pair["r"]
            score = native.fgw_score(q["xyz"], r["xyz"], alpha)
            row_target, col_target = target_arrays(q, r)
            rows.extend(records_from_score_matrix(
                method="vanilla_fgw", fold=fold, seed=None,
                pair_index=pair_index, pair_id=pair["pair_id"], score=score,
                q_uid=Path(q["path"]).stem, r_uid=Path(r["path"]).stem,
                q_ids=q["ids"], r_ids=r["ids"], row_target=row_target,
                col_target=col_target,
            ))
            print(f"fold={fold} pair={pair_index + 1}/{len(pairs)}", flush=True)
        if len(rows) != native.EXPECTED_Q[fold]:
            raise RuntimeError(f"fold {fold}: {len(rows)} != {native.EXPECTED_Q[fold]}")
        path = args.output_root / "vanilla_fgw" / f"fold_{fold}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QUERY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

