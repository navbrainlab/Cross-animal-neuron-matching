#!/usr/bin/env python3
"""Evaluate saved NGM-v2 zebrafish checkpoints and export query records."""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.zebrafish.query_record_io import QUERY_COLUMNS, records_from_score_matrix


def targets(native, q: dict, r: dict) -> tuple[np.ndarray, np.ndarray]:
    qm, rm = native.unique_supervised_map(q), native.unique_supervised_map(r)
    row = np.full(len(q["cell_id"]), -1, dtype=np.int64)
    col = np.full(len(r["cell_id"]), -1, dtype=np.int64)
    for identity in sorted(set(qm) & set(rm)):
        row[qm[identity]] = rm[identity]
        col[rm[identity]] = qm[identity]
    return row, col


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    sys.path.insert(0, str(args.official_root.resolve()))
    from scripts.zebrafish import run_ngmv2_zebrafish_lofo_fold as native
    device = torch.device(args.device)

    for fold in args.folds:
        run = args.run_root / f"fold_{fold}"
        payload = torch.load(run / "best.pt", map_location=device, weights_only=False)
        model = native.base.AtanasNGMv2(feature_dim=int(payload["feature_dim"])).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        norm = np.load(run / "train_normalization.npz")
        pairs = native.normalize_pairs(
            native.load_physical_pairs(args.data_root / f"fold_{fold}", "test"),
            norm["mean"], norm["std"],
        )
        rows = []
        with torch.no_grad():
            for pair_index, pair in enumerate(pairs):
                q, r = pair["q"], pair["r"]
                gq = native.base.make_graph(q["xyz"], device)
                gr = native.base.make_graph(r["xyz"], device)
                score = model(gq, gr)[0].detach().float().cpu().numpy()
                row_target, col_target = targets(native, q, r)
                rows.extend(records_from_score_matrix(
                    method="ngm_v2", fold=fold, seed=42,
                    pair_index=pair_index, pair_id=pair["pair_id"], score=score,
                    q_uid=Path(q["path"]).stem, r_uid=Path(r["path"]).stem,
                    q_ids=q["cell_id"], r_ids=r["cell_id"],
                    row_target=row_target, col_target=col_target,
                ))
                print(f"fold={fold} pair={pair_index + 1}/{len(pairs)}", flush=True)
        if len(rows) != native.EXPECTED_Q[fold]:
            raise RuntimeError(f"fold {fold}: {len(rows)} != {native.EXPECTED_Q[fold]}")
        path = args.output_root / "ngm_v2" / f"fold_{fold}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QUERY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
