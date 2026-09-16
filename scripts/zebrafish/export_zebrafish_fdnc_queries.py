#!/usr/bin/env python3
"""Evaluate saved zebrafish fDNC checkpoints and export query predictions.

``--implementation-script`` is the audited from-scratch zebrafish fDNC runner
used to create the checkpoints.  Loading it dynamically keeps this exporter
compatible with an official-source checkout or the archived source snapshot.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.zebrafish.query_record_io import QUERY_COLUMNS, records_from_score_matrix


EXPECTED = {1: 3536, 2: 3954, 3: 768, 4: 1820, 5: 1464, 6: 938, 7: 2492, 8: 2184}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("_zebrafish_fdnc_native", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def targets(native, q, r):
    qm, rm = native.unique_map(q.labels), native.unique_map(r.labels)
    row = np.full(len(q.labels), -1, dtype=np.int64)
    col = np.full(len(r.labels), -1, dtype=np.int64)
    for identity in sorted(set(qm) & set(rm)):
        row[qm[identity]] = rm[identity]
        col[rm[identity]] = qm[identity]
    return row, col


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-script", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    native = load_module(args.implementation_script)
    device = torch.device(args.device)

    for fold in args.folds:
        checkpoint_path = args.run_root / f"fold_{fold}" / "fdnc" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved = checkpoint["args"]
        model = native.FdncRegistrationBackbone(
            n_hidden=int(saved["n_hidden"]), n_layer=int(saved["n_layer"])
        ).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        pairs = native.load_pairs(
            args.prepared_root / f"fold_{fold}" / "features" / "test",
            str(saved["normalization"]),
        )
        rows = []
        with torch.no_grad():
            for pair_index, (q, r) in enumerate(pairs):
                qxyz = torch.as_tensor(q.xyz, dtype=torch.float32, device=device)
                rxyz = torch.as_tensor(r.xyz, dtype=torch.float32, device=device)
                qr = model.score_direction(rxyz, qxyz)[:, :len(r.labels)].cpu().numpy()
                rq = model.score_direction(qxyz, rxyz)[:, :len(q.labels)].cpu().numpy()
                row_target, col_target = targets(native, q, r)
                rows.extend(records_from_score_matrix(
                    method="fdnc", fold=fold, seed=42, pair_index=pair_index,
                    pair_id=q.pair_id, score=qr, reverse_score=rq,
                    hungarian_score=qr, reverse_hungarian_score=rq,
                    q_uid=Path(q.path).stem, r_uid=Path(r.path).stem,
                    q_ids=q.labels, r_ids=r.labels, row_target=row_target,
                    col_target=col_target,
                ))
                print(f"fold={fold} pair={pair_index + 1}/{len(pairs)}", flush=True)
        if len(rows) != EXPECTED[fold]:
            raise RuntimeError(f"fold {fold}: {len(rows)} != {EXPECTED[fold]}")
        path = args.output_root / "fdnc" / f"fold_{fold}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QUERY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
