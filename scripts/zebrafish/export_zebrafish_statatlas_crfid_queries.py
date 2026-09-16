#!/usr/bin/env python3
"""Replay locked StatAtlas/CRF-ID tests and export canonical query records."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from mprt_net.data import PairIndex, WormCache, build_pair_targets
from scripts.zebrafish import evaluate_zebrafish_statatlas_crfid_lofo8 as native
from scripts.zebrafish.query_record_io import QUERY_COLUMNS, records_from_score_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--methods", nargs="+", choices=native.METHODS, default=list(native.METHODS))
    args = parser.parse_args()

    for fold in args.folds:
        fold_root = args.data_root / f"fold_{fold}"
        cache = WormCache(activity_length=128)
        index = PairIndex(fold_root, "test", min_shared=20)
        native_pairs = native.load_pairs(fold_root, "test", cache)
        if len(index.pairs) != len(native_pairs):
            raise RuntimeError(f"fold {fold}: pair-order mismatch")

        samples = []
        for path_a, path_b in index.pairs:
            samples.append(build_pair_targets(cache.get(path_a), cache.get(path_b)))

        for method in args.methods:
            lock_path = args.run_root / "locks" / f"fold_{fold}_{method}.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            native.verify_lock(lock, fold_root, fold, method)
            cfg, state = lock["selected_config"], lock["train_state"]
            rows = []
            for pair_index, (pair, sample_bundle) in enumerate(zip(native_pairs, samples)):
                sample_a, sample_b, targets = sample_bundle
                if method == "statatlas_pair":
                    ab, ba = native.statatlas_scores(pair, state, cfg)
                else:
                    ab, ba = native.crfid_scores(pair, state, cfg)
                hs = 0.5 * (ab + ba.T)
                rows.extend(records_from_score_matrix(
                    method=method, fold=fold, seed=None, pair_index=pair_index,
                    pair_id=pair.uid_a.rsplit("__q", 1)[0], score=ab,
                    reverse_score=ba, hungarian_score=hs, q_uid=pair.uid_a,
                    r_uid=pair.uid_b, q_ids=sample_a.cell_ids,
                    r_ids=sample_b.cell_ids, row_target=pair.row_target,
                    col_target=pair.col_target,
                ))
                print(f"fold={fold} method={method} pair={pair_index + 1}/{len(native_pairs)}", flush=True)

            expected = native.EXPECTED_TEST_QUERIES[fold]
            if len(rows) != expected:
                raise RuntimeError(f"fold {fold}/{method}: {len(rows)} != {expected}")
            path = args.output_root / method / f"fold_{fold}.csv.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=QUERY_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()

