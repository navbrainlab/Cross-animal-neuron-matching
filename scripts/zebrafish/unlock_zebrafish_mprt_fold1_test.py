#!/usr/bin/env python3
"""Unlock the held-out zebrafish fold-1 test fish after model selection.

The script refuses to read or convert the test split until the validation-
selected checkpoint exists.  It reuses the audited conversion functions from
``prepare_zebrafish_mprt_fold1.py`` and records the checkpoint hash alongside
the converted test protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from scripts.zebrafish import prepare_zebrafish_mprt_fold1 as prep


EXPECTED_TEST_RECORDS = 32
EXPECTED_TEST_PAIRS = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/home/ubuntu/klb/nuclr/nuclr/Data/"
            "Zebrafish_LOFO8_joint_from_scratch/fold_1/features"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/ubuntu/klb/nuclr/nuclr/Data/"
            "Zebrafish_MPRT_LOFO8_60m/fold_1"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/ubuntu/klb/nuclr/nuclr/runs/mprt_v1_1/"
            "zebrafish_lofo/fold1/seed42/full/best.pt"
        ),
    )
    parser.add_argument("--min-shared", type=int, default=20)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    checkpoint = args.checkpoint.resolve()
    source_test = source_root / "test"
    output_test = output_root / "test"

    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Validation-selected checkpoint is missing; refusing to unlock test: "
            f"{checkpoint}"
        )
    if not (output_root / "protocol.json").is_file():
        raise FileNotFoundError(
            f"Audited train/val conversion is missing: {output_root / 'protocol.json'}"
        )
    if not source_test.is_dir():
        raise FileNotFoundError(f"Prepared locked test split is missing: {source_test}")

    prep.EXPECTED["test"] = {
        "records": EXPECTED_TEST_RECORDS,
        "pairs": EXPECTED_TEST_PAIRS,
    }

    if output_test.exists():
        report = prep.audit_split(output_test, "test", args.min_shared)
        print("[REUSE] Existing held-out test conversion passed audit")
    else:
        source_paths = sorted(source_test.glob("*.npz"))
        if len(source_paths) != EXPECTED_TEST_RECORDS:
            raise RuntimeError(
                f"Expected {EXPECTED_TEST_RECORDS} test records, "
                f"found {len(source_paths)} in {source_test}"
            )

        building = output_root / "test.building"
        if building.exists():
            raise FileExistsError(f"Stale test build directory exists: {building}")
        building.mkdir(parents=True)
        try:
            for source in source_paths:
                prep.convert_record(source, building / source.name)
            report = prep.audit_split(building, "test", args.min_shared)
            building.rename(output_test)
        except Exception:
            shutil.rmtree(building, ignore_errors=True)
            raise

    protocol = {
        "protocol": "zebrafish_longitudinal_LOFO_fold1_locked_test",
        "matching_unit": "same fish, q/r windows separated by 60 minutes",
        "held_out_test_fish": "func_20150410",
        "checkpoint_selected_using": "validation fish func_20150417 only",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "activity_points": 128,
        "minimum_shared": args.min_shared,
        "test": report,
    }
    protocol_path = output_root / "test_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(protocol, indent=2, ensure_ascii=False))
    print("ZEBRAFISH MPRT FOLD1 LOCKED TEST UNLOCK: PASS")
    print("Output:", output_test)


if __name__ == "__main__":
    main()
