#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
RUN_BASE = ROOT / "runs/fdnc_current_grouped_cv_v2/rld"
DATA_BASE = ROOT / "Data/Dunn_001623/cv5_grouped_v1"
MEDOID_EVAL = ROOT / "scripts/fair_identity/evaluate_train_reference_ensemble.py"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def find_terminal_summary(candidate: Path) -> tuple[Path, dict]:
    """
    Current fDNC trainer no longer necessarily writes old summary.json.
    Find a terminal JSON by semantic fields instead of filename.
    """
    matches: list[tuple[Path, dict]] = []
    for path in sorted(candidate.rglob("*.json")):
        obj = read_json(path)
        if not isinstance(obj, dict):
            continue
        bv = obj.get("best_validation")
        if not isinstance(bv, dict):
            continue
        required_top = {"dataset", "fold", "seed", "best_epoch"}
        if not required_top.issubset(obj):
            continue
        if "ranking_top1" not in bv or "mrr" not in bv:
            continue
        # Current trainer explicitly reports that test was not accessed.
        test_flag = obj.get("test_data_accessed", obj.get("test_data_used"))
        if test_flag is not False:
            continue
        matches.append((path, obj))

    if not matches:
        raise RuntimeError(
            f"{candidate}: no terminal training JSON found with "
            "dataset/fold/seed/best_epoch/best_validation and test_data_accessed=false"
        )

    # Prefer a root-level JSON, then shorter path; refuse if multiple summaries
    # disagree on the selected validation result.
    def signature(x: dict):
        bv = x["best_validation"]
        return (
            int(x["best_epoch"]),
            float(bv["ranking_top1"]),
            float(bv["mrr"]),
        )

    sig0 = signature(matches[0][1])
    disagreement = [
        (p, signature(obj))
        for p, obj in matches[1:]
        if signature(obj) != sig0
    ]
    if disagreement:
        # Multiple epoch/checkpoint summaries may exist; choose the one with the
        # largest best validation Top-1, then MRR, then epoch. This is still
        # validation-only.
        matches.sort(
            key=lambda po: (
                -float(po[1]["best_validation"]["ranking_top1"]),
                -float(po[1]["best_validation"]["mrr"]),
                -int(po[1]["best_epoch"]),
                len(po[0].parts),
                str(po[0]),
            )
        )
    else:
        matches.sort(key=lambda po: (len(po[0].parts), str(po[0])))

    return matches[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    fold = args.fold
    seed = args.seed
    if fold not in range(5):
        raise ValueError("--fold must be 0..4")

    run_root = RUN_BASE / f"fold{fold}/seed{seed}"
    cand_root = run_root / "candidates"
    if not cand_root.is_dir():
        raise FileNotFoundError(cand_root)

    expected_names = {
        "last1_lr1e-5",
        "last2_lr1e-5",
        "last1_lr5e-6",
        "last2_lr5e-6",
    }
    observed_names = {p.name for p in cand_root.iterdir() if p.is_dir()}
    missing = sorted(expected_names - observed_names)
    if missing:
        raise RuntimeError(f"Missing candidate directories: {missing}")

    rows = []
    for name in sorted(expected_names):
        candidate = cand_root / name
        best_pt = candidate / "best.pt"
        if not best_pt.is_file():
            raise FileNotFoundError(best_pt)

        summary_path, obj = find_terminal_summary(candidate)
        if str(obj["dataset"]).lower() != "rld":
            raise RuntimeError(f"{summary_path}: dataset={obj['dataset']} != rld")
        if int(obj["seed"]) != seed:
            raise RuntimeError(f"{summary_path}: seed={obj['seed']} != {seed}")
        if int(obj["fold"]) != fold + 1:
            raise RuntimeError(
                f"{summary_path}: trainer fold={obj['fold']} != expected {fold+1}"
            )

        bv = obj["best_validation"]
        row = {
            "candidate": name,
            "candidate_dir": str(candidate.resolve()),
            "summary_file": str(summary_path.resolve()),
            "best_checkpoint": str(best_pt.resolve()),
            "best_checkpoint_sha256": sha256(best_pt),
            "best_epoch": int(obj["best_epoch"]),
            "top1": float(bv["ranking_top1"]),
            "mrr": float(bv["mrr"]),
            "top5": float(bv.get("top5", bv.get("ranking_top5", float("nan")))),
            "assignment_top1": float(
                bv.get("assignment_top1", bv.get("hungarian_top1", float("nan")))
            ),
            "test_data_accessed": False,
            "official_pretrained_sha256": obj.get("official_pretrained_sha256"),
            "official_repo_commit": obj.get("official_repo_commit"),
        }
        rows.append(row)
        print(
            f"[CANDIDATE] {name:16s} "
            f"Top1={100*row['top1']:.3f}% "
            f"MRR={row['mrr']:.5f} "
            f"epoch={row['best_epoch']} "
            f"summary={summary_path.name}",
            flush=True,
        )

    # validation Top1 -> MRR -> stable candidate name
    rows.sort(key=lambda r: (-r["top1"], -r["mrr"], r["candidate"]))
    winner = rows[0]

    selected = run_root / "selected"
    selected.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner["best_checkpoint"], selected / "best.pt")
    selection = {
        "protocol": "fdnc_current_rld_cv5_grouped_v1",
        "current_grouped_fold": fold,
        "trainer_fold_argument": fold + 1,
        "seed": seed,
        "selection_split": "validation only",
        "primary_selection_metric": "ranking_top1",
        "tie_breaker": "mrr then candidate name",
        "selected": winner,
        "candidates": rows,
        "test_data_used_for_selection": False,
    }
    (selected / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    print("\n[SELECTED]")
    print(json.dumps(selection, indent=2))

    # Clean grouped-CV medoid evaluation.
    data_root = DATA_BASE / f"fold_{fold}"
    clean_out = run_root / "outer_test_medoid_template_v1"
    cache = run_root / "cache_clean_medoid"

    if clean_out.exists():
        shutil.rmtree(clean_out)
    clean_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(MEDOID_EVAL),
        "--method",
        "fdnc",
        "--method-label",
        "fDNC official fine-tuned (current RLD grouped CV)",
        "--fold-root",
        str(data_root),
        "--checkpoint",
        str(selected / "best.pt"),
        "--normalization",
        "zscore",
        "--device",
        "cuda",
        "--workers",
        "1",
        "--cache-dir",
        str(cache),
        "--output-dir",
        str(clean_out),
    ]
    print("\n[CLEAN MEDOID]", " ".join(cmd), flush=True)

    env = dict(__import__("os").environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    metrics_path = clean_out / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = read_json(metrics_path)

    print("\n" + "=" * 100)
    print("RECOVERY COMPLETE — CURRENT GROUPED-CV fDNC CLEAN")
    print("=" * 100)
    print("selection:", selected / "selection.json")
    print("metrics:  ", metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
