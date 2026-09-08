#!/usr/bin/env python3
"""
Prepare CURRENT final RLD candidate-score exports for natural population-size scaling.

Outputs
-------
MPRT Static (final atlas):
  runs/unified_benchmark/ours_static/rld/foldF/seed42/test_candidate_scores.csv

fDNC current-grouped fine-tuned:
  runs/accuracy_scaling_current_v1/fdnc_current_grouped/rld/foldF/seed42/test_candidate_scores.csv

The fDNC export uses the EXACT canonical query set defined by the freshly exported
MPRT Static score file. Queries whose GT is absent from the medoid template remain
in the CSV (with all available candidate scores), so missing-GT queries are counted
as incorrect by the downstream scaling evaluator rather than silently excluded.

No retraining. No test-time selection. Template is selected from outer-train only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
CV_ROOT = ROOT / "Data/Dunn_001623/cv5_grouped_v1"
MPRT_EXPORTER = ROOT / "scripts/scaling/export_unified_candidate_scores.py"
FDNC_ROOT = ROOT / "runs/fdnc_current_grouped_cv_v2/rld"
OUT_FDNC = ROOT / "runs/accuracy_scaling_current_v1/fdnc_current_grouped/rld"

FIELDS = (
    "query_uid",
    "group_uid",
    "gt_label",
    "candidate_label",
    "score",
    "assignment_score",
    "reference_uid",
)

INVALID = {"", "nan", "none", "null", "-1", "unknown", "unk", "?", "unlabeled", "unlabelled"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def norm_label(x):
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    s = str(x).strip()
    return "" if s.lower() in INVALID else s


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_canonical(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = str(row["query_uid"])
            meta = (str(row["group_uid"]), norm_label(row["gt_label"]))
            old = out.setdefault(qid, meta)
            if old != meta:
                raise RuntimeError(f"Canonical metadata conflict for {qid}: {old} vs {meta}")
    if not out:
        raise RuntimeError(f"No canonical queries in {path}")
    return out


def import_eval_module():
    path = ROOT / "scripts/fair_identity/evaluate_train_reference_ensemble.py"
    spec = importlib.util.spec_from_file_location("current_medoid_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_mprt_export(fold: int, seed: int, device: str, overwrite: bool):
    path = ROOT / f"runs/unified_benchmark/ours_static/rld/fold{fold}/seed{seed}/test_candidate_scores.csv"
    if overwrite and path.exists():
        path.unlink()
        audit = path.with_name("test_candidate_scores.audit.json")
        audit.unlink(missing_ok=True)

    if path.is_file() and path.stat().st_size > 0:
        print(f"[REUSE MPRT] {path}")
        return path

    cmd = [
        sys.executable,
        "-u",
        str(MPRT_EXPORTER),
        "--method", "ours_static",
        "--dataset", "rld",
        "--fold", str(fold),
        "--seed", str(seed),
        "--device", device,
    ]
    print("[EXPORT MPRT]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def unique_supervised_indices(labels, mask):
    counts = {}
    for x, keep in zip(labels, mask):
        s = norm_label(x)
        if bool(keep) and s:
            counts[s] = counts.get(s, 0) + 1
    return [
        i for i, (x, keep) in enumerate(zip(labels, mask))
        if bool(keep) and norm_label(x) and counts[norm_label(x)] == 1
    ]


def export_fdnc_current(fold: int, seed: int, device: torch.device, canonical_path: Path, mod, overwrite: bool):
    out = OUT_FDNC / f"fold{fold}" / f"seed{seed}" / "test_candidate_scores.csv"
    audit = out.with_name("test_candidate_scores.audit.json")
    if out.is_file() and not overwrite:
        print(f"[REUSE fDNC current] {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    audit.unlink(missing_ok=True)

    checkpoint = FDNC_ROOT / f"fold{fold}/seed{seed}/selected/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    fold_root = CV_ROOT / f"fold_{fold}"
    train_paths = mod.split_files(fold_root, "train")
    test_paths = mod.split_files(fold_root, "test")
    cache = mod.WormCache(512, max_items=len(train_paths) + len(test_paths))
    train_samples = [cache.get(p) for p in train_paths]
    test_samples = [cache.get(p) for p in test_paths]

    # EXACT same medoid rule as evaluate_train_reference_ensemble.py.
    normalized_train_xyz = [
        mod.normalize_xyz(sample.xyz.cpu().numpy()) for sample in train_samples
    ]
    medoid = mod.select_geometry_medoid(
        [sample.uid for sample in train_samples], normalized_train_xyz
    )
    template = train_samples[medoid.index]

    # Load current fine-tuned fDNC exactly as clean evaluator.
    from engines import evaluate_atanas_fdnc_unified as fdnc_loader
    from engines import train_atanas_fdnc_fold as fdnc_train

    model = fdnc_loader.load_checkpoint(checkpoint, device, 128, 6)
    model.eval()

    canonical = read_canonical(canonical_path)

    ref_labels = [norm_label(x) for x in template.cell_ids]
    ref_mask = template.supervised_mask.cpu().numpy().astype(bool)
    candidate_indices = unique_supervised_indices(ref_labels, ref_mask)
    if not candidate_indices:
        raise RuntimeError(f"fold{fold}: medoid has no unique supervised candidates")
    candidate_labels = [ref_labels[i] for i in candidate_indices]

    seen = set()
    rows = 0
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()

        ref_xyz = torch.from_numpy(
            mod.normalize_xyz(template.xyz.cpu().numpy())
        ).to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            for sample in test_samples:
                group_uid = str(sample.uid)
                q_labels = [norm_label(x) for x in sample.cell_ids]
                q_mask = sample.supervised_mask.cpu().numpy().astype(bool)
                valid_q = unique_supervised_indices(q_labels, q_mask)

                selected = []
                for qi in valid_q:
                    qid = f"{group_uid}::{int(qi)}"
                    if qid not in canonical:
                        continue
                    expected_group, expected_gt = canonical[qid]
                    observed_gt = q_labels[qi]
                    if (group_uid, observed_gt) != (expected_group, expected_gt):
                        raise RuntimeError(
                            f"fold{fold}: canonical metadata mismatch {qid}: "
                            f"observed={(group_uid, observed_gt)} expected={(expected_group, expected_gt)}"
                        )
                    selected.append((qi, qid, observed_gt))
                    seen.add(qid)

                if not selected:
                    continue

                q_xyz = torch.from_numpy(
                    mod.normalize_xyz(sample.xyz.cpu().numpy())
                ).to(device=device, dtype=torch.float32)

                # score_pair(a,b) returns b->a first and a->b second.
                _, q_to_ref = fdnc_train.score_pair(model, q_xyz, ref_xyz)
                scores = q_to_ref[:, :-1].detach().float().cpu().numpy()

                if scores.shape != (sample.num_nodes, template.num_nodes):
                    raise RuntimeError(
                        f"fold{fold} {group_uid}: score shape {scores.shape} "
                        f"!= {(sample.num_nodes, template.num_nodes)}"
                    )

                for qi, qid, gt in selected:
                    for ci, cand in zip(candidate_indices, candidate_labels):
                        value = float(scores[qi, ci])
                        w.writerow({
                            "query_uid": qid,
                            "group_uid": group_uid,
                            "gt_label": gt,
                            "candidate_label": cand,
                            "score": value,
                            "assignment_score": value,
                            "reference_uid": str(template.uid),
                        })
                        rows += 1

    expected = set(canonical)
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"fold{fold}: fDNC current canonical query mismatch "
            f"expected={len(expected)} seen={len(seen)} "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    if rows == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"fold{fold}: no fDNC rows written")
    tmp.replace(out)

    audit_payload = {
        "protocol": "rld_natural_accuracy_scaling_current_v1",
        "method": "fDNC current-grouped fine-tuned",
        "fold": fold,
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "template_uid": str(template.uid),
        "template_selection": "outer-train-only symmetric geometry medoid",
        "template_mean_distance": float(medoid.mean_distance),
        "canonical_query_source": str(canonical_path.resolve()),
        "canonical_query_source_sha256": sha256(canonical_path),
        "queries": len(seen),
        "candidate_identities": len(candidate_labels),
        "rows": rows,
        "missing_gt_policy_downstream": "incorrect",
    }
    audit.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")

    print(
        f"[DONE fDNC current] fold{fold} Q={len(seen)} "
        f"candidates={len(candidate_labels)} rows={rows} template={template.uid}",
        flush=True,
    )
    return out


def main():
    a = parse_args()
    folds = [int(x) for x in a.folds.split(",") if x.strip()]
    if a.seed != 42:
        raise ValueError("This locked scaling preparation is seed42-only.")
    device = torch.device(a.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    mod = import_eval_module()

    manifest = []
    for fold in folds:
        print("\n" + "=" * 112)
        print(f"PREPARE ACCURACY SCALING fold{fold} seed{a.seed}")
        print("=" * 112)

        mprt = ensure_mprt_export(fold, a.seed, a.device, a.overwrite)
        fdnc = export_fdnc_current(
            fold, a.seed, device, mprt, mod, a.overwrite
        )
        manifest.append({
            "fold": fold,
            "seed": a.seed,
            "mprt": str(mprt.resolve()),
            "fdnc_current": str(fdnc.resolve()),
        })

    man = ROOT / "runs/accuracy_scaling_current_v1/input_manifest.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\nDONE:", man)


if __name__ == "__main__":
    main()
