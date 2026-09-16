#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
KLB = Path("/home/ubuntu/klb")
OUT = ROOT / "runs/rld_robustness_cv5_seed42_v1"
METHODS = ("cpd", "fdnc", "nuclr", "geotransformer", "ngm_v2")

ALIASES = {
    "cpd": ("cpd",),
    "fdnc": ("fdnc",),
    "nuclr": ("nuclr",),
    "geotransformer": ("geotransformer", "geo_transformer"),
    "ngm_v2": ("ngm_v2", "ngmv2", "ngm-v2", "ngm"),
}

def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def paths_containing(base: Path, aliases: tuple[str, ...], suffixes: tuple[str, ...], max_items=100):
    if not base.exists():
        return []
    out = []
    # Deliberately use os.walk so we can prune huge irrelevant trees.
    skip = {".git", "__pycache__", "node_modules", "Data", "data", ".cache"}
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip]
        r = Path(root)
        for name in files:
            lower = name.lower()
            if suffixes and not lower.endswith(suffixes):
                continue
            full_text = str(r / name).lower()
            if any(a.lower() in full_text for a in aliases):
                out.append(str((r / name).resolve()))
                if len(out) >= max_items:
                    return out
    return out

def nearby_files(score: Path):
    result = {}
    d = score.parent
    patterns = {
        "audits": ("*.audit.json", "*audit*.json"),
        "json": ("*.json",),
        "logs": ("*.log",),
        "checkpoints": ("*.pt", "*.pth", "*.ckpt"),
    }
    for key, pats in patterns.items():
        found = []
        for parent in [d, d.parent, d.parent.parent]:
            if not parent.exists():
                continue
            for pat in pats:
                found += [str(x.resolve()) for x in parent.glob(pat)]
        result[key] = sorted(set(found))
    return result

def extract_paths_from_json(obj: Any):
    found = []
    def visit(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                visit(v, str(k))
        elif isinstance(x, list):
            for v in x:
                visit(v, key)
        elif isinstance(x, str):
            lowkey = key.lower()
            if any(w in lowkey for w in ("checkpoint", "script", "source", "repo", "config", "model", "runner")):
                if "/" in x or x.endswith((".py", ".sh", ".pt", ".pth", ".ckpt")):
                    found.append({"key": key, "value": x})
    visit(obj)
    return found

def grep_commands(log: Path):
    out = []
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return out
    for line in lines:
        s = line.strip()
        if (
            "python " in s
            or "python3 " in s
            or "python -u " in s
            or s.startswith("[RUN]")
            or "CUDA_VISIBLE_DEVICES" in s
        ):
            if len(s) < 3000:
                out.append(s)
    return out[-20:]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "RLD robustness replay audit; no model is trained or evaluated",
        "dataset_root": str((ROOT / "Data/Dunn_001623/cv5_grouped_v1").resolve()),
        "folds": folds,
        "model_seed": args.seed,
        "methods": {},
    }

    for method in METHODS:
        aliases = ALIASES[method]
        m = {"clean_cells": [], "source_candidates": [], "checkpoint_candidates": []}
        folder = method
        # Keep aliases flexible because NGM-v2 may have a slightly different output folder.
        possible_folders = [method]
        if method == "ngm_v2":
            possible_folders += ["ngmv2", "ngm-v2", "ngm"]
        for fold in folds:
            score = None
            for f in possible_folders:
                candidate = ROOT / f"runs/unified_benchmark/{f}/rld/fold{fold}/seed{args.seed}/test_candidate_scores.csv"
                if candidate.is_file():
                    score = candidate
                    break
            cell = {"fold": fold, "score": str(score.resolve()) if score else None}
            if score:
                near = nearby_files(score)
                cell.update(near)
                embedded = []
                commands = []
                for js in near["audits"] + near["json"]:
                    obj = read_json(Path(js))
                    if obj is not None:
                        embedded.extend(extract_paths_from_json(obj))
                for log in near["logs"]:
                    commands.extend(grep_commands(Path(log)))
                cell["embedded_provenance_paths"] = embedded
                cell["command_lines"] = commands
            m["clean_cells"].append(cell)

        search_roots = [ROOT]
        if method == "geotransformer":
            search_roots.append(KLB / "nuclr/geotransformer_official")
        elif method == "ngm_v2":
            search_roots.append(KLB)

        sources, ckpts = [], []
        for base in search_roots:
            sources += paths_containing(base, aliases, (".py", ".sh"), max_items=150)
            ckpts += paths_containing(base, aliases, (".pt", ".pth", ".ckpt"), max_items=150)

        # rank likely runner/export/eval files first
        def score_source(x):
            s = x.lower()
            return (
                20 * ("unified" in s)
                + 15 * ("export" in s)
                + 12 * ("evaluate" in s or "eval" in s)
                + 10 * ("run_" in s)
                + 8 * ("benchmark" in s)
                + 5 * ("train" in s)
                - len(s) / 10000
            )
        m["source_candidates"] = sorted(set(sources), key=score_source, reverse=True)[:40]
        m["checkpoint_candidates"] = sorted(set(ckpts))[:80]
        report["methods"][method] = m

    out = OUT / "BASELINE_REPLAY_AUDIT.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 120)
    print("RLD ROBUSTNESS BASELINE REPLAY AUDIT — seed", args.seed)
    print("=" * 120)
    for method, m in report["methods"].items():
        existing = [c for c in m["clean_cells"] if c["score"]]
        print(f"\n[{method}] clean unified cells: {len(existing)}/{len(folds)}")
        for c in existing:
            print(f"  fold{c['fold']}: {c['score']}")
            for item in c.get("embedded_provenance_paths", [])[:6]:
                print(f"    provenance {item['key']}: {item['value']}")
            for cmd in c.get("command_lines", [])[:3]:
                print(f"    command: {cmd}")
        print("  likely source/runner files:")
        for x in m["source_candidates"][:12]:
            print("   ", x)
        print("  likely checkpoints:")
        for x in m["checkpoint_candidates"][:8]:
            print("   ", x)

    print("\nSaved:", out)
    print("\nIMPORTANT: this command only audits provenance; it does not touch test data or rerun any model.")

if __name__ == "__main__":
    main()
