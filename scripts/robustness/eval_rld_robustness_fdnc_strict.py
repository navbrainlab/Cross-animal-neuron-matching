#!/usr/bin/env python3
"""
Strict RLD controlled-corruption evaluation for official released fDNC.

This script follows the CURRENT unified main-benchmark protocol exactly:

  * official released fDNC checkpoint, deterministic seed42 only;
  * clean outer-fold TRAIN worms are all reference templates;
  * every corrupted held-out test worm is scored against every train reference;
  * train-neuron scores are mapped to the locked training identity universe;
  * repeated scores for the same candidate identity across train references are
    reduced by MEAN, exactly as recompute_unified_benchmark.py;
  * query/rank/Hungarian semantics are delegated to the same unified evaluator.

Corruption is TEST-ONLY. No retraining/fine-tuning/hyperparameter selection.

Missing-neuron handling:
  canonical clean query IDs are worm_uid::ORIGINAL_ROW_INDEX.
  After deletion, current row numbers change, so surviving rows are mapped back
  to clean original rows by roi_index (preferred) or aligned_table_id.

Outlier handling:
  the corruption materializer appends synthetic distractors after the real
  prefix. They enter the fDNC input but are never GT queries.

Before any nonzero corruption for a fold, severity=0 MUST reproduce the
existing unified clean fDNC metrics exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
FDNC_SOURCE = ROOT / "baselines/official/third_party/fdnc_official"
FDNC_EXPECTED_COMMIT = "19c678781cd11a17866af7b6348ac0096a168c06"
FDNC_MODEL_SHA256 = "ab529eb6a886cb6ab3f199b7aaa4e49b82562dc280b3ac736ed61e25d5138ec9"

CV_ROOT = ROOT / "Data/Dunn_001623/cv5_grouped_v1"
CORR_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v1/corruptions"
CLEAN_SCORE_ROOT = ROOT / "runs/unified_benchmark/fdnc/rld"
OURS_SCORE_ROOT = ROOT / "runs/unified_benchmark/ours_static/rld"
OUT_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v1/results/fdnc_strict_v1"

COND_RE = re.compile(r"^(coord_noise|missing|outlier)_l([0-9]+(?:\.[0-9]+)?)_p([0-9]+)$")
FIELDS = (
    "query_uid",
    "group_uid",
    "gt_label",
    "candidate_label",
    "score",
    "assignment_score",
    "reference_uid",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def jdump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def norm_label(x: Any) -> str:
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    s = str(x).strip()
    return "" if s.lower() in {"", "nan", "none", "null", "-1"} else s


def recording_uid(path: Path) -> str:
    with np.load(path, allow_pickle=True) as z:
        for key in ("recording_uid", "worm_id"):
            if key in z.files:
                a = np.asarray(z[key]).reshape(-1)
                if len(a):
                    v = a[0]
                    if isinstance(v, np.generic):
                        v = v.item()
                    if isinstance(v, bytes):
                        v = v.decode("utf-8", errors="replace")
                    if str(v):
                        return str(v)
    parts = path.stem.split("__")
    return parts[1] if len(parts) >= 3 else path.stem


def _git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        # Worktrees/submodules can have .git as a file; git itself is authoritative.
        pass
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _find_released_model(source: Path) -> Path | None:
    candidates = []
    model_dir = source / "model"
    if model_dir.exists():
        candidates.extend(p for p in model_dir.rglob("*") if p.is_file())
    # Some preserved copies put the released model elsewhere.
    try:
        candidates.extend(
            p for p in source.rglob("*")
            if p.is_file() and p.stat().st_size < 1_000_000_000
        )
    except OSError:
        pass

    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp in seen:
            continue
        seen.add(rp)
        try:
            if sha256(rp) == FDNC_MODEL_SHA256:
                return rp
        except OSError:
            continue
    return None


def discover_fdnc_assets() -> tuple[Path, Path]:
    """
    Locate the exact SOURCE and exact RELEASED MODEL independently.

    This matches export_unified_candidate_scores.validate_fdnc_assets(source, model):
      - source is audited by git commit + locked source-file hashes;
      - model is audited independently by model.bin SHA256.

    They do NOT need to live in the same checkout directory.
    """
    # ------------------------------------------------------------------
    # 1) Exact source checkout.
    # ------------------------------------------------------------------
    source_candidates = [
        ROOT / "third_party/fDNC_Neuron_ID_19c6787",
        ROOT / "baselines/official/third_party/fDNC_Neuron_ID_19c6787",
        ROOT / "baselines/official/third_party/fdnc_official",
        Path("/home/ubuntu/klb/fDNC_Neuron_ID"),
        ROOT / "third_party/fDNC_Neuron_ID",
    ]

    # Discover additional source copies, but keep preferred paths first.
    for base in (ROOT, Path("/home/ubuntu/klb")):
        if not base.exists():
            continue
        try:
            for p in base.rglob("model.py"):
                if p.parent.name == "src":
                    candidate = p.parent.parent
                    if "fdnc" in str(candidate).lower() or "neuron_id" in str(candidate).lower():
                        source_candidates.append(candidate)
        except (OSError, PermissionError):
            pass

    exact_source = None
    source_audit = []
    seen = set()
    for candidate in source_candidates:
        try:
            source = candidate.resolve()
        except OSError:
            continue
        if source in seen or not (source / "src/model.py").is_file():
            continue
        seen.add(source)
        commit = _git_commit(source)
        source_audit.append((source, commit))
        if commit == FDNC_EXPECTED_COMMIT:
            exact_source = source
            break

    if exact_source is None:
        print("[fDNC ASSET] exact source checkout not found. Audited:", flush=True)
        for source, commit in source_audit:
            print(f"  source={source} commit={commit}", flush=True)
        raise RuntimeError(
            "Could not find exact fDNC SOURCE checkout. "
            f"Required commit={FDNC_EXPECTED_COMMIT}"
        )

    # ------------------------------------------------------------------
    # 2) Exact released checkpoint. It may live in a different checkout.
    # ------------------------------------------------------------------
    model_candidates = [
        ROOT / "baselines/official/third_party/fdnc_official/model/model.bin",
        Path("/home/ubuntu/klb/fDNC_Neuron_ID/model/model.bin"),
        exact_source / "model/model.bin",
    ]

    # Add all model.bin files under the two project roots.
    for base in (ROOT, Path("/home/ubuntu/klb")):
        if not base.exists():
            continue
        try:
            model_candidates.extend(base.rglob("model.bin"))
        except (OSError, PermissionError):
            pass

    exact_model = None
    model_audit = []
    seen_models = set()
    for candidate in model_candidates:
        try:
            model = candidate.resolve()
        except OSError:
            continue
        if model in seen_models or not model.is_file():
            continue
        seen_models.add(model)
        try:
            observed = sha256(model)
        except OSError:
            continue
        model_audit.append((model, observed))
        if observed == FDNC_MODEL_SHA256:
            exact_model = model
            break

    if exact_model is None:
        print("[fDNC ASSET] exact released model not found. Audited:", flush=True)
        for model, observed in model_audit[:20]:
            print(f"  model={model} sha256={observed}", flush=True)
        raise RuntimeError(
            "Could not find exact released fDNC MODEL. "
            f"Required sha256={FDNC_MODEL_SHA256}"
        )

    print(f"[fDNC ASSET] source     = {exact_source}", flush=True)
    print(f"[fDNC ASSET] commit     = {_git_commit(exact_source)}", flush=True)
    print(f"[fDNC ASSET] checkpoint = {exact_model}", flush=True)
    print(f"[fDNC ASSET] model sha  = {sha256(exact_model)}", flush=True)
    print(
        "[fDNC ASSET] source/model may be in different directories; "
        "this is valid because the locked benchmark validates them independently.",
        flush=True,
    )
    return exact_source, exact_model


def _load_module_from_path(module_name: str, path: Path):
    import importlib.util

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    for parent in (ROOT, path.parent):
        text = str(parent)
        if text not in sys.path:
            sys.path.insert(0, text)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_protocol_script(name: str, required_tokens: tuple[str, ...]) -> Path:
    """
    Locate the preserved canonical unified-benchmark script after repository
    cleanup/archiving.

    Archived copies are allowed here because the already-produced unified
    benchmark scores identify that preserved implementation as the source of
    truth; the script path and SHA256 are printed for audit.
    """
    preferred = [
        ROOT / name,
        ROOT / "experiments/mprt_population_relational_transport"
               / "medoid_template_cv5x3_20260825" / "protocol" / name,
        ROOT / "baselines" / "official" / name,
        ROOT / "archive/non_primary_models_20260825/root_scripts" / name,
    ]

    def valid(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return all(token in text for token in required_tokens)

    for path in preferred:
        if valid(path):
            chosen = path.resolve()
            print(f"[PROTOCOL] {name} -> {chosen}", flush=True)
            return chosen

    candidates = []
    for path in ROOT.rglob(name):
        if valid(path):
            candidates.append(path.resolve())

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate a valid {name} under {ROOT}. "
            f"Required tokens={required_tokens}"
        )

    candidates = sorted(set(candidates), key=lambda p: (len(p.parts), str(p)))
    chosen = candidates[0]
    print(f"[PROTOCOL] {name} -> {chosen}", flush=True)
    if len(candidates) > 1:
        print("[PROTOCOL] other valid copies:", flush=True)
        for p in candidates[1:6]:
            print(f"  - {p}", flush=True)
    return chosen


def load_modules():
    exporter_path = _resolve_protocol_script(
        "export_unified_candidate_scores.py",
        (
            "FDNC_MODEL_SHA256",
            "def export_fdnc",
            "def fdnc_scores",
            "load_canonical_query_map",
        ),
    )
    evaluator_path = _resolve_protocol_script(
        "recompute_unified_benchmark.py",
        (
            "def aggregate_scores",
            "def compute_cell",
            "reference_reducer",
            "hungarian_accuracy",
        ),
    )

    exporter = _load_module_from_path(
        "_mprt_locked_export_unified_candidate_scores", exporter_path
    )
    evaluator = _load_module_from_path(
        "_mprt_locked_recompute_unified_benchmark", evaluator_path
    )

    print(f"[PROTOCOL] exporter sha256  = {sha256(exporter_path)}", flush=True)
    print(f"[PROTOCOL] evaluator sha256 = {sha256(evaluator_path)}", flush=True)
    return exporter, evaluator


def _resolve_ours_clean_score_file(fold: int) -> Path:
    rel = Path("ours_static") / "rld" / f"fold{fold}" / "seed42" / "test_candidate_scores.csv"
    preferred = [
        ROOT / "runs/unified_benchmark" / rel,
        ROOT / "experiments/mprt_population_relational_transport"
               / "medoid_template_cv5x3_20260825" / "runs/unified_benchmark" / rel,
        ROOT / "archive/non_primary_models_20260825" / "runs/unified_benchmark" / rel,
    ]
    for p in preferred:
        if p.is_file():
            return p.resolve()

    candidates = []
    try:
        for p in ROOT.rglob("test_candidate_scores.csv"):
            parts = [x.lower() for x in p.parts]
            if (
                "ours_static" in parts
                and "rld" in parts
                and f"fold{fold}" in parts
                and "seed42" in parts
            ):
                candidates.append(p.resolve())
    except (OSError, PermissionError):
        pass

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate canonical Ours-Static score file for RLD fold{fold} seed42"
        )
    candidates = sorted(set(candidates), key=lambda p: (len(p.parts), str(p)))
    chosen = candidates[0]
    print(f"[CANONICAL] fold{fold} -> {chosen}", flush=True)
    return chosen


def load_clean_canonical(fold: int) -> dict[str, tuple[str, str]]:
    path = _resolve_ours_clean_score_file(fold)
    out: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = str(row["query_uid"])
            meta = (str(row["group_uid"]), norm_label(row["gt_label"]))
            old = out.setdefault(qid, meta)
            if old != meta:
                raise RuntimeError(f"Canonical query metadata conflict: {qid}")
    if not out:
        raise RuntimeError(f"Empty canonical query source: {path}")
    return out


def clean_test_paths(fold: int) -> list[Path]:
    paths = sorted((CV_ROOT / f"fold_{fold}" / "test").glob("*.npz"))
    if not paths:
        raise FileNotFoundError(CV_ROOT / f"fold_{fold}" / "test")
    return paths


def clean_train_paths(fold: int) -> list[Path]:
    paths = sorted((CV_ROOT / f"fold_{fold}" / "train").glob("*.npz"))
    if not paths:
        raise FileNotFoundError(CV_ROOT / f"fold_{fold}" / "train")
    return paths


def by_uid(paths: list[Path]) -> dict[str, Path]:
    out = {}
    for p in paths:
        uid = recording_uid(p)
        if uid in out:
            raise RuntimeError(f"Duplicate recording UID {uid}")
        out[uid] = p
    return out


def _as1d(a) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def _unique_nonnegative_index_map(values: np.ndarray) -> dict[int, int] | None:
    out: dict[int, int] = {}
    for i, v in enumerate(_as1d(values).tolist()):
        try:
            key = int(v)
        except Exception:
            return None
        if key < 0:
            continue
        if key in out:
            return None
        out[key] = i
    return out


def current_to_clean_rows(clean_path: Path, corrupt_path: Path) -> np.ndarray:
    """
    Return length=current_n array whose entries are original clean row indices,
    or -1 for synthetic/unresolvable rows.
    """
    with np.load(clean_path, allow_pickle=True) as z0:
        clean = {k: np.asarray(z0[k]) for k in z0.files}
    with np.load(corrupt_path, allow_pickle=True) as z1:
        cur = {k: np.asarray(z1[k]) for k in z1.files}

    if "xyz" not in clean or "xyz" not in cur:
        raise KeyError("Both clean and corruption NPZ require xyz")

    clean_n = int(np.asarray(clean["xyz"]).shape[0])
    cur_n = int(np.asarray(cur["xyz"]).shape[0])

    # Clean/coord_noise/outlier: original neurons are preserved as prefix.
    if cur_n >= clean_n:
        mapping = np.full(cur_n, -1, dtype=np.int64)
        mapping[:clean_n] = np.arange(clean_n, dtype=np.int64)
        if "roi_index" in clean and "roi_index" in cur:
            a = _as1d(clean["roi_index"])
            b = _as1d(cur["roi_index"])
            if len(a) == clean_n and len(b) >= clean_n and not np.array_equal(a, b[:clean_n]):
                raise RuntimeError(f"{corrupt_path}: original roi_index prefix changed")
        return mapping

    # Missing: use stable row metadata.
    for key in ("roi_index", "aligned_table_id"):
        if key not in clean or key not in cur:
            continue
        a = _as1d(clean[key])
        b = _as1d(cur[key])
        if len(a) != clean_n or len(b) != cur_n:
            continue
        clean_map = _unique_nonnegative_index_map(a)
        if clean_map is None:
            continue
        mapping = np.full(cur_n, -1, dtype=np.int64)
        matched = 0
        for i, v in enumerate(b.tolist()):
            try:
                token = int(v)
            except Exception:
                continue
            if token < 0:
                continue
            j = clean_map.get(token)
            if j is not None:
                mapping[i] = int(j)
                matched += 1
        if matched == cur_n:
            return mapping

    # Last-resort stable unique cell_id mapping.
    if "cell_id" in clean and "cell_id" in cur:
        clean_ids = [norm_label(x) for x in _as1d(clean["cell_id"]).tolist()]
        cur_ids = [norm_label(x) for x in _as1d(cur["cell_id"]).tolist()]
        positions: dict[str, int] = {}
        dup = set()
        for i, token in enumerate(clean_ids):
            if not token:
                continue
            if token in positions:
                dup.add(token)
            else:
                positions[token] = i
        for token in dup:
            positions.pop(token, None)
        mapping = np.full(cur_n, -1, dtype=np.int64)
        matched = 0
        for i, token in enumerate(cur_ids):
            if not token or token.startswith("__OUTLIER"):
                continue
            j = positions.get(token)
            if j is not None:
                mapping[i] = int(j)
                matched += 1
        if matched > 0:
            return mapping

    raise RuntimeError(
        f"Could not map corruption rows back to clean rows: clean={clean_path}, corrupt={corrupt_path}"
    )


def discover_conditions(fold: int, kinds: set[str]) -> list[dict]:
    root = CORR_ROOT / f"fold{fold}"
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        m = COND_RE.match(p.name)
        if not m:
            continue
        kind, sev, seed = m.groups()
        if kind not in kinds:
            continue
        if not (p / "test").is_dir():
            raise FileNotFoundError(p / "test")
        rows.append({
            "fold": fold,
            "kind": kind,
            "severity": float(sev),
            "perturbation_seed": int(seed),
            "name": p.name,
            "root": p,
        })
    order = {"coord_noise": 0, "missing": 1, "outlier": 2}
    rows.sort(key=lambda r: (order[r["kind"]], r["severity"], r["perturbation_seed"]))
    return rows


def clean_condition(fold: int) -> dict:
    p = CORR_ROOT / f"fold{fold}" / "coord_noise_l0.00_p0"
    if not (p / "test").is_dir():
        raise FileNotFoundError(p / "test")
    return {
        "fold": fold,
        "kind": "coord_noise",
        "severity": 0.0,
        "perturbation_seed": 0,
        "name": p.name,
        "root": p,
    }


def load_identity_data(exporter, path: Path):
    xyz, labels, supervised = exporter.load_npz_identity_data(path)
    return (
        np.asarray(xyz),
        np.asarray(labels),
        np.asarray(supervised, dtype=bool),
    )


def write_condition_scores(
    exporter,
    model,
    fold: int,
    row: dict,
    device: torch.device,
    force: bool,
) -> Path:
    out_dir = OUT_ROOT / f"fold{fold}" / row["name"]
    score_path = out_dir / "test_candidate_scores.csv"
    audit_path = out_dir / "score_audit.json"
    if score_path.is_file() and not force:
        return score_path

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = score_path.with_suffix(".csv.tmp")
    tmp.unlink(missing_ok=True)

    canonical = load_clean_canonical(fold)
    universe = set(exporter.atlas_identity_mapping(ROOT, "rld", fold, 42))

    # Exact clean outer-train reference set used by unified main benchmark.
    train = []
    for p in clean_train_paths(fold):
        xyz, labels, supervised = load_identity_data(exporter, p)
        indices = exporter.unique_indices(labels, supervised)
        train.append((
            recording_uid(p),
            exporter.normalize_fdnc_xyz(xyz),
            labels,
            indices,
        ))

    clean_test = by_uid(clean_test_paths(fold))
    corrupt_paths = sorted((row["root"] / "test").glob("*.npz"))
    corrupt_by_uid = by_uid(corrupt_paths)
    if set(corrupt_by_uid) != set(clean_test):
        raise RuntimeError(
            f"fold{fold} {row['name']}: test worm UID set changed. "
            f"clean={len(clean_test)} corrupt={len(corrupt_by_uid)} "
            f"missing={sorted(set(clean_test)-set(corrupt_by_uid))[:5]} "
            f"extra={sorted(set(corrupt_by_uid)-set(clean_test))[:5]}"
        )

    seen_queries: set[str] = set()
    real_rows = synthetic_rows = 0
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for uid in sorted(corrupt_by_uid):
            clean_p = clean_test[uid]
            corrupt_p = corrupt_by_uid[uid]
            xyz, labels, supervised = load_identity_data(exporter, corrupt_p)
            query_xyz = exporter.normalize_fdnc_xyz(xyz)
            row_map = current_to_clean_rows(clean_p, corrupt_p)

            # Surviving canonical clean queries, identified by ORIGINAL clean row.
            query_rows: list[tuple[int, str, str]] = []
            for current_i, original_i in enumerate(row_map.tolist()):
                if original_i < 0:
                    synthetic_rows += 1
                    continue
                qid = f"{uid}::{int(original_i)}"
                meta = canonical.get(qid)
                if meta is None:
                    continue
                expected_uid, expected_gt = meta
                if expected_uid != uid:
                    raise RuntimeError(f"Canonical UID mismatch for {qid}")
                # Never derive GT from corrupted labels; use locked clean canonical metadata.
                query_rows.append((int(current_i), qid, expected_gt))
                seen_queries.add(qid)
                real_rows += 1

            for reference_uid, reference_xyz, reference_labels, reference_indices in train:
                scores = exporter.fdnc_scores(model, reference_xyz, query_xyz)
                candidate_indices = [
                    int(j)
                    for j in reference_indices
                    if norm_label(reference_labels[int(j)]) in universe
                ]
                for current_i, qid, gt in query_rows:
                    for candidate_i in candidate_indices:
                        candidate_label = norm_label(reference_labels[candidate_i])
                        value = float(scores[current_i, candidate_i])
                        writer.writerow({
                            "query_uid": qid,
                            "group_uid": uid,
                            "gt_label": gt,
                            "candidate_label": candidate_label,
                            "score": value,
                            "assignment_score": value,
                            "reference_uid": reference_uid,
                        })

    if not seen_queries:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"fold{fold} {row['name']}: no surviving canonical queries")

    tmp.replace(score_path)
    jdump(audit_path, {
        "method": "fDNC official released",
        "fold": fold,
        "condition": row["name"],
        "kind": row["kind"],
        "severity": row["severity"],
        "perturbation_seed": row["perturbation_seed"],
        "model_seed": 42,
        "canonical_clean_queries": len(canonical),
        "surviving_canonical_queries": len(seen_queries),
        "missing_canonical_queries": len(canonical) - len(seen_queries),
        "train_references": len(train),
        "query_real_rows_written": real_rows,
        "synthetic_or_unmapped_rows": synthetic_rows,
        "reference_reducer": "mean",
        "test_only_corruption": True,
        "fdnc_checkpoint_sha256": FDNC_MODEL_SHA256,
        "score_file_sha256": sha256(score_path),
    })
    return score_path


def evaluate_score_file(evaluator, path: Path, fold: int) -> dict:
    frame = evaluator.load_long_scores(path)
    frame = evaluator.aggregate_scores(frame, "mean")
    key = evaluator.CellKey(method="fDNC", dataset="rld", fold=fold, seed=42)
    result, signature = evaluator.compute_cell(
        frame,
        key,
        stochastic=False,
        protocol_id="unified_rld_corruption_fdnc_v1",
        split="test",
        source_file=path,
        missing_gt_policy="incorrect",
    )
    return {
        "result": result,
        "row": result.row(),
        "signature": signature,
        "aggregated_frame": frame,
    }


def resolve_clean_fdnc_score_file(evaluator, fold: int) -> Path:
    """
    Locate the exact saved unified fDNC clean candidate-score file after repo
    cleanup/archiving.

    We only accept files whose path semantics match:
        fdnc / rld / fold{fold} / seed42 / test_candidate_scores.csv
    and whose query signature is EXACTLY the canonical Ours-Static clean
    signature for this biological fold.

    If several byte-identical / metric-identical archived copies exist, choose
    deterministically and print all accepted copies for audit.
    """
    rel = Path("fdnc") / "rld" / f"fold{fold}" / "seed42" / "test_candidate_scores.csv"

    preferred = [
        ROOT / "runs/unified_benchmark" / rel,
        ROOT / "experiments/mprt_population_relational_transport"
               / "medoid_template_cv5x3_20260825" / "runs/unified_benchmark" / rel,
        ROOT / "archive/non_primary_models_20260825" / "runs/unified_benchmark" / rel,
    ]

    candidates = []
    seen = set()

    for p in preferred:
        if p.is_file():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                candidates.append(rp)

    # General fallback: exact basename + required path components.
    try:
        for p in ROOT.rglob("test_candidate_scores.csv"):
            parts = [x.lower() for x in p.parts]
            if "fdnc" not in parts or "rld" not in parts:
                continue
            if f"fold{fold}" not in parts or "seed42" not in parts:
                continue
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                candidates.append(rp)
    except (OSError, PermissionError):
        pass

    if not candidates:
        raise FileNotFoundError(
            "Could not locate saved unified fDNC clean score file for "
            f"fold{fold}. Expected suffix={rel}. "
            "The model/source assets are already valid; only the archived "
            "clean candidate-score reference is missing."
        )

    # Canonical clean query signature from Ours Static.
    canonical = load_clean_canonical(fold)
    expected_signature = tuple(
        sorted((qid, meta[0], meta[1]) for qid, meta in canonical.items())
    )

    accepted = []
    rejected = []
    for path in sorted(candidates, key=str):
        try:
            ev = evaluate_score_file(evaluator, path, fold)
            sig = tuple(sorted(ev["signature"]))
            if sig != expected_signature:
                rejected.append((path, f"query_signature {len(sig)} != canonical {len(expected_signature)}"))
                continue
            accepted.append((path, ev))
        except Exception as exc:
            rejected.append((path, f"{type(exc).__name__}: {exc}"))

    if not accepted:
        print("[CLEAN REF] candidates found but none passed canonical query audit:", flush=True)
        for p, why in rejected[:20]:
            print(f"  REJECT {p}: {why}", flush=True)
        raise RuntimeError(
            f"No saved fDNC clean score file for fold{fold} passed the "
            "canonical query-signature audit."
        )

    # If there are multiple accepted archived copies, require their computed
    # clean metrics to be exactly identical before selecting one.
    metric_keys = (
        "queries", "gt_covered", "top1_real", "top5_real",
        "mrr_real", "hungarian_queries", "hungarian_accuracy",
    )
    ref_row = accepted[0][1]["row"]
    for path, ev in accepted[1:]:
        row = ev["row"]
        for key in metric_keys:
            a = ref_row[key]
            b = row[key]
            if key in {"queries", "gt_covered", "hungarian_queries"}:
                same = int(a) == int(b)
            else:
                same = abs(float(a) - float(b)) <= 1e-12
            if not same:
                raise RuntimeError(
                    "Multiple archived fDNC clean score files pass the query "
                    f"audit but disagree on {key}: "
                    f"{accepted[0][0]} -> {a}, {path} -> {b}. "
                    "Refuse to choose silently."
                )

    # Prefer the shallowest/most canonical-looking path.
    accepted.sort(
        key=lambda x: (
            0 if "runs/unified_benchmark" in str(x[0]) else 1,
            len(x[0].parts),
            str(x[0]),
        )
    )
    chosen = accepted[0][0]
    print(f"[CLEAN REF] fold{fold} -> {chosen}", flush=True)
    print(f"[CLEAN REF] sha256 = {sha256(chosen)}", flush=True)
    if len(accepted) > 1:
        print("[CLEAN REF] equivalent accepted copies:", flush=True)
        for p, _ in accepted[1:6]:
            print(f"  - {p}", flush=True)
    return chosen


def clean_reference_metrics(evaluator, fold: int) -> dict:
    path = resolve_clean_fdnc_score_file(evaluator, fold)
    out = evaluate_score_file(evaluator, path, fold)
    out["path"] = path
    return out


def clean_guard(fold: int, replay: dict, saved: dict) -> dict:
    a = replay["row"]
    b = saved["row"]
    keys = (
        "queries",
        "gt_covered",
        "top1_real",
        "top5_real",
        "mrr_real",
        "hungarian_queries",
        "hungarian_accuracy",
    )
    diffs = {}
    ok = replay["signature"] == saved["signature"]
    for k in keys:
        av, bv = a[k], b[k]
        if isinstance(av, (int, np.integer)) or k in {"queries", "gt_covered", "hungarian_queries"}:
            same = int(av) == int(bv)
            diff = abs(int(av) - int(bv))
        else:
            same = abs(float(av) - float(bv)) <= 1e-12
            diff = abs(float(av) - float(bv))
        diffs[k] = {"saved": bv, "replay": av, "abs_diff": diff}
        ok &= same

    print(
        f"[CLEAN GUARD {'EXACT' if ok else 'MISMATCH'}] fold{fold} "
        f"Q={a['queries']} Cov={100*a['candidate_coverage']:.2f}% "
        f"Top1={100*a['top1_real']:.4f}% "
        f"Top5={100*a['top5_real']:.4f}% "
        f"MRR={a['mrr_real']:.6f} "
        f"Hung={100*a['hungarian_accuracy']:.4f}%",
        flush=True,
    )
    return {
        "status": "exact" if ok else "mismatch",
        "query_signature_exact": replay["signature"] == saved["signature"],
        "saved_clean_score_file": str(saved["path"]),
        "diffs": diffs,
    }


def result_record(row: dict, evaluated: dict, clean_eval: dict, guard=None) -> dict:
    r = evaluated["result"]
    clean_r = clean_eval["result"]

    clean_q = int(clean_r.queries)
    q = int(r.queries)
    clean_hq = int(clean_r.hungarian_queries)
    hq = int(r.hungarian_queries)

    # Current score file contains only surviving canonical queries under missing corruption.
    survival_coverage = q / clean_q if clean_q else float("nan")
    effective_top1 = r.top1_correct / clean_q if clean_q else float("nan")
    effective_top5 = r.top5_correct / clean_q if clean_q else float("nan")
    effective_mrr = r.reciprocal_rank_sum / clean_q if clean_q else float("nan")
    effective_hungarian = r.hungarian_correct / clean_hq if clean_hq else float("nan")

    out = {
        "method": "fDNC",
        "fold": int(row["fold"]),
        "condition": row["name"],
        "kind": row["kind"],
        "severity": float(row["severity"]),
        "perturbation_seed": int(row["perturbation_seed"]),
        "model_seed": 42,
        "queries": q,
        "clean_queries": clean_q,
        "candidate_coverage": float(r.metrics()["candidate_coverage"]),
        "survival_coverage": float(survival_coverage),
        "top1": float(r.metrics()["top1_real"]),
        "top5": float(r.metrics()["top5_real"]),
        "mrr": float(r.metrics()["mrr_real"]),
        "hungarian": float(r.metrics()["hungarian_accuracy"]),
        "effective_top1": float(effective_top1),
        "effective_top5": float(effective_top5),
        "effective_mrr": float(effective_mrr),
        "effective_hungarian": float(effective_hungarian),
        "top1_correct": int(r.top1_correct),
        "hungarian_correct": int(r.hungarian_correct),
        "hungarian_queries": int(r.hungarian_queries),
        "protocol": {
            "official_released_fdnc": True,
            "all_outer_train_worms_as_references": True,
            "reference_reducer": "mean",
            "canonical_query_source": "Ours Static unified benchmark seed42",
            "metric_evaluator": "recompute_unified_benchmark.py",
            "test_only_corruption": True,
            "retraining": False,
        },
    }
    if guard is not None:
        out["clean_replay_guard"] = guard

    dst = OUT_ROOT / f"fold{row['fold']}" / row["name"] / "robustness_metrics.json"
    jdump(dst, out)

    print(
        f"[fDNC] fold{row['fold']} {row['name']}: "
        f"Q={q}/{clean_q} "
        f"Top1={100*out['top1']:.2f}% "
        f"Top5={100*out['top5']:.2f}% "
        f"Hung={100*out['hungarian']:.2f}% "
        f"SurvivalCov={100*out['survival_coverage']:.2f}% "
        f"EffTop1={100*out['effective_top1']:.2f}%",
        flush=True,
    )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize(results: list[dict]) -> None:
    fields = [
        "method", "fold", "kind", "severity", "perturbation_seed", "model_seed",
        "queries", "clean_queries", "candidate_coverage", "survival_coverage",
        "top1", "top5", "mrr", "hungarian",
        "effective_top1", "effective_top5", "effective_mrr", "effective_hungarian",
    ]
    cells = [{k: r[k] for k in fields} for r in results]
    write_csv(OUT_ROOT / "fdnc_all_cells.csv", cells)

    metrics = [
        "candidate_coverage", "survival_coverage",
        "top1", "top5", "mrr", "hungarian",
        "effective_top1", "effective_top5", "effective_mrr", "effective_hungarian",
    ]

    # perturbation seeds -> mean inside biological fold
    fold_rows = []
    df = pd.DataFrame(cells)
    for (kind, sev, fold), part in df.groupby(["kind", "severity", "fold"], sort=True):
        row = {
            "method": "fDNC",
            "kind": kind,
            "severity": float(sev),
            "fold": int(fold),
            "perturbation_seeds": int(len(part)),
        }
        for m in metrics:
            row[m] = float(part[m].mean())
        fold_rows.append(row)
    write_csv(OUT_ROOT / "fdnc_fold_level.csv", fold_rows)

    # biological folds -> mean ± sample SD
    macro = []
    fdf = pd.DataFrame(fold_rows)
    for (kind, sev), part in fdf.groupby(["kind", "severity"], sort=True):
        row = {
            "method": "fDNC",
            "kind": kind,
            "severity": float(sev),
            "folds": int(len(part)),
        }
        for m in metrics:
            vals = part[m].to_numpy(dtype=float)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        macro.append(row)
    write_csv(OUT_ROOT / "fdnc_macro_summary.csv", macro)

    print("\n" + "=" * 110)
    print("fDNC ROBUSTNESS MACRO SUMMARY")
    print("=" * 110)
    for r in macro:
        print(
            f"{r['kind']:12s} level={r['severity']:.2f} "
            f"Top1={100*r['top1_mean']:.2f}±{100*r['top1_sd']:.2f}% "
            f"Hung={100*r['hungarian_mean']:.2f}±{100*r['hungarian_sd']:.2f}% "
            f"SurvCov={100*r['survival_coverage_mean']:.2f}±{100*r['survival_coverage_sd']:.2f}% "
            f"EffTop1={100*r['effective_top1_mean']:.2f}±{100*r['effective_top1_sd']:.2f}%"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--kinds", default="coord_noise")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    kinds = {x.strip() for x in args.kinds.split(",") if x.strip()}
    valid = {"coord_noise", "missing", "outlier"}
    if not kinds or not kinds <= valid:
        raise ValueError(f"--kinds must be subset of {sorted(valid)}")

    exporter, evaluator = load_modules()
    fdnc_source, model_file = discover_fdnc_assets()
    exporter.validate_fdnc_assets(fdnc_source.resolve(), model_file.resolve())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = exporter.load_fdnc(fdnc_source.resolve(), model_file.resolve(), device)
    model.eval()

    print(f"[fDNC] source     = {fdnc_source}")
    print(f"[fDNC] checkpoint = {model_file}")
    print(f"[fDNC] device     = {device}")
    print(f"[fDNC] reducer    = mean")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for fold in folds:
        saved = clean_reference_metrics(evaluator, fold)
        clean_row = clean_condition(fold)

        clean_score = write_condition_scores(
            exporter, model, fold, clean_row, device, args.force
        )
        replay = evaluate_score_file(evaluator, clean_score, fold)
        guard = clean_guard(fold, replay, saved)
        clean_result = result_record(clean_row, replay, saved, guard=guard)
        if guard["status"] != "exact":
            raise RuntimeError(
                f"fold{fold}: severity=0 does not exactly reproduce unified clean fDNC. "
                "STOP before nonzero corruption."
            )
        results.append(clean_result)

        for row in discover_conditions(fold, kinds):
            if (
                row["kind"] == "coord_noise"
                and abs(row["severity"]) <= 1e-12
                and row["perturbation_seed"] == 0
            ):
                continue
            score_path = write_condition_scores(
                exporter, model, fold, row, device, args.force
            )
            evaluated = evaluate_score_file(evaluator, score_path, fold)
            results.append(result_record(row, evaluated, saved))

    summarize(results)
    print("\nCOMPLETE")
    print(f"results: {OUT_ROOT}")
    print(f"macro:   {OUT_ROOT / 'fdnc_macro_summary.csv'}")


if __name__ == "__main__":
    main()
