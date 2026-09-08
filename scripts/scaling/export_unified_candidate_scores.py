#!/usr/bin/env python3
"""Export fair canonical-annotation candidate scores for existing baselines.

Each test neuron is evaluated once against an identity universe learned from
the current fold's training set.  The output is consumed by
``recompute_unified_benchmark.py``.

Supported methods:
  * ours_static / ours_dynamic: existing MPRT Atlas checkpoints;
  * fdnc: locked official fDNC source and released checkpoint;
  * nuclr: existing fold-pure NuCLR checkpoints, reused only after an exact
    train/validation animal-set audit against the current grouped fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch


DATA_ROOTS = {
    "atanas": "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
    "rld": "Data/Dunn_001623/cv5_grouped_v1",
}
SEEDS = (1, 42, 123)
FDNC_COMMIT = "19c678781cd11a17866af7b6348ac0096a168c06"
FDNC_MODEL_SHA256 = "ab529eb6a886cb6ab3f199b7aaa4e49b82562dc280b3ac736ed61e25d5138ec9"
FDNC_SOURCE_HASHES = {
    "src/model.py": "8969dae44995e4430f045acf84ab5a42154d5fe3124d445122f96ffeae8e9f5c",
    "src/DNC_predict.py": "9607980589d46de04ad4e32ed44d1f735fd3041134f3b725c83ba70d01cb1683",
    "src/fDNC_eval.py": "d2fafacd5f35df3d350be25f29a922b97c3ce042a08e9e78d79f471033a82f21",
}
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "-1"} else text


def split_paths(fold_root: Path, split: str) -> list[Path]:
    paths = sorted((fold_root / split).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ files in {fold_root / split}")
    return paths


def recording_uid(path: Path) -> str:
    with np.load(path, allow_pickle=True) as data:
        if "recording_uid" in data.files:
            return str(np.asarray(data["recording_uid"]).reshape(-1)[0])
    return path.stem


def path_stems(root: Path) -> set[str]:
    return {path.stem for path in root.rglob("*.npz")}


def unique_indices(labels: Sequence[str], supervised: np.ndarray) -> np.ndarray:
    normalized = [normalize_label(value) for value in labels]
    counts = Counter(value for value, keep in zip(normalized, supervised) if keep and value)
    return np.asarray(
        [
            index
            for index, (value, keep) in enumerate(zip(normalized, supervised))
            if keep and value and counts[value] == 1
        ],
        dtype=np.int64,
    )


def load_npz_identity_data(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float32)[:, :3]
        labels = [normalize_label(value) for value in np.asarray(data["cell_id"]).reshape(-1)]
        if "labeled_mask" in data.files:
            supervised = np.asarray(data["labeled_mask"], dtype=bool).reshape(-1)
        elif "clean_mask" in data.files:
            supervised = np.asarray(data["clean_mask"], dtype=bool).reshape(-1)
        else:
            supervised = np.asarray([bool(value) for value in labels], dtype=bool)
    if len(xyz) != len(labels) or len(labels) != len(supervised):
        raise ValueError(f"Length mismatch in {path}")
    return xyz, labels, supervised


def atlas_checkpoint(repo: Path, dataset: str, fold: int, seed: int, method: str) -> Path:
    cell = (
        repo
        / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1"
        / dataset
        / f"fold{fold}"
        / f"seed{seed}"
    )
    if method == "ours_static":
        return cell / "static_atlas/anchored_pure.pt"
    if method == "ours_dynamic":
        return cell / "dynamic/low_rank_r8/best.pt"
    raise ValueError(method)


def atlas_identity_mapping(repo: Path, dataset: str, fold: int, seed: int) -> dict[str, int]:
    checkpoint = atlas_checkpoint(repo, dataset, fold, seed, "ours_static")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mapping = payload.get("atlas_identity_to_slot")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"No atlas_identity_to_slot in {checkpoint}")
    result = {normalize_label(key): int(value) for key, value in mapping.items()}
    if len(set(result.values())) != len(result):
        raise ValueError(f"Duplicate Atlas slots in {checkpoint}")
    return result


class AtomicCsv:
    def __init__(self, path: Path):
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.handle = None
        self.writer = None
        self.rows = 0

    def __enter__(self) -> "AtomicCsv":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.handle = self.temporary.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=FIELDS)
        self.writer.writeheader()
        return self

    def write(self, row: dict[str, Any]) -> None:
        assert self.writer is not None
        self.writer.writerow(row)
        self.rows += 1

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self.handle is not None
        self.handle.close()
        if exc_type is None:
            if self.rows == 0:
                self.temporary.unlink(missing_ok=True)
                raise RuntimeError(f"No rows written to {self.path}")
            self.temporary.replace(self.path)
        elif self.temporary.exists():
            self.temporary.unlink()


def output_path(repo: Path, method: str, dataset: str, fold: int, seed: int) -> Path:
    folder = {
        "ours_static": "ours_static",
        "ours_dynamic": "ours_dynamic",
        "fdnc": "fdnc",
        "nuclr": "nuclr",
    }[method]
    return (
        repo
        / "runs/unified_benchmark"
        / folder
        / dataset
        / f"fold{fold}"
        / f"seed{seed}"
        / "test_candidate_scores.csv"
    )


def load_canonical_query_map(
    repo: Path,
    dataset: str,
    fold: int,
    seed: int,
) -> tuple[dict[str, tuple[str, str]], Path]:
    """Load the paper query signature locked by Ours (Static)."""
    source = output_path(repo, "ours_static", dataset, fold, seed)
    if not source.is_file():
        raise FileNotFoundError(
            f"Canonical query source is missing: {source}. "
            "Export ours_static for this cell first."
        )
    result: dict[str, tuple[str, str]] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            query_uid = str(row["query_uid"])
            meta = (str(row["group_uid"]), normalize_label(row["gt_label"]))
            previous = result.setdefault(query_uid, meta)
            if previous != meta:
                raise RuntimeError(
                    f"Canonical query {query_uid} has inconsistent metadata: "
                    f"{previous} vs {meta}"
                )
    if not result:
        raise RuntimeError(f"No canonical queries in {source}")
    return result, source


def assert_exact_query_set(
    seen: set[str],
    canonical: dict[str, tuple[str, str]],
    method: str,
) -> None:
    expected = set(canonical)
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise RuntimeError(
            f"{method} cannot reproduce the canonical query set: "
            f"expected={len(expected)} seen={len(seen)} "
            f"missing={missing[:10]} extra={extra[:10]}"
        )


def write_audit(path: Path, payload: dict[str, Any]) -> None:
    audit = path.with_name("test_candidate_scores.audit.json")
    payload = {**payload, "score_file": str(path.resolve()), "score_sha256": sha256(path)}
    audit.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_ours(args: argparse.Namespace) -> Path:
    repo = args.repo_root
    path = output_path(repo, args.method, args.dataset, args.fold, args.seed)
    if path.is_file():
        print(f"[REUSE] {path}")
        return path
    package = repo / "neurid"
    sys.path.insert(0, str(package))
    from mprt_net.data import WormCache, split_files
    from mprt_net.evaluate import load_checkpoint

    checkpoint = atlas_checkpoint(repo, args.dataset, args.fold, args.seed, args.method)
    fold_root = repo / DATA_ROOTS[args.dataset] / f"fold_{args.fold}"
    device = torch.device(args.device)
    model, payload = load_checkpoint(checkpoint, device)
    mapping = {
        normalize_label(key): int(value)
        for key, value in payload["atlas_identity_to_slot"].items()
    }
    slot_to_label = {slot: label for label, slot in mapping.items()}
    slot_labels = [
        slot_to_label.get(index, f"__UNMAPPED_SLOT_{index:04d}")
        for index in range(model.config.atlas_size)
    ]
    atlas = model.atlas_encoding()
    files = split_files(fold_root, "test")
    cache = WormCache(activity_length=args.activity_length, max_items=max(32, len(files)))
    queries = 0
    model.eval()
    with AtomicCsv(path) as sink, torch.no_grad():
        for sample_path in files:
            sample_cpu = cache.get(sample_path)
            sample = sample_cpu.to(device)
            query = model.encode_population(sample)
            if args.method == "ours_static":
                output = model.match_encodings(query, atlas)
            else:
                dynamic = model.deform_atlas(query, atlas)
                output = model.match_encodings(query, dynamic.encoding)
            ranking = output.row_conditional[:, :-1].detach().float().cpu().numpy()
            assignment = output.plan[:-1, :-1].detach().float().cpu().numpy()
            labels = [normalize_label(value) for value in sample_cpu.cell_ids]
            supervised = sample_cpu.supervised_mask.detach().cpu().numpy().astype(bool)
            for query_index in unique_indices(labels, supervised):
                gt = labels[int(query_index)]
                query_uid = f"{sample_cpu.uid}::{int(query_index)}"
                queries += 1
                for slot, candidate in enumerate(slot_labels):
                    sink.write(
                        {
                            "query_uid": query_uid,
                            "group_uid": sample_cpu.uid,
                            "gt_label": gt,
                            "candidate_label": candidate,
                            "score": float(ranking[int(query_index), slot]),
                            "assignment_score": float(assignment[int(query_index), slot]),
                            "reference_uid": "TRAIN_ATLAS",
                        }
                    )
    write_audit(
        path,
        {
            "protocol": "canonical_annotation_cv5_test_v1",
            "method": args.method,
            "dataset": args.dataset,
            "fold": args.fold,
            "seed": args.seed,
            "queries": queries,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
        },
    )
    print(f"[DONE] {path} queries={queries}")
    return path


def validate_fdnc_assets(source: Path, model: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != FDNC_COMMIT:
        raise AssertionError(f"fDNC commit={head}, expected={FDNC_COMMIT}")
    for relative, expected in FDNC_SOURCE_HASHES.items():
        observed = sha256(source / relative)
        if observed != expected:
            raise AssertionError(f"Official fDNC source changed: {relative}")
    if sha256(model) != FDNC_MODEL_SHA256:
        raise AssertionError("Official fDNC model.bin hash mismatch")


def load_fdnc(source: Path, checkpoint: Path, device: torch.device):
    source_file = source / "src/model.py"
    name = "fdnc_official_unified_export"
    spec = importlib.util.spec_from_file_location(name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(source_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    model = module.NIT_Registration(
        input_dim=3,
        n_hidden=128,
        n_layer=6,
        p_rotate=0,
        feat_trans=0,
        cuda=device.type == "cuda",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def normalize_fdnc_xyz(xyz: np.ndarray) -> np.ndarray:
    centered = xyz.astype(np.float32) - np.median(xyz.astype(np.float32), axis=0, keepdims=True)
    return centered / 200.0


@torch.no_grad()
def fdnc_scores(model: Any, reference_xyz: np.ndarray, query_xyz: np.ndarray) -> np.ndarray:
    _, output = model(
        [reference_xyz, query_xyz], match_dict=None, ref_idx=0, mode="eval"
    )
    scores = output["p_m"][1, : len(query_xyz), : len(reference_xyz)]
    return scores.detach().float().cpu().numpy()


def export_fdnc(args: argparse.Namespace) -> Path:
    if args.seed != 42:
        raise ValueError("Released fDNC is deterministic; export it only with seed42")
    if args.fdnc_source is None or args.fdnc_model is None:
        raise ValueError("--fdnc-source and --fdnc-model are required")
    path = output_path(args.repo_root, "fdnc", args.dataset, args.fold, 42)
    if path.is_file():
        print(f"[REUSE] {path}")
        return path
    source = args.fdnc_source.resolve()
    released = args.fdnc_model.resolve()
    validate_fdnc_assets(source, released)
    model = load_fdnc(source, released, torch.device(args.device))
    fold_root = args.repo_root / DATA_ROOTS[args.dataset] / f"fold_{args.fold}"
    canonical, canonical_source = load_canonical_query_map(
        args.repo_root, args.dataset, args.fold, 42
    )
    universe = set(atlas_identity_mapping(args.repo_root, args.dataset, args.fold, 42))
    train = []
    for item in split_paths(fold_root, "train"):
        xyz, labels, supervised = load_npz_identity_data(item)
        train.append((recording_uid(item), normalize_fdnc_xyz(xyz), labels, unique_indices(labels, supervised)))
    test = []
    seen_queries: set[str] = set()
    for item in split_paths(fold_root, "test"):
        xyz, labels, supervised = load_npz_identity_data(item)
        group_uid = recording_uid(item)
        selected = []
        for index in unique_indices(labels, supervised):
            query_uid = f"{group_uid}::{int(index)}"
            if query_uid not in canonical:
                continue
            expected_group, expected_gt = canonical[query_uid]
            observed_gt = labels[int(index)]
            if (group_uid, observed_gt) != (expected_group, expected_gt):
                raise RuntimeError(
                    f"fDNC canonical metadata mismatch for {query_uid}: "
                    f"observed={(group_uid, observed_gt)} "
                    f"expected={(expected_group, expected_gt)}"
                )
            selected.append(int(index))
            seen_queries.add(query_uid)
        test.append((group_uid, normalize_fdnc_xyz(xyz), labels, np.asarray(selected)))
    assert_exact_query_set(seen_queries, canonical, "fDNC")
    queries = len(seen_queries)
    with AtomicCsv(path) as sink:
        for query_uid_base, query_xyz, query_labels, query_indices in test:
            for reference_uid, reference_xyz, reference_labels, reference_indices in train:
                scores = fdnc_scores(model, reference_xyz, query_xyz)
                candidate_indices = [
                    int(index)
                    for index in reference_indices
                    if reference_labels[int(index)] in universe
                ]
                for query_index in query_indices:
                    gt = query_labels[int(query_index)]
                    for candidate_index in candidate_indices:
                        value = float(scores[int(query_index), candidate_index])
                        sink.write(
                            {
                                "query_uid": f"{query_uid_base}::{int(query_index)}",
                                "group_uid": query_uid_base,
                                "gt_label": gt,
                                "candidate_label": reference_labels[candidate_index],
                                "score": value,
                                "assignment_score": value,
                                "reference_uid": reference_uid,
                            }
                        )
    write_audit(
        path,
        {
            "protocol": "canonical_annotation_cv5_test_v1",
            "method": "fdnc_official_released",
            "dataset": args.dataset,
            "fold": args.fold,
            "seed": 42,
            "queries": queries,
            "official_commit": FDNC_COMMIT,
            "official_checkpoint_sha256": FDNC_MODEL_SHA256,
            "canonical_query_source": str(canonical_source.resolve()),
            "canonical_query_source_sha256": sha256(canonical_source),
        },
    )
    print(f"[DONE] {path} queries={queries}")
    return path


def import_activity(path: Path, repo: Path):
    sys.path.insert(0, str(repo))
    name = "hybrid_activity_unified_export"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_xformers_compat() -> bool:
    """Accept newer NuCLR's device= hint with the torch-2.1 xFormers API."""
    import xformers.ops as xops

    mask = xops.fmha.BlockDiagonalMask
    original = mask.from_seqlens
    if "device" in inspect.signature(original).parameters:
        return False

    def from_seqlens_compat(cls, q_seqlen, kv_seqlen=None, device=None):
        # xFormers 0.0.23 infers execution device from the Q/K/V tensors.
        # Ignoring this newer convenience keyword leaves attention unchanged.
        return original(q_seqlen, kv_seqlen)

    mask.from_seqlens = classmethod(from_seqlens_compat)
    print(
        "[COMPAT] xFormers BlockDiagonalMask.from_seqlens: "
        "ignoring unsupported device= keyword"
    )
    return True


def checkpoint_matches_current_split(
    checkpoint: Path,
    current_fold_root: Path,
) -> bool:
    expected_train = path_stems(current_fold_root / "train")
    expected_val = path_stems(current_fold_root / "val")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload.get("args", {})
    train_root = Path(str(saved.get("train_root", "")))
    val_root = Path(str(saved.get("val_root", "")))
    return (
        train_root.is_dir()
        and val_root.is_dir()
        and path_stems(train_root) == expected_train
        and path_stems(val_root) == expected_val
    )


def find_matching_nuclr_checkpoint(
    repo: Path,
    old_root: Path,
    dataset: str,
    fold: int,
    seed: int,
    current_fold_root: Path,
) -> tuple[Path, str]:
    current = (
        repo
        / "runs/unified_benchmark/nuclr_training"
        / dataset
        / f"fold{fold}"
        / f"seed{seed}"
        / "best.pt"
    )
    if current.is_file():
        if not checkpoint_matches_current_split(current, current_fold_root):
            raise RuntimeError(f"Current NuCLR checkpoint has wrong split: {current}")
        return current, "current_cv_fold"

    candidates: list[tuple[Path, str]] = []
    for old_fold in range(1, 6):
        checkpoint = (
            old_root
            / "folds"
            / dataset
            / f"fold_{old_fold}"
            / "pipeline"
            / f"seed_{seed}"
            / "nuclr_t1st1/best.pt"
        )
        if not checkpoint.is_file():
            continue
        if checkpoint_matches_current_split(checkpoint, current_fold_root):
            candidates.append((checkpoint, f"legacy_fold_{old_fold}"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one split-identical NuCLR checkpoint for "
            f"{dataset} seed{seed} {current_fold_root}; found={candidates}. "
            "Run train_current_fold_nuclr.py for this cell first."
        )
    return candidates[0]


def apply_saved_scalers(records: Iterable[Any], scalers: dict[str, Any]) -> None:
    for name in ("absolute_features", "population_features"):
        mean = np.asarray(scalers[name]["mean"], dtype=np.float32)
        std = np.asarray(scalers[name]["std"], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        for record in records:
            values = np.asarray(getattr(record, name), dtype=np.float32)
            scaled = (values - mean) / std
            setattr(
                record,
                name,
                np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
            )


def export_nuclr(args: argparse.Namespace) -> Path:
    path = output_path(args.repo_root, "nuclr", args.dataset, args.fold, args.seed)
    if path.is_file():
        print(f"[REUSE] {path}")
        return path
    fold_root = args.repo_root / DATA_ROOTS[args.dataset] / f"fold_{args.fold}"
    canonical, canonical_source = load_canonical_query_map(
        args.repo_root, args.dataset, args.fold, args.seed
    )
    old_root = args.nuclr_run_root.resolve()
    checkpoint, checkpoint_source = find_matching_nuclr_checkpoint(
        args.repo_root,
        old_root,
        args.dataset,
        args.fold,
        args.seed,
        fold_root,
    )
    install_xformers_compat()
    activity = import_activity(args.activity_script.resolve(), args.repo_root)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = dict(payload["args"])
    saved_args.update(
        {
            "supervision_mask_key": "labeled_mask",
            "same_worm_neurons": "all",
            "device": args.device,
        }
    )
    model_args = SimpleNamespace(**saved_args)
    train_records = activity.load_records(
        fold_root / "train",
        "train",
        model_args.activity_key,
        model_args.label_key,
        model_args.source_fs,
        model_args,
    )
    test_records = activity.load_records(
        fold_root / "test",
        "test",
        model_args.activity_key,
        model_args.label_key,
        model_args.source_fs,
        model_args,
    )
    apply_saved_scalers([*train_records, *test_records], payload["feature_scalers"])
    device = torch.device(args.device)
    absolute_dim = int(train_records[0].absolute_features.shape[1])
    population_dim = int(train_records[0].population_features.shape[1])
    model = activity.AtanasMultiBranchEncoder(
        model_args.model_variant,
        absolute_dim,
        population_dim,
        model_args.feature_dropout,
        device,
        **activity.temporal_value_kwargs(model_args),
        args=model_args,
    ).to(device)
    criterion = activity.SampleWiseDCL(
        input_dim=model.output_dim,
        projection_dim=model_args.projector_dim,
        tau=model_args.temperature,
        full_denom=model_args.full_denom,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    criterion.load_state_dict(payload["loss_state_dict"], strict=True)
    model.eval()
    criterion.eval()
    universe = set(atlas_identity_mapping(args.repo_root, args.dataset, args.fold, args.seed))

    def embedded(records: Sequence[Any]) -> list[tuple[Any, np.ndarray, np.ndarray]]:
        result = []
        with torch.no_grad():
            for record in records:
                encoder, _, labels = activity.embed_record(
                    record, model, criterion, model_args, device
                )
                result.append((record, encoder, np.asarray(labels).astype(str)))
        return result

    train_emb = embedded(train_records)
    test_emb = embedded(test_records)
    seen_queries: set[str] = set()
    with AtomicCsv(path) as sink:
        for query_record, query_matrix, query_labels in test_emb:
            query_indices = query_record.valid_unique_indices
            query_group_uid = recording_uid(query_record.path)
            selected_queries: list[tuple[int, int, str, str]] = []
            for qoffset, query_index in enumerate(query_indices):
                query_uid = f"{query_group_uid}::{int(query_index)}"
                if query_uid not in canonical:
                    continue
                gt = normalize_label(query_labels[qoffset])
                expected_group, expected_gt = canonical[query_uid]
                if (query_group_uid, gt) != (expected_group, expected_gt):
                    raise RuntimeError(
                        f"NuCLR canonical metadata mismatch for {query_uid}: "
                        f"observed={(query_group_uid, gt)} "
                        f"expected={(expected_group, expected_gt)}"
                    )
                selected_queries.append((qoffset, int(query_index), query_uid, gt))
                seen_queries.add(query_uid)
            for reference_record, reference_matrix, reference_labels in train_emb:
                reference_group_uid = recording_uid(reference_record.path)
                similarity = query_matrix @ reference_matrix.T
                reference_indices = reference_record.valid_unique_indices
                for qoffset, query_index, query_uid, gt in selected_queries:
                    for coffset, candidate_index in enumerate(reference_indices):
                        candidate = normalize_label(reference_labels[coffset])
                        if candidate not in universe:
                            continue
                        value = float(similarity[qoffset, coffset])
                        sink.write(
                            {
                                "query_uid": query_uid,
                                "group_uid": query_group_uid,
                                "gt_label": gt,
                                "candidate_label": candidate,
                                "score": value,
                                "assignment_score": value,
                                "reference_uid": reference_group_uid,
                            }
                        )
    assert_exact_query_set(seen_queries, canonical, "NuCLR")
    queries = len(seen_queries)
    write_audit(
        path,
        {
            "protocol": "canonical_annotation_cv5_test_v1",
            "method": "nuclr_encoder",
            "dataset": args.dataset,
            "fold": args.fold,
            "seed": args.seed,
            "queries": queries,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "split_reuse_audit": "exact train and val stem sets",
            "checkpoint_source": checkpoint_source,
            "canonical_query_source": str(canonical_source.resolve()),
            "canonical_query_source_sha256": sha256(canonical_source),
        },
    )
    print(f"[DONE] {path} queries={queries} source={checkpoint_source}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("ours_static", "ours_dynamic", "fdnc", "nuclr"), required=True
    )
    parser.add_argument("--dataset", choices=tuple(DATA_ROOTS), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--repo-root", type=Path, default=Path("/home/ubuntu/klb/nuclr/nuclr"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--fdnc-source", type=Path, default=None)
    parser.add_argument("--fdnc-model", type=Path, default=None)
    parser.add_argument(
        "--nuclr-run-root",
        type=Path,
        default=Path(
            "/home/ubuntu/klb/nuclr/nuclr/"
            "runs/fold_pure_table1init_candidates_ab_20260817"
        ),
    )
    parser.add_argument(
        "--activity-script",
        type=Path,
        default=Path("/home/ubuntu/klb/nuclr/nuclr/hybrid/activity.py"),
    )
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return args


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    if args.method in {"ours_static", "ours_dynamic"}:
        export_ours(args)
    elif args.method == "fdnc":
        export_fdnc(args)
    else:
        export_nuclr(args)


if __name__ == "__main__":
    main()
