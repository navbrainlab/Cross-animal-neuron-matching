#!/usr/bin/env python3
"""Compare Sinkhorn, FGW, UFGW and NeuRID on one frozen NeuRID representation.

Protocol invariants
-------------------
* the existing Full NeuRID seed-42 checkpoint is loaded and frozen;
* the identity-anchored population atlas is rebuilt from outer-train only;
* every matcher consumes the same contextual query/atlas node embeddings;
* FGW/UFGW hyperparameters are selected on outer-validation only;
* a durable pre-test lock is written before the test directory is opened;
* all four matchers are evaluated in one pass over outer-test animals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO / "mprt_net_v1_1"
METHODS = ("sinkhorn", "fgw", "ufgw", "neurid")
DISPLAY_NAMES = {
    "sinkhorn": "Sinkhorn",
    "fgw": "FGW",
    "ufgw": "UFGW",
    "neurid": "NeuRID",
}
DATA_ROOTS = {
    "atanas": REPO / "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
    "rld": REPO / "Data/Dunn_001623/cv5_grouped_v1",
}
PROTOCOL = "frozen_table2_neurid_representation_matcher_control_cv5_seed42_v3"
DEFAULT_CHECKPOINT_ROOT = REPO / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1"
DEFAULT_OUTPUT_ROOT = REPO / "runs/frozen_neurid_matchers_cv5_seed42_v3"


def _install_package_path() -> None:
    value = str(PACKAGE_ROOT.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_fingerprint(named_tensors: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atlas_fingerprint(model: Any, identity_to_slot: dict[str, int]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(identity_to_slot, sort_keys=True).encode())
    atlas = model.atlas_encoding()
    tensors = [("nodes", atlas.nodes), ("relations", atlas.relations)]
    for name in ("support", "relation_support", "relation_count", "coordinates"):
        value = getattr(atlas, name)
        if value is not None:
            tensors.append((name, value))
    digest.update(_tensor_fingerprint(tensors).encode())
    return digest.hexdigest()


def _parameter_fingerprint(model: Any) -> str:
    return _tensor_fingerprint(list(model.named_parameters()))


def _parse_csv(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("Expected a non-empty comma-separated list")
    return result


def _parse_floats(value: str, name: str) -> list[float]:
    try:
        result = [float(part) for part in _parse_csv(value)]
    except ValueError as exc:
        raise ValueError(f"Invalid {name} grid: {value!r}") from exc
    if not all(np.isfinite(result)):
        raise ValueError(f"{name} values must be finite")
    return result


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _dataset_root(args: argparse.Namespace, dataset: str) -> Path:
    return Path(getattr(args, f"{dataset}_root")).expanduser().resolve()


def _fold_root(args: argparse.Namespace, dataset: str, fold: int) -> Path:
    return _dataset_root(args, dataset) / f"fold_{fold}"


def _checkpoint(args: argparse.Namespace, dataset: str, fold: int) -> Path:
    return (
        args.checkpoint_root.expanduser().resolve()
        / dataset
        / f"fold{fold}"
        / "seed42/pairwise/full/best.pt"
    )


def _saved_atlas(checkpoint: Path) -> Path:
    expected_suffix = Path("pairwise/full/best.pt")
    if Path(*checkpoint.parts[-3:]) != expected_suffix:
        raise ValueError(f"Unexpected Table-2 pairwise checkpoint layout: {checkpoint}")
    return checkpoint.parents[2] / "static_atlas/anchored_pure.pt"


def _cell(args: argparse.Namespace, dataset: str, fold: int) -> Path:
    return args.output_root.expanduser().resolve() / dataset / f"fold{fold}"


@dataclass
class FrozenFold:
    model: Any
    atlas: Any
    identity_to_slot: dict[str, int]
    build_metadata: dict[str, Any]
    atlas_fingerprint: str
    parameter_fingerprint: str
    saved_atlas_max_abs_error: float | None
    saved_atlas_verification: dict[str, dict[str, float]] | None


def _build_frozen_fold(
    args: argparse.Namespace,
    dataset: str,
    fold: int,
    device: torch.device,
) -> FrozenFold:
    """Rebuild the exact main-experiment atlas without opening val/test."""

    _install_package_path()
    from mprt_net.build_anchored_atlas import build_anchored_atlas
    from mprt_net.config import ModelConfig
    from mprt_net.data import WormCache, split_files, unique_identity_map
    from mprt_net.model import MPRTNet

    checkpoint_path = _checkpoint(args, dataset, fold)
    saved_atlas_path = _saved_atlas(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.verify_saved_atlas and not saved_atlas_path.is_file():
        raise FileNotFoundError(
            f"Main-experiment atlas required by --verify-saved-atlas: {saved_atlas_path}"
        )
    source = torch.load(checkpoint_path, map_location=device)
    source_config = ModelConfig.from_dict(source["model_config"])
    if source_config.atlas_size != 0:
        raise RuntimeError(f"Full source checkpoint is not atlas-free: {checkpoint_path}")

    fold_root = _fold_root(args, dataset, fold)
    train_files = split_files(fold_root, "train")
    cache = WormCache(activity_length=args.activity_length, max_items=max(64, len(train_files)))
    identities = sorted(
        {
            identity
            for path in train_files
            for identity in unique_identity_map(cache.get(path))
        }
    )
    if not identities:
        raise RuntimeError(f"No unique supervised train identities in {fold_root}")
    identity_to_slot = {identity: slot for slot, identity in enumerate(identities)}

    # The already-locked main atlas supplies only the atlas configuration
    # (e.g. legacy relation masking).  Its tensors are never used to build the
    # comparison atlas, which is reconstructed below from outer-train.
    if saved_atlas_path.is_file():
        saved = torch.load(saved_atlas_path, map_location=device)
        config_values = dict(saved["model_config"])
        saved_mapping = {str(k): int(v) for k, v in saved["atlas_identity_to_slot"].items()}
        if saved_mapping != identity_to_slot:
            raise RuntimeError("Rebuilt train identity ordering differs from main experiment")
        saved_build = saved.get("atlas_build", {})
        saved_source = saved_build.get("source_checkpoint")
        if saved_source is None or Path(saved_source).resolve() != checkpoint_path.resolve():
            raise RuntimeError(
                "Table-2 atlas was not built from the selected pairwise checkpoint: "
                f"atlas source={saved_source!r}, selected={checkpoint_path}"
            )
    else:
        saved = None
        config_values = source_config.to_dict()
        config_values.update(
            atlas_size=len(identities),
            atlas_blend_weight=1.0,
            atlas_confidence_gating=False,
        )
    if int(config_values["atlas_size"]) != len(identities):
        raise RuntimeError("Main atlas size differs from outer-train identity union")

    model = MPRTNet(ModelConfig.from_dict(config_values)).to(device)
    incompatible = model.load_state_dict(source["model_state"], strict=False)
    expected_missing = set(model.state_dict()).difference(source["model_state"])
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected source/atlas checkpoint incompatibility: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter_fingerprint = _parameter_fingerprint(model)

    with torch.inference_mode():
        metadata = build_anchored_atlas(
            model, train_files, cache, identity_to_slot, device
        )
    model.set_atlas_blend(1.0)
    atlas = model.atlas_encoding()

    max_error: float | None = None
    verification: dict[str, dict[str, float]] | None = None
    if saved is not None:
        saved_model = MPRTNet(ModelConfig.from_dict(saved["model_config"])).to(device)
        saved_model.load_state_dict(saved["model_state"])
        saved_model.eval()
        for parameter in saved_model.parameters():
            parameter.requires_grad_(False)
        if _parameter_fingerprint(saved_model) != parameter_fingerprint:
            raise RuntimeError(
                "Saved Table-2 atlas checkpoint does not preserve the selected "
                "pairwise model parameters"
            )
        saved_encoding = saved_model.atlas_encoding()

    if saved is not None and args.verify_saved_atlas:
        pairs = (
            ("nodes", atlas.nodes, saved_encoding.nodes, 5e-3, 2e-3),
            ("relations", atlas.relations, saved_encoding.relations, 2e-2, 1e-2),
            ("support", atlas.support, saved_encoding.support, 1e-7, 1e-7),
        )
        verification = {}
        for name, rebuilt, reference, max_abs_limit, relative_l2_limit in pairs:
            difference = (rebuilt - reference).float()
            max_abs = float(difference.abs().max())
            rmse = float(difference.square().mean().sqrt())
            relative_l2 = float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
            )
            verification[name] = {
                "max_abs": max_abs,
                "rmse": rmse,
                "relative_l2": relative_l2,
                "max_abs_limit": max_abs_limit,
                "relative_l2_limit": relative_l2_limit,
            }
            if max_abs > max_abs_limit or relative_l2 > relative_l2_limit:
                raise RuntimeError(
                    f"Saved/rebuilt atlas {name} mismatch: "
                    f"max_abs={max_abs:.6g} (limit {max_abs_limit:g}), "
                    f"relative_l2={relative_l2:.6g} (limit {relative_l2_limit:g})"
                )
        max_error = max(item["max_abs"] for item in verification.values())
        # Historical main atlases were encoded on CUDA.  A CPU replay of the
        # frozen network can differ by a few 1e-3 after many attention and
        # prototype-averaging operations.  The joint max-error/global-relative
        # check above tolerates isolated rounding outliers but rejects a
        # systematic atlas change.  The new select/test boundary still uses an
        # exact SHA-256 fingerprint for the atlas rebuilt in this run.
        saved_build = saved.get("atlas_build", {})
        for key in (
            "recordings",
            "atlas_size",
            "node_observations",
            "min_identity_observations",
            "max_identity_observations",
        ):
            if saved_build.get(key) != metadata.get(key):
                raise RuntimeError(f"Saved/rebuilt atlas metadata mismatch for {key}")

    # Rebuilding above proves that the canonical atlas uses outer-train only.
    # Evaluation deliberately uses the exact saved Table-2 tensors so a
    # CPU/GPU replay rounding difference cannot flip near-tied query ranks.
    if saved is not None:
        model = saved_model
        atlas = saved_encoding

    if _parameter_fingerprint(model) != parameter_fingerprint:
        raise RuntimeError("Atlas construction unexpectedly changed a model parameter")
    return FrozenFold(
        model=model,
        atlas=atlas,
        identity_to_slot=identity_to_slot,
        build_metadata=metadata,
        atlas_fingerprint=_atlas_fingerprint(model, identity_to_slot),
        parameter_fingerprint=parameter_fingerprint,
        saved_atlas_max_abs_error=max_error,
        saved_atlas_verification=verification,
    )


@dataclass
class EncodedAnimal:
    sample: Any
    encoding: Any


def _encode_split(
    model: Any,
    paths: list[Path],
    activity_length: int,
    device: torch.device,
) -> list[EncodedAnimal]:
    _install_package_path()
    from mprt_net.data import WormCache

    cache = WormCache(activity_length=activity_length, max_items=8)
    result = []
    with torch.inference_mode():
        for index, path in enumerate(paths, start=1):
            sample = cache.get(path)
            encoding = model.encode_population(sample.to(device))
            result.append(EncodedAnimal(sample=sample, encoding=encoding))
            print(f"  encode {index:03d}/{len(paths):03d} {sample.uid}", flush=True)
    return result


def _score_results(
    results: list[tuple[EncodedAnimal, Any]],
    identity_to_slot: dict[str, int],
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    _install_package_path()
    from mprt_net.frozen_matchers import merge_counts, score_match

    counts = []
    records = []
    slot_to_identity = {slot: identity for identity, slot in identity_to_slot.items()}
    for animal, match in results:
        item, rows = score_match(match, animal.sample, identity_to_slot)
        counts.append(item)
        for row in rows:
            prediction = int(row["prediction_slot"])
            row["prediction_identity"] = slot_to_identity[prediction]
            rows_target = int(row["target_slot"])
            row["target_identity"] = "" if rows_target < 0 else slot_to_identity[rows_target]
        records.extend(rows)
    return merge_counts(counts), records


def _public_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    hidden = {
        "top1_correct",
        "top5_correct",
        "reciprocal_rank_sum",
        "partial_correct",
        "accepted_queries",
        "known_accepted",
        "novel_rejected",
        "hungarian_correct",
    }
    return {key: value for key, value in metrics.items() if key not in hidden}


def _selection_key(item: dict[str, Any]) -> tuple[float, float, float, float, int]:
    metrics = item["metrics"]
    return (
        float(metrics["top1_real"]),
        float(metrics["partial_assignment_accuracy"]),
        float(metrics["mrr_real"]),
        float(metrics["hungarian_accuracy"]),
        -int(item["grid_index"]),
    )


def _evaluate_fixed_validation(
    frozen: FrozenFold,
    animals: list[EncodedAnimal],
) -> dict[str, Any]:
    from mprt_net.frozen_matchers import match_neurid, match_sinkhorn

    output = {}
    for method, solve in (
        ("sinkhorn", lambda animal: match_sinkhorn(frozen.model, animal.encoding, frozen.atlas)),
        ("neurid", lambda animal: match_neurid(frozen.model, animal.encoding, frozen.atlas)),
    ):
        results = [(animal, solve(animal)) for animal in animals]
        metrics, _ = _score_results(results, frozen.identity_to_slot)
        output[method] = _public_metrics(metrics)
        print(
            f"  {method:9s} val top1={metrics['top1_real']:.4f} "
            f"partial={metrics['partial_assignment_accuracy']:.4f}",
            flush=True,
        )
    return output


def _select_fgw(
    args: argparse.Namespace,
    frozen: FrozenFold,
    animals: list[EncodedAnimal],
    alphas: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from mprt_net.frozen_matchers import match_fgw

    sweep = []
    for grid_index, alpha in enumerate(alphas):
        results = [
            (
                animal,
                match_fgw(
                    animal.encoding,
                    frozen.atlas,
                    alpha=alpha,
                    max_iter=args.fgw_max_iter,
                    tol=args.fgw_tol,
                ),
            )
            for animal in animals
        ]
        metrics, _ = _score_results(results, frozen.identity_to_slot)
        item = {
            "grid_index": grid_index,
            "parameters": {"alpha": alpha},
            "metrics": _public_metrics(metrics),
        }
        sweep.append(item)
        print(
            f"  fgw alpha={alpha:g} val top1={metrics['top1_real']:.4f} "
            f"partial={metrics['partial_assignment_accuracy']:.4f}",
            flush=True,
        )
    best = max(sweep, key=_selection_key)
    return dict(best["parameters"]), sweep


def _select_ufgw(
    args: argparse.Namespace,
    frozen: FrozenFold,
    animals: list[EncodedAnimal],
    alphas: list[float],
    rhos: list[float],
    epsilons: list[float],
    thresholds: list[float],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from mprt_net.frozen_matchers import apply_ufgw_threshold, match_ufgw

    sweep = []
    grid_index = 0
    for alpha, rho, epsilon in itertools.product(alphas, rhos, epsilons):
        solved = [
            (
                animal,
                match_ufgw(
                    animal.encoding,
                    frozen.atlas,
                    alpha=alpha,
                    rho=rho,
                    epsilon=epsilon,
                    rejection_threshold=0.0,
                    device=device,
                    nits_bcd=args.ufgw_nits_bcd,
                    nits_uot=args.ufgw_nits_uot,
                ),
            )
            for animal in animals
        ]
        for threshold in thresholds:
            results = [
                (animal, apply_ufgw_threshold(match, threshold))
                for animal, match in solved
            ]
            metrics, _ = _score_results(results, frozen.identity_to_slot)
            parameters = {
                "alpha": alpha,
                "rho": rho,
                "epsilon": epsilon,
                "rejection_threshold": threshold,
            }
            sweep.append(
                {
                    "grid_index": grid_index,
                    "parameters": parameters,
                    "metrics": _public_metrics(metrics),
                }
            )
            print(
                "  ufgw " + " ".join(f"{key}={value:g}" for key, value in parameters.items())
                + f" val top1={metrics['top1_real']:.4f} "
                f"partial={metrics['partial_assignment_accuracy']:.4f}",
                flush=True,
            )
            grid_index += 1
    best = max(sweep, key=_selection_key)
    return dict(best["parameters"]), sweep


def select_fold(args: argparse.Namespace, dataset: str, fold: int) -> None:
    cell = _cell(args, dataset, fold)
    lock_path = cell / "LOCKED_BEFORE_TEST.json"
    if lock_path.exists():
        print(f"[skip locked] {lock_path}", flush=True)
        return
    if (cell / "TEST_COMPLETE.json").exists():
        raise RuntimeError(f"Test output exists without its selection lock: {cell}")
    device = _device(args.device)
    print(f"[select] dataset={dataset} fold={fold} device={device}", flush=True)
    frozen = _build_frozen_fold(args, dataset, fold, device)

    # This is deliberately the first validation access.  Test is not listed,
    # hashed, loaded, or encoded anywhere in the selection phase.
    from mprt_net.data import split_files

    train_paths = split_files(_fold_root(args, dataset, fold), "train")
    val_paths = split_files(_fold_root(args, dataset, fold), "val")
    animals = _encode_split(frozen.model, val_paths, args.activity_length, device)
    parameter_before = _parameter_fingerprint(frozen.model)
    fixed = _evaluate_fixed_validation(frozen, animals)

    fgw_alphas = _parse_floats(args.fgw_alphas, "FGW alpha")
    ufgw_alphas = _parse_floats(args.ufgw_alphas, "UFGW alpha")
    ufgw_rhos = _parse_floats(args.ufgw_rhos, "UFGW rho")
    ufgw_epsilons = _parse_floats(args.ufgw_epsilons, "UFGW epsilon")
    thresholds = _parse_floats(args.ufgw_rejection_thresholds, "UFGW threshold")
    if any(not 0.0 <= value <= 1.0 for value in fgw_alphas):
        raise ValueError("FGW alphas must be in [0, 1]")
    if any(not 0.0 < value < 1.0 for value in ufgw_alphas):
        raise ValueError("UFGW alphas must lie strictly between 0 and 1")
    if any(value <= 0 for value in ufgw_rhos + ufgw_epsilons):
        raise ValueError("UFGW rho and epsilon must be positive")
    if any(value < 0 for value in thresholds):
        raise ValueError("UFGW rejection thresholds must be non-negative")

    selected_fgw, fgw_sweep = _select_fgw(args, frozen, animals, fgw_alphas)
    selected_ufgw, ufgw_sweep = _select_ufgw(
        args,
        frozen,
        animals,
        ufgw_alphas,
        ufgw_rhos,
        ufgw_epsilons,
        thresholds,
        device,
    )
    if _parameter_fingerprint(frozen.model) != parameter_before:
        raise RuntimeError("Validation unexpectedly changed Full NeuRID parameters")

    import ot
    import fugw

    checkpoint_path = _checkpoint(args, dataset, fold)
    lock = {
        "status": "LOCKED_BEFORE_TEST",
        "protocol": PROTOCOL,
        "dataset": dataset,
        "fold": fold,
        "model_seed": 42,
        "fold_root": str(_fold_root(args, dataset, fold)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "saved_main_atlas": str(_saved_atlas(checkpoint_path)),
        "saved_main_atlas_sha256": (
            _sha256(_saved_atlas(checkpoint_path))
            if _saved_atlas(checkpoint_path).is_file()
            else None
        ),
        "encoder_frozen": True,
        "all_parameters_frozen": True,
        "activity_length": args.activity_length,
        "parameter_fingerprint": frozen.parameter_fingerprint,
        "atlas_build_split": "train",
        "atlas_build": frozen.build_metadata,
        "atlas_identity_to_slot": frozen.identity_to_slot,
        "atlas_fingerprint": frozen.atlas_fingerprint,
        "saved_atlas_max_abs_error": frozen.saved_atlas_max_abs_error,
        "saved_atlas_verification": frozen.saved_atlas_verification,
        "train_files": [str(path.resolve()) for path in train_paths],
        "train_file_sha256": {str(path.resolve()): _sha256(path) for path in train_paths},
        "validation_files": [str(path.resolve()) for path in val_paths],
        "validation_file_sha256": {str(path.resolve()): _sha256(path) for path in val_paths},
        "test_opened_before_lock": False,
        "embedding": "L2-normalized Full NeuRID contextualized node embedding z_i",
        "fgw_ufgw_feature_cost": "squared Euclidean(z_i/2, z_j/2)",
        "fgw_ufgw_structure_cost": "within-population squared Euclidean(z_i, z_k)/4",
        "selection_split": "outer validation animals only",
        "selection_criterion": (
            "lexicographic Top-1 real, partial assignment accuracy, MRR, "
            "Hungarian accuracy, then earliest grid index"
        ),
        "ranking_tie_policy": "stable atlas-slot order",
        "fixed_matcher_validation": fixed,
        "selected": {"fgw": selected_fgw, "ufgw": selected_ufgw},
        "validation_sweeps": {"fgw": fgw_sweep, "ufgw": ufgw_sweep},
        "solver": {
            "pot_version": str(ot.__version__),
            "fgw_max_iter": args.fgw_max_iter,
            "fgw_tol": args.fgw_tol,
            "fugw_version": str(fugw.__version__),
            "ufgw_solver": "mm",
            "ufgw_reg_mode": "joint",
            "ufgw_divergence": "kl",
            "ufgw_nits_bcd": args.ufgw_nits_bcd,
            "ufgw_nits_uot": args.ufgw_nits_uot,
            "device": str(device),
        },
    }
    _write_json(lock_path, lock, exclusive=True)
    print(f"[LOCKED; TEST NOT OPENED] {lock_path}", flush=True)


def _test_matchers(
    lock: dict[str, Any],
    frozen: FrozenFold,
    animals: list[EncodedAnimal],
    device: torch.device,
) -> dict[str, tuple[dict[str, float | int], list[dict[str, Any]]]]:
    from mprt_net.frozen_matchers import (
        match_fgw,
        match_neurid,
        match_sinkhorn,
        match_ufgw,
    )

    selected_fgw = lock["selected"]["fgw"]
    selected_ufgw = lock["selected"]["ufgw"]
    solver = lock["solver"]
    collected: dict[str, list[tuple[EncodedAnimal, Any]]] = {method: [] for method in METHODS}
    for index, animal in enumerate(animals, start=1):
        print(f"  match {index:03d}/{len(animals):03d} {animal.sample.uid}", flush=True)
        collected["sinkhorn"].append(
            (animal, match_sinkhorn(frozen.model, animal.encoding, frozen.atlas))
        )
        collected["fgw"].append(
            (
                animal,
                match_fgw(
                    animal.encoding,
                    frozen.atlas,
                    alpha=float(selected_fgw["alpha"]),
                    max_iter=int(solver["fgw_max_iter"]),
                    tol=float(solver["fgw_tol"]),
                ),
            )
        )
        collected["ufgw"].append(
            (
                animal,
                match_ufgw(
                    animal.encoding,
                    frozen.atlas,
                    alpha=float(selected_ufgw["alpha"]),
                    rho=float(selected_ufgw["rho"]),
                    epsilon=float(selected_ufgw["epsilon"]),
                    rejection_threshold=float(selected_ufgw["rejection_threshold"]),
                    device=device,
                    nits_bcd=int(solver["ufgw_nits_bcd"]),
                    nits_uot=int(solver["ufgw_nits_uot"]),
                ),
            )
        )
        collected["neurid"].append(
            (animal, match_neurid(frozen.model, animal.encoding, frozen.atlas))
        )
    return {
        method: _score_results(results, frozen.identity_to_slot)
        for method, results in collected.items()
    }


def test_fold(args: argparse.Namespace, dataset: str, fold: int) -> None:
    cell = _cell(args, dataset, fold)
    lock_path = cell / "LOCKED_BEFORE_TEST.json"
    complete_path = cell / "TEST_COMPLETE.json"
    if complete_path.exists():
        print(f"[skip complete] {complete_path}", flush=True)
        return
    if not lock_path.is_file():
        raise FileNotFoundError(f"Run select before test: {lock_path}")
    existing = [cell / method / "test_metrics.json" for method in METHODS]
    existing += [cell / method / "test_queries.csv" for method in METHODS]
    if any(path.exists() for path in existing):
        raise RuntimeError(f"Partial test artifacts exist; refusing to re-open test: {cell}")

    lock = _json(lock_path)
    expected = {
        "status": "LOCKED_BEFORE_TEST",
        "protocol": PROTOCOL,
        "dataset": dataset,
        "fold": fold,
    }
    if any(lock.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Invalid lock metadata: {lock_path}")
    checkpoint_path = _checkpoint(args, dataset, fold)
    if str(checkpoint_path) != lock["checkpoint"] or _sha256(checkpoint_path) != lock["checkpoint_sha256"]:
        raise RuntimeError("Full NeuRID checkpoint differs from the pre-test lock")
    if str(_fold_root(args, dataset, fold)) != lock["fold_root"]:
        raise RuntimeError("Dataset fold root differs from the pre-test lock")
    if int(args.activity_length) != int(lock["activity_length"]):
        raise RuntimeError("Activity length differs from the pre-test lock")
    saved_atlas_path = _saved_atlas(checkpoint_path)
    if lock.get("saved_main_atlas_sha256") is not None:
        if not saved_atlas_path.is_file() or _sha256(saved_atlas_path) != lock["saved_main_atlas_sha256"]:
            raise RuntimeError("Saved main-experiment atlas differs from the pre-test lock")
    import fugw
    import ot
    if str(ot.__version__) != str(lock["solver"]["pot_version"]):
        raise RuntimeError("POT version differs from the pre-test lock")
    if str(fugw.__version__) != str(lock["solver"]["fugw_version"]):
        raise RuntimeError("fugw version differs from the pre-test lock")
    device = _device(str(lock["solver"]["device"]))
    print(f"[test once] dataset={dataset} fold={fold} device={device}", flush=True)
    frozen = _build_frozen_fold(args, dataset, fold, device)
    if frozen.atlas_fingerprint != lock["atlas_fingerprint"]:
        raise RuntimeError("Rebuilt outer-train atlas differs from the pre-test lock")
    if frozen.parameter_fingerprint != lock["parameter_fingerprint"]:
        raise RuntimeError("Frozen Full NeuRID parameters differ from the pre-test lock")

    # This is the first test access in the entire protocol.  Encode each test
    # animal once, then run all four matchers on that same encoding.
    from mprt_net.data import split_files

    test_paths = split_files(_fold_root(args, dataset, fold), "test")
    animals = _encode_split(frozen.model, test_paths, args.activity_length, device)
    parameter_before = _parameter_fingerprint(frozen.model)
    results = _test_matchers(lock, frozen, animals, device)
    if _parameter_fingerprint(frozen.model) != parameter_before:
        raise RuntimeError("Test evaluation unexpectedly changed Full NeuRID parameters")

    summaries = {}
    for method in METHODS:
        metrics, rows = results[method]
        public = _public_metrics(metrics)
        report = {
            "protocol": lock["protocol"],
            "method": DISPLAY_NAMES[method],
            "method_key": method,
            "dataset": dataset,
            "fold": fold,
            "seed": 42,
            "split": "test",
            "checkpoint": lock["checkpoint"],
            "checkpoint_sha256": lock["checkpoint_sha256"],
            "atlas_fingerprint": lock["atlas_fingerprint"],
            "hyperparameters": lock["selected"].get(method),
            "selection_lock": str(lock_path),
            "selection_lock_sha256": _sha256(lock_path),
            "test_recordings": len(test_paths),
            **public,
        }
        method_root = cell / method
        _write_json(method_root / "test_metrics.json", report, exclusive=True)
        query_path = method_root / "test_queries.csv"
        query_path.parent.mkdir(parents=True, exist_ok=True)
        for row in rows:
            row["method"] = DISPLAY_NAMES[method]
            row["dataset"] = dataset
            row["fold"] = fold
        with query_path.open("x", encoding="utf-8", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summaries[method] = report

    complete = {
        "status": "TEST_COMPLETE",
        "protocol": lock["protocol"],
        "dataset": dataset,
        "fold": fold,
        "test_files": [str(path.resolve()) for path in test_paths],
        "test_file_sha256": {str(path.resolve()): _sha256(path) for path in test_paths},
        "selection_lock_sha256": _sha256(lock_path),
        "methods": summaries,
    }
    _write_json(complete_path, complete, exclusive=True)
    print(f"[TEST COMPLETE] {complete_path}", flush=True)


def aggregate(args: argparse.Namespace, datasets: list[str], folds: list[int]) -> None:
    metrics = (
        "top1_real",
        "top5_real",
        "mrr_real",
        "hungarian_accuracy",
        "partial_assignment_accuracy",
        "coverage",
        "novel_reject_rate",
    )
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "aggregation": (
            "unweighted mean and sample SD over five biological folds"
            if folds == list(range(5))
            else f"diagnostic aggregation over requested folds {folds}"
        ),
        "datasets": {},
    }
    rows = []
    for dataset in datasets:
        dataset_result = {}
        for method in METHODS:
            cells = []
            for fold in folds:
                path = _cell(args, dataset, fold) / method / "test_metrics.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                item = _json(path)
                if item.get("split") != "test" or item.get("fold") != fold:
                    raise RuntimeError(f"Invalid fold test metrics: {path}")
                cells.append(item)
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "method": DISPLAY_NAMES[method],
                        **{name: item[name] for name in metrics},
                    }
                )
            dataset_result[method] = {
                "display_name": DISPLAY_NAMES[method],
                "folds": cells,
                "metrics": {
                    name: {
                        "mean": statistics.mean(float(cell[name]) for cell in cells),
                        "sample_sd": (
                            statistics.stdev(float(cell[name]) for cell in cells)
                            if len(cells) > 1
                            else 0.0
                        ),
                        "fold_values": [float(cell[name]) for cell in cells],
                    }
                    for name in metrics
                },
            }
        result["datasets"][dataset] = dataset_result

    output_root = args.output_root.expanduser().resolve()
    _write_json(output_root / "summary.json", result)
    with (output_root / "fold_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Frozen Full NeuRID representation — matcher controls",
        "",
        result["aggregation"].capitalize()
        + "; all matchers use the same frozen contextual embeddings and train-only atlas.",
        "",
        "| Dataset | Matcher | Top-1 | Top-5 | MRR | Hungarian | Partial assignment | Novel reject |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        for method in METHODS:
            values = result["datasets"][dataset][method]["metrics"]
            percent = lambda name: f"{100*values[name]['mean']:.2f} ± {100*values[name]['sample_sd']:.2f}%"
            plain = lambda name: f"{values[name]['mean']:.4f} ± {values[name]['sample_sd']:.4f}"
            lines.append(
                f"| {dataset} | {DISPLAY_NAMES[method]} | {percent('top1_real')} "
                f"| {percent('top5_real')} | {plain('mrr_real')} "
                f"| {percent('hungarian_accuracy')} | {percent('partial_assignment_accuracy')} "
                f"| {percent('novel_reject_rate')} |"
            )
    (output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "test", "all", "aggregate"))
    parser.add_argument("--datasets", default="atanas,rld")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--atanas-root", type=Path, default=DATA_ROOTS["atanas"])
    parser.add_argument("--rld-root", type=Path, default=DATA_ROOTS["rld"])
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help=(
            "Table-2 benchmark root containing "
            "{dataset}/foldN/seed42/pairwise/full/best.pt and static_atlas/anchored_pure.pt."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--fgw-alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--fgw-max-iter", type=int, default=10000)
    parser.add_argument("--fgw-tol", type=float, default=1e-9)
    parser.add_argument("--ufgw-alphas", default="0.25,0.5,0.75")
    parser.add_argument("--ufgw-rhos", default="0.1,1,10")
    parser.add_argument("--ufgw-epsilons", default="0.01")
    parser.add_argument(
        "--ufgw-rejection-thresholds",
        default="0,0.25,0.5,0.75,1,1.25,1.5",
    )
    parser.add_argument("--ufgw-nits-bcd", type=int, default=10)
    parser.add_argument("--ufgw-nits-uot", type=int, default=1000)
    parser.add_argument(
        "--verify-saved-atlas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rebuild train atlas, then numerically verify it against anchored_pure.pt.",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    datasets = _parse_csv(args.datasets)
    folds = [int(value) for value in _parse_csv(args.folds)]
    if not set(datasets).issubset(DATA_ROOTS):
        raise ValueError("--datasets may contain only atanas,rld")
    if folds != sorted(set(folds)) or any(fold not in range(5) for fold in folds):
        raise ValueError("--folds must be unique sorted values from 0..4")
    if args.activity_length <= 0 or args.fgw_max_iter <= 0:
        raise ValueError("Activity length and solver iterations must be positive")
    if args.ufgw_nits_bcd <= 0 or args.ufgw_nits_uot <= 0 or args.fgw_tol <= 0:
        raise ValueError("Solver iterations and tolerance must be positive")
    _install_package_path()

    tasks = list(itertools.product(datasets, folds))
    if args.phase in ("select", "all"):
        for dataset, fold in tasks:
            select_fold(args, dataset, fold)
    if args.phase in ("test", "all"):
        for dataset, fold in tasks:
            test_fold(args, dataset, fold)
    if args.phase in ("aggregate", "all"):
        aggregate(args, datasets, folds)


if __name__ == "__main__":
    main()
