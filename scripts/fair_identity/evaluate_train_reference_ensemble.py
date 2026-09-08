#!/usr/bin/env python3
"""Evaluate pairwise matchers against one training-geometry medoid template.

Supported adapters are CPD, fDNC, pre-extracted NuCLR, and an MPRT-family
checkpoint (including compatible FGW/RGM ablations). GeoTransformer is handled
by the preserved full official pipeline under ``../geotransformer_official``.
For each outer fold, the template is selected
using only the normalized geometry of outer-training animals.  Every held-out
animal is then matched independently to that fixed training template.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mprt_net_v1_1"))

from baselines.atanas_locked import cpd_scores, normalize_xyz
from scripts.lib.fair_identity_protocol import (
    IdentityAnimal,
    ensemble_pairwise_scores,
    evaluate_identity_scores,
    pool_metrics,
    select_geometry_medoid,
    training_vocabulary,
    unique_identity_map,
)
from mprt_net.data import WormCache, WormSample, split_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("cpd", "fdnc", "nuclr", "mprt"),
        required=True,
    )
    parser.add_argument("--method-label", default=None)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--embedding-root",
        type=Path,
        help="Directory of NuCLR NPZs containing worm_id and embedding for all train/test animals",
    )
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--normalization", choices=("softmax", "zscore"), default="zscore")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel reference-pair workers for CPU CPD; neural adapters remain serial",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def identity_animal(sample: WormSample) -> IdentityAnimal:
    return IdentityAnimal(
        uid=sample.uid,
        labels=sample.cell_ids,
        supervised_mask=sample.supervised_mask.cpu().numpy().astype(bool),
    )


def cache_key(signature: str, query: WormSample, reference: WormSample) -> str:
    digest = hashlib.sha256(
        f"{signature}\0{query.uid}\0{reference.uid}\0{query.num_nodes}\0{reference.num_nodes}".encode()
    ).hexdigest()[:24]
    return digest


def make_raw_scorer(
    args: argparse.Namespace,
    samples: dict[str, WormSample],
    device: torch.device,
) -> Callable[[IdentityAnimal, IdentityAnimal], np.ndarray]:
    model = None
    nuclr_embeddings: dict[str, np.ndarray] = {}
    signature = args.method
    if args.method == "mprt":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for mprt")
        from mprt_net.evaluate import load_checkpoint

        model, _ = load_checkpoint(args.checkpoint, device)
        model.eval()
        signature += f":{hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}"
    elif args.method == "fdnc":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for fdnc")
        from engines import evaluate_atanas_fdnc_unified as fdnc_loader

        model = fdnc_loader.load_checkpoint(args.checkpoint, device, 128, 6)
        model.eval()
        signature += f":{hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}"
    elif args.method == "nuclr":
        if args.embedding_root is None:
            raise ValueError("--embedding-root is required for nuclr")
        for path in sorted(args.embedding_root.glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                uid = str(np.asarray(data["worm_id"]).reshape(()).item())
                embedding = np.asarray(data["embedding"], dtype=np.float64)
            if uid in nuclr_embeddings:
                raise ValueError(f"Duplicate NuCLR embedding UID: {uid}")
            embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
            nuclr_embeddings[uid] = embedding
        missing = set(samples).difference(nuclr_embeddings)
        if missing:
            raise FileNotFoundError(f"Missing NuCLR embeddings for: {sorted(missing)}")
        digest = hashlib.sha256()
        for uid in sorted(samples):
            digest.update(uid.encode())
            digest.update(np.ascontiguousarray(nuclr_embeddings[uid]).tobytes())
        signature += f":{digest.hexdigest()}"
    @torch.inference_mode()
    def score(query: IdentityAnimal, reference: IdentityAnimal) -> np.ndarray:
        query_sample = samples[query.uid]
        reference_sample = samples[reference.uid]
        destination = args.cache_dir / f"{cache_key(signature, query_sample, reference_sample)}.npz"
        if destination.is_file():
            with np.load(destination, allow_pickle=False) as data:
                if str(data["query_uid"].item()) != query.uid or str(data["reference_uid"].item()) != reference.uid:
                    raise RuntimeError(f"Score-cache collision: {destination}")
                value = np.asarray(data["scores"], dtype=np.float64)
            return value

        if args.method == "cpd":
            query_xyz = normalize_xyz(query_sample.xyz.cpu().numpy())
            reference_xyz = normalize_xyz(reference_sample.xyz.cpu().numpy())
            value, _ = cpd_scores(query_xyz, reference_xyz)
        elif args.method == "fdnc":
            from scripts.lib import fdnc_scoring

            query_xyz = torch.from_numpy(normalize_xyz(query_sample.xyz.cpu().numpy())).to(
                device=device, dtype=torch.float32
            )
            reference_xyz = torch.from_numpy(
                normalize_xyz(reference_sample.xyz.cpu().numpy())
            ).to(device=device, dtype=torch.float32)
            # score_pair(a, b) returns b->a first and a->b second.
            _, query_to_reference = fdnc_scoring.score_pair(
                model, query_xyz, reference_xyz
            )
            value = query_to_reference[:, :-1].float().cpu().numpy()
        elif args.method == "nuclr":
            query_embedding = nuclr_embeddings[query.uid]
            reference_embedding = nuclr_embeddings[reference.uid]
            if len(query_embedding) != query_sample.num_nodes:
                raise ValueError(f"{query.uid}: NuCLR embedding/node count mismatch")
            if len(reference_embedding) != reference_sample.num_nodes:
                raise ValueError(f"{reference.uid}: NuCLR embedding/node count mismatch")
            value = query_embedding @ reference_embedding.T
        else:
            output = model(query_sample.to(device), reference_sample.to(device))
            value = output.row_conditional[:, :-1].detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            query_uid=np.asarray(query.uid),
            reference_uid=np.asarray(reference.uid),
            scores=value,
        )
        return value

    return score


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write empty query table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    train_paths = split_files(args.fold_root, "train")
    test_paths = split_files(args.fold_root, "test")
    cache = WormCache(args.activity_length, max_items=len(train_paths) + len(test_paths))
    train_samples = [cache.get(path) for path in train_paths]
    test_samples = [cache.get(path) for path in test_paths]
    all_samples = train_samples + test_samples
    sample_by_uid = {sample.uid: sample for sample in all_samples}
    if len(sample_by_uid) != len(all_samples):
        raise ValueError("Training and test animals must have globally unique UIDs")
    normalized_train_xyz = [normalize_xyz(sample.xyz.cpu().numpy()) for sample in train_samples]
    medoid = select_geometry_medoid(
        [sample.uid for sample in train_samples], normalized_train_xyz
    )
    template_sample = train_samples[medoid.index]
    references = [identity_animal(template_sample)]
    fixed_vocabulary = training_vocabulary(references)
    fixed_vocabulary_set = set(fixed_vocabulary)
    scorer = make_raw_scorer(args, sample_by_uid, device)

    all_rows: list[dict] = []
    animal_metrics: list[dict] = []
    vocabulary: tuple[str, ...] | None = None
    vocabulary_support: np.ndarray | None = None
    skipped_test_animals: list[dict[str, str]] = []
    for number, sample in enumerate(test_samples, start=1):
        query = identity_animal(sample)
        if not (set(unique_identity_map(query)) & fixed_vocabulary_set):
            skipped_test_animals.append(
                {
                    "query_uid": query.uid,
                    "reason": "no clean test identity occurs in the training vocabulary",
                }
            )
            print(
                f"method={args.method_label or args.method} animal={number}/{len(test_samples)} "
                f"uid={query.uid} skipped=no_training_vocabulary_overlap",
                flush=True,
            )
            continue
        if args.method == "cpd" and args.workers > 1:
            # Kept for CLI compatibility.  A single medoid template requires
            # only one registration, so no test-time reference pool exists.
            with ThreadPoolExecutor(max_workers=1) as executor:
                list(executor.map(lambda reference: scorer(query, reference), references))
        ensemble = ensemble_pairwise_scores(
            query, references, scorer, normalization=args.normalization
        )
        vocabulary = ensemble.vocabulary
        vocabulary_support = ensemble.support_animals
        aggregation = "mean_score"
        metrics, rows = evaluate_identity_scores(query, ensemble, aggregation=aggregation)
        animal_metrics.append({"query_uid": query.uid, "aggregation": "template_score", **metrics})
        all_rows.extend(
            {
                "method": args.method_label or args.method,
                "aggregation": "template_score",
                "reference_uid": medoid.uid,
                **row,
            }
            for row in rows
        )
        print(
            f"method={args.method_label or args.method} animal={number}/{len(test_samples)} "
            f"uid={query.uid} vocabulary={len(ensemble.vocabulary)}",
            flush=True,
        )

    pooled = {"template_score": pool_metrics(animal_metrics)}
    report = {
        "protocol": "outer_training_geometry_medoid_template_v1",
        "method": args.method_label or args.method,
        "adapter": args.method,
        "fold_root": str(args.fold_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "training_animals": len(train_samples),
        "test_animals": len(test_samples),
        "evaluated_test_animals": len(test_samples) - len(skipped_test_animals),
        "test_to_template_pairs": len(test_samples) - len(skipped_test_animals),
        "directed_test_test_pairs": 0,
        "skipped_test_animals": skipped_test_animals,
        "training_identity_vocabulary_size": len(vocabulary or ()),
        "template_selection": {
            "selection_split": "outer_train_only",
            "uses_validation": False,
            "uses_test": False,
            "uses_identity_labels": False,
            "geometry_normalization": "per-animal median centering, then divide XYZ by 200",
            "distance": "symmetric mean nearest-neighbour Euclidean distance",
            "criterion": "minimum mean distance to all other outer-training animals",
            "tie_break": "lexicographic worm UID, then original train-list index",
            "template_uid": medoid.uid,
            "template_path": str(Path(template_sample.source_path).resolve()),
            "template_train_index": int(medoid.index),
            "template_mean_distance": medoid.mean_distance,
            "training_candidates": [
                {
                    "uid": sample.uid,
                    "path": str(Path(sample.source_path).resolve()),
                    "mean_geometry_distance": medoid.mean_distances[index],
                }
                for index, sample in enumerate(train_samples)
            ],
        },
        "pair_score_normalization": args.normalization,
        "primary_aggregation": "template_score",
        "tie_policy": "expected metric under uniform tie breaking",
        "metrics": pooled,
        "per_animal": animal_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if vocabulary_support is None:
        raise AssertionError("Evaluation produced no vocabulary support counts")
    (args.output_dir / "training_vocabulary.json").write_text(
        json.dumps(
            [
                {"identity": identity, "support_templates": int(support)}
                for identity, support in zip(vocabulary or (), vocabulary_support.tolist())
            ],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "per_query.csv", all_rows)
    print(json.dumps({key: value for key, value in report.items() if key != "per_animal"}, indent=2))


if __name__ == "__main__":
    main()
