#!/usr/bin/env python3
"""Export locked biological splits from Candidate A/B without confusing seed-specific embeddings with worms.

Key rule
--------
The saved train/val/test lists may point to seed-specific embedding NPZ files.
Therefore this script NEVER compares those list-file paths across seeds.
Instead it resolves every entry back to its biological worm (`worm_id`) and
original source NPZ (`source_path`) and audits the biological split.

Outputs
-------
benchmark_official/protocols/<dataset>/fold_<k>/
    protocol.json
    test.txt                         # shared outer-test raw NPZs
    seed_1/{train,val,test}.txt      # raw NPZs
    seed_42/{train,val,test}.txt
    seed_123/{train,val,test}.txt

If train/val biological splits are identical across seeds, convenience copies
train.txt and val.txt are also written at the fold root. If not, they are not
silently merged.

No split is regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
RUN_ROOT = ROOT / "runs" / "fold_pure_table1init_candidates_ab_20260817"
OUT_ROOT = ROOT / "benchmark_official" / "protocols"

DATASETS = ("atanas", "rld")
FOLDS = (1, 2, 3, 4, 5)
SEEDS = (1, 42, 123)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def saved_args(cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    if isinstance(cfg.get("args"), dict):
        return dict(cfg["args"])
    if any(k in cfg for k in ("train_list", "val_list", "test_list")):
        return dict(cfg)
    raise KeyError(
        f"{cfg_path}: cannot find saved args; top-level keys={sorted(cfg.keys())}"
    )


def read_list(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    out: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        x = line.strip()
        if not x or x.startswith("#"):
            continue
        p = Path(x)
        if not p.is_absolute():
            p = (path.parent / p).resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"List entry does not exist: {p} (from {path})")
        out.append(p)
    if not out:
        raise RuntimeError(f"Empty list: {path}")
    return out


def list_path_from_cfg(cfg: dict[str, Any], cfg_path: Path, key: str) -> Path:
    args = saved_args(cfg, cfg_path)
    raw = args.get(key)
    if raw in (None, ""):
        raise KeyError(
            f"{cfg_path}: saved args missing {key}; keys={sorted(args.keys())}"
        )
    p = Path(str(raw))
    if p.is_absolute():
        p = p.resolve()
    else:
        candidates = [
            (ROOT / p).resolve(),
            (cfg_path.parent / p).resolve(),
        ]
        p = next((q for q in candidates if q.is_file()), candidates[0])
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def scalar_text(d: Any, key: str) -> str | None:
    if key not in d.files:
        return None
    arr = np.asarray(d[key]).reshape(-1)
    if len(arr) == 0:
        return None
    x = arr[0]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    s = str(x).strip()
    return s or None


def resolve_path_text(text: str, entry: Path) -> Path | None:
    p = Path(text)
    candidates = (
        [p.resolve()] if p.is_absolute()
        else [
            (ROOT / p).resolve(),
            (entry.parent / p).resolve(),
        ]
    )
    for q in candidates:
        if q.is_file():
            return q
    return None


def biological_record(entry: Path) -> dict[str, str]:
    """Resolve an embedding/raw NPZ to biological worm ID and raw source NPZ."""
    with np.load(entry, allow_pickle=True) as d:
        keys = set(d.files)

        explicit_worm_id = (
            scalar_text(d, "worm_id")
            or scalar_text(d, "worm_name")
            or scalar_text(d, "name")
        )
        source_text = scalar_text(d, "source_path")

        # A raw unified NPZ can be used directly.
        looks_raw = (
            "xyz" in keys
            and ("cell_id" in keys or "labels" in keys)
            and ("activity_raw" in keys or "clean_mask" in keys or "cell_id" in keys)
        )

    source: Path | None = None
    if source_text:
        source = resolve_path_text(source_text, entry)

    if source is None and looks_raw:
        source = entry.resolve()

    if source is None:
        raise RuntimeError(
            f"Cannot resolve biological source NPZ for:\n  {entry}\n"
            "Expected an existing `source_path` inside the NPZ or a raw unified "
            "NPZ containing xyz/cell_id."
        )

    # Prefer explicit worm_id saved by the embedding extractor.
    worm_id = explicit_worm_id
    if not worm_id:
        try:
            with np.load(source, allow_pickle=True) as d:
                worm_id = (
                    scalar_text(d, "worm_id")
                    or scalar_text(d, "worm_name")
                    or scalar_text(d, "name")
                )
        except Exception:
            worm_id = None
    if not worm_id:
        worm_id = source.stem

    return {
        "worm_id": str(worm_id),
        "source_path": str(source.resolve()),
        "listed_path": str(entry.resolve()),
    }


def canonical_split(entries: list[Path]) -> list[dict[str, str]]:
    rows = [biological_record(p) for p in entries]

    ids = [r["worm_id"] for r in rows]
    if len(ids) != len(set(ids)):
        dup = sorted({x for x in ids if ids.count(x) > 1})
        raise RuntimeError(f"Duplicate biological worm IDs in one split: {dup}")

    sources = [r["source_path"] for r in rows]
    if len(sources) != len(set(sources)):
        raise RuntimeError("Duplicate biological source NPZ in one split")

    # Canonical ordering removes irrelevant list ordering differences.
    return sorted(rows, key=lambda r: (r["worm_id"], r["source_path"]))


def biological_signature(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [(r["worm_id"], r["source_path"]) for r in rows]


def digest(rows: list[dict[str, str]]) -> str:
    payload = "\n".join(
        f"{r['worm_id']}\t{r['source_path']}" for r in rows
    ) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_dir(dataset: str, fold: int, seed: int, candidate: str) -> Path:
    return (
        RUN_ROOT
        / "folds" / dataset / f"fold_{fold}"
        / "pipeline" / f"seed_{seed}" / candidate
    )


def load_candidate_split(
    dataset: str,
    fold: int,
    seed: int,
    candidate: str,
) -> dict[str, Any]:
    cfg_path = run_dir(dataset, fold, seed, candidate) / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    cfg = read_json(cfg_path)

    splits: dict[str, list[dict[str, str]]] = {}
    list_files: dict[str, str] = {}
    for split in ("train", "val", "test"):
        lp = list_path_from_cfg(cfg, cfg_path, f"{split}_list")
        splits[split] = canonical_split(read_list(lp))
        list_files[split] = str(lp.resolve())

    return {
        "config": str(cfg_path.resolve()),
        "splits": splits,
        "list_files": list_files,
    }


def same_biology(a: list[dict[str, str]], b: list[dict[str, str]]) -> bool:
    return biological_signature(a) == biological_signature(b)


def write_paths(path: Path, rows: list[dict[str, str]], force: bool) -> None:
    text = "\n".join(r["source_path"] for r in rows) + "\n"
    if path.exists() and not force:
        old = path.read_text(encoding="utf-8")
        if old != text:
            raise RuntimeError(
                f"Refusing to silently overwrite changed protocol: {path}\n"
                "Inspect first, or rerun with --force after audit."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_disjoint(seed: int, splits: dict[str, list[dict[str, str]]]) -> None:
    sets = {
        k: {r["worm_id"] for r in v}
        for k, v in splits.items()
    }
    overlaps = {
        "train_val": sets["train"] & sets["val"],
        "train_test": sets["train"] & sets["test"],
        "val_test": sets["val"] & sets["test"],
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise RuntimeError(f"Seed {seed}: biological split leakage: {bad}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--folds", nargs="+", type=int, choices=FOLDS, default=list(FOLDS))
    args = p.parse_args()

    manifest: dict[str, Any] = {
        "source_run_root": str(RUN_ROOT.resolve()),
        "comparison_unit": "biological worm_id + resolved source_path, not embedding path",
        "datasets": {},
    }

    for dataset in args.datasets:
        manifest["datasets"][dataset] = {}

        for fold in args.folds:
            per_seed: dict[int, dict[str, Any]] = {}

            for seed in SEEDS:
                cand_a = load_candidate_split(dataset, fold, seed, "candidate_a")
                cand_b = load_candidate_split(dataset, fold, seed, "candidate_b")

                # A/B must represent the same biological split within a seed.
                for split in ("train", "val", "test"):
                    if not same_biology(
                        cand_a["splits"][split],
                        cand_b["splits"][split],
                    ):
                        raise RuntimeError(
                            f"Candidate A/B biological mismatch: "
                            f"dataset={dataset} fold={fold} seed={seed} split={split}"
                        )

                ensure_disjoint(seed, cand_b["splits"])
                per_seed[seed] = cand_b

            # The defining property of outer CV: held-out test biology must be the
            # same across stochastic training seeds.
            reference_test = per_seed[SEEDS[0]]["splits"]["test"]
            for seed in SEEDS[1:]:
                if not same_biology(
                    per_seed[seed]["splits"]["test"],
                    reference_test,
                ):
                    raise RuntimeError(
                        f"OUTER TEST biological mismatch across seeds: "
                        f"dataset={dataset} fold={fold} seed={seed}"
                    )

            train_same = all(
                same_biology(
                    per_seed[s]["splits"]["train"],
                    per_seed[SEEDS[0]]["splits"]["train"],
                )
                for s in SEEDS[1:]
            )
            val_same = all(
                same_biology(
                    per_seed[s]["splits"]["val"],
                    per_seed[SEEDS[0]]["splits"]["val"],
                )
                for s in SEEDS[1:]
            )

            out = OUT_ROOT / dataset / f"fold_{fold}"
            out.mkdir(parents=True, exist_ok=True)

            # Seed-specific raw biological protocols are always written.
            seed_blocks = {}
            for seed in SEEDS:
                seed_out = out / f"seed_{seed}"
                splits = per_seed[seed]["splits"]
                for split in ("train", "val", "test"):
                    write_paths(
                        seed_out / f"{split}.txt",
                        splits[split],
                        args.force,
                    )

                seed_blocks[str(seed)] = {
                    "train_worms": len(splits["train"]),
                    "val_worms": len(splits["val"]),
                    "test_worms": len(splits["test"]),
                    "train_sha256": digest(splits["train"]),
                    "val_sha256": digest(splits["val"]),
                    "test_sha256": digest(splits["test"]),
                    "source_embedding_lists": per_seed[seed]["list_files"],
                    "worm_ids": {
                        k: [r["worm_id"] for r in splits[k]]
                        for k in ("train", "val", "test")
                    },
                }

            # Shared outer test convenience file for deterministic pretrained
            # baselines such as fDNC official pretrained.
            write_paths(out / "test.txt", reference_test, args.force)

            # Only expose shared train/val convenience files when biology is
            # actually identical. Otherwise learned baselines must use seed_*.
            if train_same:
                write_paths(
                    out / "train.txt",
                    per_seed[SEEDS[0]]["splits"]["train"],
                    args.force,
                )
            elif (out / "train.txt").exists():
                (out / "train.txt").unlink()

            if val_same:
                write_paths(
                    out / "val.txt",
                    per_seed[SEEDS[0]]["splits"]["val"],
                    args.force,
                )
            elif (out / "val.txt").exists():
                (out / "val.txt").unlink()

            block = {
                "dataset": dataset,
                "fold": fold,
                "candidate_a_b_biology_identical_within_seed": True,
                "outer_test_biology_identical_across_seeds": True,
                "train_biology_identical_across_seeds": train_same,
                "val_biology_identical_across_seeds": val_same,
                "seeds": seed_blocks,
                "shared_test_worms": len(reference_test),
                "shared_test_sha256": digest(reference_test),
                "primary_metric": "direct per-query ranking Top-1 before Hungarian",
                "policy": (
                    "deterministic pretrained baselines may use fold/test.txt; "
                    "learned baselines use fold/seed_<seed>/train|val|test.txt"
                ),
            }

            (out / "protocol.json").write_text(
                json.dumps(block, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["datasets"][dataset][f"fold_{fold}"] = block

            print(
                f"[OK] {dataset:6s} fold={fold} | "
                f"outer_test_same=True | "
                f"train_same={train_same} | val_same={val_same} | "
                f"test_worms={len(reference_test)}"
            )

    manifest_path = OUT_ROOT / "locked_protocol_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nBIOLOGICAL PROTOCOL AUDIT PASSED")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
