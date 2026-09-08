#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
SOURCE = ROOT / "Data/Dunn_001623/cv5_grouped_v1"
OUT = ROOT / "runs/rld_robustness_cv5_seed42_v2/corruptions"
SOURCES = {
    "atanas": ROOT / "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
    "rld": SOURCE,
}

GRIDS = {
    "coord_noise": (0.00, 0.02, 0.05, 0.10, 0.20),
    "activity_noise": (0.00, 0.10, 0.20, 0.50, 1.00, 2.00),
    "missing": (0.00, 0.10, 0.20, 0.30, 0.40, 0.50),
    "outlier": (0.00, 0.10, 0.20, 0.30, 0.40, 0.50),
}
PSEEDS = (0, 1, 2)

NON_NEURON_KEYS = {
    "timestamps", "time", "times", "grid_spacing", "sampling_rate_hz",
    "source_fs", "recording_uid", "worm_id", "source_path", "source_nwb",
    "date", "session", "metadata",
}

# These masks control whether a row is valid model input.  Synthetic
# distractors must be valid population rows even though they are never valid
# supervised identity targets.  Treating every ``*mask`` field alike used to
# set ``valid_xyz_mask=False`` for inserted rows; the shared WormCache then
# removed every distractor before CPD/fDNC inference.
INPUT_VALIDITY_MASKS = {"valid_xyz_mask"}
SUPERVISION_MASKS = {
    "labeled_mask", "certain_mask", "clean_mask", "supervised_mask",
    "supervised", "clean",
}

def stable_seed(*parts) -> int:
    s = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "little") % (2**32)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def sha256_array(value) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()

def uid(path: Path, payload):
    if "recording_uid" in payload:
        x = np.asarray(payload["recording_uid"]).reshape(-1)
        if len(x):
            return str(x[0])
    return path.stem

def neuron_axis(key, arr, n):
    return key not in NON_NEURON_KEYS and arr.ndim >= 1 and arr.shape[0] == n

def median_pairwise(xyz):
    if len(xyz) <= 1:
        return 1.0
    i, j = np.triu_indices(len(xyz), 1)
    d = np.sqrt(((xyz[i].astype(np.float64) - xyz[j].astype(np.float64)) ** 2).sum(1))
    return float(np.median(d)) if len(d) else 1.0

def transform(path: Path, kind: str, severity: float, pseed: int):
    with np.load(path, allow_pickle=True) as z:
        p = {k: np.asarray(z[k]) for k in z.files}
    xyz = np.asarray(p["xyz"], dtype=np.float32)[:, :3]
    n = len(xyz)
    if severity <= 0:
        return p

    rng = np.random.default_rng(stable_seed(kind, f"{severity:.8f}", pseed, uid(path, p)))

    if kind == "coord_noise":
        dmed = median_pairwise(xyz)
        noise = rng.normal(0, severity * max(dmed, 1e-6), size=xyz.shape).astype(np.float32)
        out = dict(p)
        x = np.asarray(p["xyz"]).copy()
        x[:, :3] = x[:, :3] + noise.astype(x.dtype, copy=False)
        out["xyz"] = x
        return out

    if kind == "activity_noise":
        if "activity_raw" not in p:
            raise KeyError(f"{path}: activity_raw is required for activity noise")
        activity = np.asarray(p["activity_raw"])
        if activity.ndim != 2:
            raise ValueError(f"{path}: activity_raw must be two-dimensional")
        neuron_first = activity.shape[0] == n
        if not neuron_first and activity.shape[1] != n:
            raise ValueError(f"{path}: activity_raw has no neuron axis of length {n}")
        work = np.asarray(activity if neuron_first else activity.T, dtype=np.float64)
        scale = work.std(axis=1, ddof=0, keepdims=True)
        noise = rng.normal(0.0, severity * np.maximum(scale, 1e-12), size=work.shape)
        noisy = work + noise
        out = dict(p)
        out["activity_raw"] = np.asarray(
            noisy if neuron_first else noisy.T, dtype=activity.dtype
        )
        return out

    if kind == "missing":
        keep_n = max(2, int(round(n * (1.0 - severity))))
        keep = np.sort(rng.choice(n, size=keep_n, replace=False))
        return {
            k: (np.asarray(v)[keep] if neuron_axis(k, np.asarray(v), n) else np.asarray(v))
            for k, v in p.items()
        }

    if kind == "outlier":
        add_n = max(1, int(round(n * severity)))
        dmed = median_pairwise(xyz)
        lo, hi = xyz.min(0), xyz.max(0)
        pad = 0.25 * max(dmed, 1e-6)
        new_xyz = rng.uniform(lo - pad, hi + pad, size=(add_n, 3)).astype(np.float32)
        base_idx = rng.integers(0, n, size=add_n)

        activity_key = "activity_raw" if "activity_raw" in p else ("activity" if "activity" in p else None)
        new_act = None
        if activity_key:
            a = np.asarray(p[activity_key], dtype=np.float32)
            base = a[base_idx]
            std = a.std(0, ddof=0)
            new_act = base + rng.normal(
                0, 0.25 * np.maximum(std, 1e-6), size=base.shape
            ).astype(np.float32)

        out = {}
        labels = [f"__OUTLIER_s{pseed}_{i:04d}" for i in range(add_n)]
        for k, v in p.items():
            a = np.asarray(v)
            if not neuron_axis(k, a, n):
                out[k] = a
                continue
            kl = k.lower()
            if k == "xyz":
                extra = new_xyz.astype(a.dtype, copy=False)
            elif k == activity_key and new_act is not None:
                extra = new_act.astype(a.dtype, copy=False)
            elif kl in {"cell_id", "cell_id_alt", "labels", "label"}:
                width = max([len(str(x)) for x in a.reshape(-1).tolist()] + [len(x) for x in labels] + [1])
                dt = f"<U{width}"
                out[k] = np.concatenate([a.astype(dt), np.asarray(labels, dtype=dt)])
                continue
            elif kl in INPUT_VALIDITY_MASKS:
                extra = np.ones((add_n,) + a.shape[1:], dtype=a.dtype)
            elif kl in SUPERVISION_MASKS or "supervision" in kl:
                extra = np.zeros((add_n,) + a.shape[1:], dtype=a.dtype)
            elif kl in {"roi_index", "aligned_table_id"}:
                extra = np.full((add_n,) + a.shape[1:], -1, dtype=a.dtype)
            else:
                extra = a[base_idx].copy()
            out[k] = np.concatenate([a, extra], axis=0)
        return out

    raise ValueError(kind)

def file_manifest(source: Path, output: Path, kind: str, severity: float, pseed: int):
    """Describe and hash the exact materialized corruption for one worm."""
    with np.load(source, allow_pickle=True) as z:
        src = {k: np.asarray(z[k]) for k in z.files}
    with np.load(output, allow_pickle=True) as z:
        dst = {k: np.asarray(z[k]) for k in z.files}

    src_xyz = np.asarray(src["xyz"])[:, :3]
    dst_xyz = np.asarray(dst["xyz"])[:, :3]
    n_src, n_dst = len(src_xyz), len(dst_xyz)
    sample_uid = uid(source, src)
    kept = np.arange(n_src, dtype=np.int64)
    deleted = np.empty(0, dtype=np.int64)

    if kind == "missing" and severity > 0:
        rng = np.random.default_rng(
            stable_seed(kind, f"{severity:.8f}", pseed, sample_uid)
        )
        keep_n = max(2, int(round(n_src * (1.0 - severity))))
        kept = np.sort(rng.choice(n_src, size=keep_n, replace=False)).astype(np.int64)
        deleted = np.setdiff1d(np.arange(n_src, dtype=np.int64), kept)
        if n_dst != len(kept) or not np.array_equal(dst_xyz, src_xyz[kept]):
            raise RuntimeError(f"{output}: materialized missing mask does not match plan")

    labeled = np.asarray(dst.get("labeled_mask", np.ones(n_dst)), dtype=bool)
    certain = np.asarray(dst.get("certain_mask", np.ones(n_dst)), dtype=bool)
    clean = np.asarray(dst.get("clean_mask", np.ones(n_dst)), dtype=bool)
    valid_xyz = np.asarray(dst.get("valid_xyz_mask", np.ones(n_dst)), dtype=bool)
    labels = np.asarray(dst["cell_id"]).astype(str).reshape(-1)
    query_mask = labeled & certain & clean & valid_xyz
    query_rows = [
        {"row": int(i), "cell_id": str(labels[i])}
        for i in np.flatnonzero(query_mask)
        if str(labels[i]).strip() and not str(labels[i]).startswith("__OUTLIER_")
    ]

    record = {
        "recording_uid": sample_uid,
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(output),
        "source_neuron_rows": n_src,
        "output_neuron_rows": n_dst,
        "kept_source_indices": kept.tolist(),
        "deleted_source_indices": deleted.tolist(),
        "evaluable_query_rows_after_corruption": query_rows,
    }
    if kind == "coord_noise":
        if n_src != n_dst:
            raise RuntimeError(f"{output}: coordinate corruption changed row count")
        record["coordinate_delta_sha256"] = sha256_array(dst_xyz - src_xyz)
    elif kind == "activity_noise":
        if n_src != n_dst or not np.array_equal(dst_xyz, src_xyz):
            raise RuntimeError(f"{output}: activity noise changed geometry or row count")
        src_activity = np.asarray(src["activity_raw"])
        dst_activity = np.asarray(dst["activity_raw"])
        if src_activity.shape != dst_activity.shape:
            raise RuntimeError(f"{output}: activity noise changed activity shape")
        record["activity_delta_sha256"] = sha256_array(dst_activity - src_activity)
        record["activity_noise_scale"] = (
            "per-neuron temporal population SD; additive iid Gaussian sigma=severity*SD"
        )
    elif kind == "outlier":
        if n_dst < n_src:
            raise RuntimeError(f"{output}: outlier corruption reduced row count")
        record["distractor_xyz_sha256"] = sha256_array(dst_xyz[n_src:])
        record["distractor_labels"] = labels[n_src:].tolist()
        record["distractor_valid_xyz"] = valid_xyz[n_src:].tolist()
    return record

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("atanas", "rld"), default="rld")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--source-root", type=Path, default=None)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    ap.add_argument("--only-clean", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    source_root = (args.source_root or SOURCES[args.dataset]).resolve()
    output_root = (
        args.output_root
        or (
            OUT if args.dataset == "rld"
            else ROOT / f"runs/{args.dataset}_robustness_cv5_seed42_v1/corruptions"
        )
    ).resolve()
    protocol_id = f"{args.dataset.upper()}_CV5_seed42_controlled_robustness_v2_hashed"

    rows = []
    conditions = [("coord_noise", 0.0, 0)] if args.only_clean else [
        (kind, sev, ps)
        for kind, levels in GRIDS.items()
        for sev in levels
        for ps in ((0,) if sev == 0 else PSEEDS)
    ]

    for fold in folds:
        src = source_root / f"fold_{fold}"
        if not (src / "test").is_dir():
            raise FileNotFoundError(src / "test")
        test_files = sorted((src / "test").glob("*.npz"))
        if not test_files:
            raise RuntimeError(f"No test npz: {src}")

        for kind, sev, ps in conditions:
            name = f"{kind}_l{sev:.2f}_p{ps}"
            dst = output_root / f"fold{fold}" / name
            if dst.exists() and args.overwrite:
                shutil.rmtree(dst)
            (dst / "test").mkdir(parents=True, exist_ok=True)

            for split in ("train", "val"):
                link = dst / split
                if not link.exists():
                    link.symlink_to((src / split).resolve(), target_is_directory=True)

            if not any((dst / "test").glob("*.npz")):
                for f in test_files:
                    np.savez_compressed(dst / "test" / f.name, **transform(f, kind, sev, ps))

            files = [
                file_manifest(f, dst / "test" / f.name, kind, sev, ps)
                for f in test_files
            ]

            meta = {
                "protocol": protocol_id,
                "dataset": args.dataset,
                "fold": fold,
                "kind": kind,
                "severity": sev,
                "perturbation_seed": ps,
                "source_fold": str(src.resolve()),
                "train_val_are_original_symlinks": True,
                "test_is_materialized_corruption": True,
                "files": files,
            }
            (dst / "CORRUPTION.json").write_text(json.dumps(meta, indent=2) + "\n")
            rows.append({**meta, "root": str(dst.resolve())})
            print(f"[DONE] fold{fold} {name} test_worms={len(test_files)}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps({
        "protocol": protocol_id,
        "dataset": args.dataset,
        "manifest_contract": (
            "All methods must verify output_sha256, pass every valid input row "
            "through inference, and score exactly the listed evaluable query rows."
        ),
        "conditions": rows,
    }, indent=2) + "\n")
    (output_root / "MANIFEST.sha256").write_text(
        f"{sha256_file(manifest_path)}  MANIFEST.json\n", encoding="utf-8"
    )
    print("Saved:", manifest_path)

if __name__ == "__main__":
    main()
