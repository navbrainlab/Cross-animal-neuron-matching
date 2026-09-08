from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat, savemat

DATA_ROOT = Path(
    "nuclr/Data/Atanas_SF_unified_000776/cv5_grouped_v1"
)
ATLAS_PATH = Path(
    "CRF_Cell_ID/sample_run/"
    "data_neuron_relationship_annotation_updated.mat"
)
RUN_ROOT = Path(
    "runs/crfid_official_atanas_cv5_v1"
)


def mstr(x):
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, np.str_):
        return str(x).strip()
    if isinstance(x, bytes):
        return x.decode().strip()
    if isinstance(x, np.ndarray):
        x = np.squeeze(x)
        if x.ndim == 0:
            return mstr(x.item())
        if x.dtype.kind in ("U", "S"):
            return "".join(str(v) for v in x.reshape(-1)).strip()
        if x.size == 1:
            return mstr(x.item())
    return str(x).strip()


def canonical_id(x):
    # Only remove uncertainty suffix.
    # Never map family/unsided labels to L/R.
    return str(x).strip().rstrip("?")


def normalize_cloud(x):
    x = np.asarray(x, dtype=np.float64)
    c = x.mean(axis=0)
    y = x - c

    rms = np.sqrt(
        np.mean(np.sum(y * y, axis=1))
    )

    if rms < 1e-12:
        raise RuntimeError("Degenerate coordinate cloud")

    return y / rms


def load_atlas():
    s = loadmat(
        ATLAS_PATH,
        squeeze_me=True,
        struct_as_record=False,
    )

    names = [
        mstr(v)
        for v in np.asarray(
            s["Neuron_head"]
        ).reshape(-1)
    ]

    xyz = np.column_stack([
        np.asarray(s["X_rot"], dtype=float).reshape(-1),
        np.asarray(s["Y_rot"], dtype=float).reshape(-1),
        np.asarray(s["Z_rot"], dtype=float).reshape(-1),
    ])

    assert xyz.shape == (len(names), 3)

    xyz = normalize_cloud(xyz)

    return names, xyz


def collect_train_centroids(fold):
    train_dir = DATA_ROOT / f"fold_{fold}" / "train"
    files = sorted(train_dir.glob("*.npz"))

    assert files, train_dir

    obs = {}

    for p in files:
        z = np.load(p, allow_pickle=True)

        xyz = np.asarray(z["xyz"], dtype=float)
        ids = np.asarray(z["cell_id"]).astype(str)
        labeled = np.asarray(
            z["labeled_mask"], dtype=bool
        )
        valid = np.asarray(
            z["valid_xyz_mask"], dtype=bool
        )

        finite = np.isfinite(xyz).all(axis=1)
        zero = np.all(
            np.isclose(xyz, 0.0), axis=1
        )

        geom = valid & finite & (~zero)

        if geom.sum() < 3:
            continue

        xnorm = normalize_cloud(xyz[geom])

        orig = np.where(geom)[0]

        old_to_new = {
            int(old): int(new)
            for new, old in enumerate(orig)
        }

        for old_i in np.where(labeled & geom)[0]:
            name = canonical_id(ids[old_i])

            if not name:
                continue

            new_i = old_to_new[int(old_i)]

            obs.setdefault(name, []).append(
                xnorm[new_i]
            )

    centroids = {
        name: np.median(
            np.asarray(points),
            axis=0,
        )
        for name, points in obs.items()
    }

    return centroids, files


def fit_frame(fold, atlas_names, atlas_xyz):
    atlas_lookup = {
        name: atlas_xyz[i]
        for i, name in enumerate(atlas_names)
    }

    train_centroids, train_files = \
        collect_train_centroids(fold)

    shared = sorted(
        set(train_centroids) &
        set(atlas_lookup)
    )

    if len(shared) < 20:
        raise RuntimeError(
            f"fold{fold}: only {len(shared)} shared IDs"
        )

    X = np.stack([
        train_centroids[x]
        for x in shared
    ])

    Y = np.stack([
        atlas_lookup[x]
        for x in shared
    ])

    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    X /= np.sqrt(
        np.mean(np.sum(X * X, axis=1))
    )

    Y /= np.sqrt(
        np.mean(np.sum(Y * Y, axis=1))
    )

    M = X.T @ Y
    U, S, Vt = np.linalg.svd(M)

    # O(3) orthogonal Procrustes.
    # Reflection is permitted; no test labels involved.
    R = U @ Vt

    pred = X @ R
    err = np.linalg.norm(pred - Y, axis=1)

    rmse = np.sqrt(
        np.mean(
            np.sum((pred - Y) ** 2, axis=1)
        )
    )

    out = RUN_ROOT / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "R.npy", R)

    audit = {
        "fold": fold,
        "train_files": len(train_files),
        "train_identities": len(train_centroids),
        "official_states": len(atlas_names),
        "shared_identities": len(shared),
        "shared_identity_names": shared,
        "det_R": float(np.linalg.det(R)),
        "alignment_rmse": float(rmse),
        "median_identity_error": float(np.median(err)),
        "frame_source": "outer_fold_train_only",
        "test_labels_used_for_frame": False,
    }

    with open(out / "frame_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print(
        f"fold{fold}: "
        f"train={len(train_files)} "
        f"shared={len(shared)} "
        f"detR={np.linalg.det(R):+.4f} "
        f"rmse={rmse:.4f}"
    )

    return R


def prepare_test(fold, R):
    test_dir = DATA_ROOT / f"fold_{fold}" / "test"
    files = sorted(test_dir.glob("*.npz"))

    assert files, test_dir

    outdir = RUN_ROOT / f"fold{fold}" / "test"
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for wi, p in enumerate(files):
        z = np.load(p, allow_pickle=True)

        xyz = np.asarray(z["xyz"], dtype=float)
        valid = np.asarray(
            z["valid_xyz_mask"], dtype=bool
        )
        labeled = np.asarray(
            z["labeled_mask"], dtype=bool
        )

        finite = np.isfinite(xyz).all(axis=1)
        zero = np.all(
            np.isclose(xyz, 0.0), axis=1
        )

        geom = valid & finite & (~zero)

        # Never silently drop supervised queries.
        bad = labeled & (~geom)
        if bad.any():
            raise RuntimeError(
                f"{p}: {bad.sum()} labeled neurons "
                f"lack valid geometry"
            )

        orig_idx = np.where(geom)[0]

        X = normalize_cloud(xyz[geom])

        # Frozen fold-train frame.
        X = X @ R

        tag = f"worm_{wi:03d}"

        input_path = outdir / f"{tag}_input.mat"
        sidecar_path = outdir / f"{tag}_sidecar.npz"
        output_path = outdir / f"{tag}_output.mat"

        # Inference input contains coordinates only.
        savemat(
            input_path,
            {"mu_marker": X},
        )

        np.savez(
            sidecar_path,
            source=str(p.resolve()),
            orig_idx=orig_idx,
        )

        manifest.append({
            "index": wi,
            "source": str(p.resolve()),
            "input": str(input_path.resolve()),
            "sidecar": str(sidecar_path.resolve()),
            "output": str(output_path.resolve()),
            "n_nodes": int(len(orig_idx)),
            "n_queries": int(labeled.sum()),
        })

    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"fold{fold}: prepared "
        f"{len(manifest)} test worms"
    )


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    atlas_names, atlas_xyz = load_atlas()

    print("=" * 90)
    print("CRF_ID — ATANAS CV5 PREPARATION")
    print("=" * 90)
    print("official states =", len(atlas_names))

    for fold in range(5):
        R = fit_frame(
            fold,
            atlas_names,
            atlas_xyz,
        )
        prepare_test(fold, R)

    print("=" * 90)
    print("PREPARATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
