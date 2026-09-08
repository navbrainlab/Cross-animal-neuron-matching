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
    # Only uncertainty marker normalization.
    # No ADA -> ADAL/ADAR etc.
    return str(x).strip().rstrip("?")


def normalize_cloud(x):
    """
    Translation and isotropic scale removal only.
    No rotation is estimated from test identities.
    """
    x = np.asarray(x, dtype=np.float64)
    c = x.mean(axis=0)
    y = x - c

    rms = np.sqrt(np.mean(np.sum(y * y, axis=1)))
    if rms < 1e-12:
        rms = 1.0

    return y / rms, c, rms


def load_official_atlas():
    s = loadmat(
        ATLAS_PATH,
        squeeze_me=True,
        struct_as_record=False
    )

    names = [
        mstr(v)
        for v in np.asarray(s["Neuron_head"]).reshape(-1)
    ]

    X = np.asarray(s["X_rot"], dtype=float).reshape(-1)
    Y = np.asarray(s["Y_rot"], dtype=float).reshape(-1)
    Z = np.asarray(s["Z_rot"], dtype=float).reshape(-1)

    xyz = np.column_stack([X, Y, Z])

    assert xyz.shape[0] == len(names)

    xyz, _, _ = normalize_cloud(xyz)

    return names, xyz


def collect_train_identity_centroids(fold):
    train_dir = DATA_ROOT / f"fold_{fold}" / "train"

    files = sorted(train_dir.glob("*.npz"))
    assert files, train_dir

    # identity -> normalized coordinates across train worms
    observations = {}

    used_worms = 0

    for p in files:
        z = np.load(p, allow_pickle=True)

        xyz = np.asarray(z["xyz"], dtype=float)
        ids = np.asarray(z["cell_id"]).astype(str)
        labeled = np.asarray(
            z["labeled_mask"],
            dtype=bool
        )
        valid = np.asarray(
            z["valid_xyz_mask"],
            dtype=bool
        )

        finite = np.isfinite(xyz).all(axis=1)
        zero = np.all(np.isclose(xyz, 0.0), axis=1)

        geom = valid & finite & (~zero)

        # Coordinate-frame normalization must not use labels.
        xyz_norm, _, _ = normalize_cloud(xyz[geom])

        original_indices = np.where(geom)[0]
        old_to_new = {
            old: new
            for new, old in enumerate(original_indices)
        }

        count_this_worm = 0

        for old_i in np.where(labeled & geom)[0]:
            name = canonical_id(ids[old_i])

            if not name:
                continue

            new_i = old_to_new[int(old_i)]

            observations.setdefault(
                name, []
            ).append(xyz_norm[new_i])

            count_this_worm += 1

        if count_this_worm:
            used_worms += 1

    centroids = {}

    for name, pts in observations.items():
        pts = np.asarray(pts, dtype=float)

        # Median is more robust than mean across worms.
        centroids[name] = np.median(
            pts,
            axis=0
        )

    return centroids, files, used_worms


def fit_orthogonal(X, Y):
    """
    Find R minimizing ||X R - Y||_F.

    O(3), i.e. reflection is allowed because microscopy
    coordinate systems may have opposite handedness.
    """
    M = X.T @ Y
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt

    return R, S


def main(fold=0):
    atlas_names, atlas_xyz = load_official_atlas()

    atlas_lookup = {
        name: atlas_xyz[i]
        for i, name in enumerate(atlas_names)
    }

    train_centroids, train_files, n_worms = \
        collect_train_identity_centroids(fold)

    shared = sorted(
        set(train_centroids) &
        set(atlas_lookup)
    )

    print("=" * 90)
    print("CRF_ID TRAIN-ONLY FRAME CALIBRATION")
    print("=" * 90)
    print("fold                 =", fold)
    print("train files          =", len(train_files))
    print("train worms used     =", n_worms)
    print("train identities     =", len(train_centroids))
    print("official states      =", len(atlas_names))
    print("shared identities    =", len(shared))

    assert len(shared) >= 20, (
        "Too few shared identities"
    )

    X = np.stack([
        train_centroids[name]
        for name in shared
    ])

    Y = np.stack([
        atlas_lookup[name]
        for name in shared
    ])

    # Recenter identity centroids.
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Equal global scale.
    X /= np.sqrt(
        np.mean(np.sum(X * X, axis=1))
    )
    Y /= np.sqrt(
        np.mean(np.sum(Y * Y, axis=1))
    )

    R, singular_values = fit_orthogonal(X, Y)

    pred = X @ R

    per_id_error = np.linalg.norm(
        pred - Y,
        axis=1
    )

    rmse = np.sqrt(
        np.mean(
            np.sum((pred - Y) ** 2, axis=1)
        )
    )

    print()
    print("R =")
    print(R)

    print()
    print("det(R)              =", np.linalg.det(R))
    print("singular values     =", singular_values)
    print("train alignment RMSE=", rmse)
    print(
        "median identity err =",
        np.median(per_id_error)
    )

    outdir = (
        RUN_ROOT /
        f"fold{fold}" /
        "frame_calibration"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        outdir / "R.npy",
        R
    )

    np.savez(
        outdir / "calibration.npz",
        R=R,
        shared=np.asarray(shared),
        per_identity_error=per_id_error,
        rmse=rmse,
        determinant=np.linalg.det(R),
    )

    with open(
        outdir / "audit.json",
        "w"
    ) as f:
        json.dump(
            {
                "fold": fold,
                "n_train_files": len(train_files),
                "n_train_worms_used": n_worms,
                "n_train_identities": len(train_centroids),
                "n_official_states": len(atlas_names),
                "n_shared_identities": len(shared),
                "shared_identities": shared,
                "determinant": float(
                    np.linalg.det(R)
                ),
                "rmse": float(rmse),
                "median_identity_error": float(
                    np.median(per_id_error)
                ),
                "test_labels_used": False,
            },
            f,
            indent=2
        )

    print()
    print(
        "saved:",
        outdir / "R.npy"
    )


if __name__ == "__main__":
    main(0)
