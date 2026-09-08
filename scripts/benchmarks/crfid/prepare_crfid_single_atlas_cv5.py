from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat, savemat

DATA_ROOT = Path(
    "nuclr/Data/Atanas_SF_unified_000776/cv5_grouped_v1"
)

# 唯一 CRF_ID atlas
ATLAS_FILE = Path(
    "CRF_Cell_ID/sample_run/"
    "data_neuron_relationship_annotation_updated.mat"
)

RUN_ROOT = Path(
    "runs/crfid_official_singleatlas_atanas_cv5_v1"
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
            return "".join(
                str(v) for v in x.reshape(-1)
            ).strip()

        if x.size == 1:
            return mstr(x.item())

    return str(x).strip()


def canonical_id(x):
    # 只去除不确定性后缀
    # 不做 ADA -> ADAL/ADAR 之类人工映射
    return str(x).strip().rstrip("?")


def normalize_cloud(X):
    X = np.asarray(X, dtype=np.float64)

    X = X - X.mean(axis=0)

    rms = np.sqrt(
        np.mean(
            np.sum(X * X, axis=1)
        )
    )

    if rms < 1e-12:
        raise RuntimeError("Degenerate coordinate cloud")

    return X / rms


def load_official_atlas():
    S = loadmat(
        ATLAS_FILE,
        squeeze_me=True,
        struct_as_record=False,
    )

    names = [
        mstr(x)
        for x in np.asarray(
            S["Neuron_head"]
        ).reshape(-1)
    ]

    xyz = np.column_stack([
        np.asarray(S["X_rot"], dtype=float).reshape(-1),
        np.asarray(S["Y_rot"], dtype=float).reshape(-1),
        np.asarray(S["Z_rot"], dtype=float).reshape(-1),
    ])

    assert len(names) == 178
    assert xyz.shape == (178, 3)

    xyz = normalize_cloud(xyz)

    return names, xyz


def collect_train_identity_centroids(fold):
    """
    注意：
    这里只利用 train animals 来估计 coordinate frame。
    不构建 CRF atlas。
    """
    train_dir = DATA_ROOT / f"fold_{fold}" / "train"

    files = sorted(
        train_dir.glob("*.npz")
    )

    assert files

    obs = {}

    for p in files:
        z = np.load(
            p,
            allow_pickle=True,
        )

        xyz = np.asarray(
            z["xyz"],
            dtype=float,
        )

        cell_id = np.asarray(
            z["cell_id"]
        ).astype(str)

        labeled = np.asarray(
            z["labeled_mask"],
            dtype=bool,
        )

        valid = np.asarray(
            z["valid_xyz_mask"],
            dtype=bool,
        )

        finite = np.isfinite(
            xyz
        ).all(axis=1)

        nonzero = ~np.all(
            np.isclose(xyz, 0.0),
            axis=1,
        )

        geom = (
            valid &
            finite &
            nonzero
        )

        if geom.sum() < 3:
            continue

        X = normalize_cloud(
            xyz[geom]
        )

        orig = np.where(
            geom
        )[0]

        index_map = {
            int(old): int(new)
            for new, old in enumerate(orig)
        }

        for old_i in np.where(
            labeled & geom
        )[0]:

            name = canonical_id(
                cell_id[old_i]
            )

            if not name:
                continue

            new_i = index_map[
                int(old_i)
            ]

            obs.setdefault(
                name, []
            ).append(
                X[new_i]
            )

    centroids = {
        name: np.median(
            np.asarray(points),
            axis=0,
        )
        for name, points
        in obs.items()
    }

    return centroids


def fit_train_only_frame(
    fold,
    atlas_names,
    atlas_xyz,
):
    """
    Fit a SINGLE fold-level O(3) transformation.

    train identities -> official CRF atlas

    No test identities used.
    No Atanas atlas constructed.
    """

    train_centroids = (
        collect_train_identity_centroids(
            fold
        )
    )

    atlas_lookup = {
        name: atlas_xyz[i]
        for i, name
        in enumerate(atlas_names)
    }

    shared = sorted(
        set(train_centroids) &
        set(atlas_lookup)
    )

    assert len(shared) >= 20

    X = np.stack([
        train_centroids[name]
        for name in shared
    ])

    Y = np.stack([
        atlas_lookup[name]
        for name in shared
    ])

    X = normalize_cloud(X)
    Y = normalize_cloud(Y)

    # Orthogonal Procrustes.
    # reflection is allowed because microscope
    # coordinate handedness may differ.
    U, S, Vt = np.linalg.svd(
        X.T @ Y
    )

    R = U @ Vt

    pred = X @ R

    rmse = np.sqrt(
        np.mean(
            np.sum(
                (pred - Y) ** 2,
                axis=1,
            )
        )
    )

    print(
        f"fold{fold}: "
        f"shared={len(shared)} "
        f"det(R)={np.linalg.det(R):+.6f} "
        f"rmse={rmse:.6f}"
    )

    return R, shared, rmse


def prepare_fold(
    fold,
    R,
):
    test_dir = (
        DATA_ROOT /
        f"fold_{fold}" /
        "test"
    )

    files = sorted(
        test_dir.glob("*.npz")
    )

    out_dir = (
        RUN_ROOT /
        f"fold{fold}" /
        "test"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = []

    for wi, p in enumerate(files):

        z = np.load(
            p,
            allow_pickle=True,
        )

        xyz = np.asarray(
            z["xyz"],
            dtype=float,
        )

        valid = np.asarray(
            z["valid_xyz_mask"],
            dtype=bool,
        )

        labeled = np.asarray(
            z["labeled_mask"],
            dtype=bool,
        )

        finite = np.isfinite(
            xyz
        ).all(axis=1)

        nonzero = ~np.all(
            np.isclose(xyz, 0.0),
            axis=1,
        )

        geom = (
            valid &
            finite &
            nonzero
        )

        # 不允许悄悄丢 labeled query
        if np.any(
            labeled & (~geom)
        ):
            raise RuntimeError(
                f"{p}: labeled neuron "
                "without valid geometry"
            )

        orig_idx = np.where(
            geom
        )[0]

        # test preprocessing:
        # 无任何 GT 信息
        X = normalize_cloud(
            xyz[geom]
        )

        # 只应用 train-only R
        X = X @ R

        tag = f"worm_{wi:03d}"

        input_file = (
            out_dir /
            f"{tag}_input.mat"
        )

        sidecar_file = (
            out_dir /
            f"{tag}_sidecar.npz"
        )

        # MATLAB inference input ONLY has XYZ.
        savemat(
            input_file,
            {
                "mu_marker": X
            }
        )

        # GT 只保存在 Python evaluation sidecar 中
        np.savez(
            sidecar_file,
            source=str(
                p.resolve()
            ),
            orig_idx=orig_idx,
        )

        manifest.append({
            "worm": wi,
            "source": str(p.resolve()),
            "input": str(
                input_file.resolve()
            ),
            "n_nodes": int(
                len(orig_idx)
            ),
            "n_queries": int(
                labeled.sum()
            ),
        })

    with open(
        out_dir / "manifest.json",
        "w",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        f"fold{fold}: "
        f"{len(files)} test worms prepared"
    )


def main():
    RUN_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    atlas_names, atlas_xyz = (
        load_official_atlas()
    )

    print("=" * 90)
    print("CRF_ID OFFICIAL SINGLE-ATLAS — ATANAS CV5")
    print("=" * 90)

    print(
        "ONLY ATLAS =",
        ATLAS_FILE.resolve()
    )

    print(
        "ATLAS STATES =",
        len(atlas_names)
    )

    for fold in range(5):

        R, shared, rmse = (
            fit_train_only_frame(
                fold,
                atlas_names,
                atlas_xyz,
            )
        )

        fold_dir = (
            RUN_ROOT /
            f"fold{fold}"
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            fold_dir / "R.npy",
            R,
        )

        with open(
            fold_dir /
            "frame_audit.json",
            "w",
        ) as f:

            json.dump(
                {
                    "fold": fold,

                    # 明确记录只有一个 atlas
                    "atlas_file":
                        str(
                            ATLAS_FILE.resolve()
                        ),

                    "atlas_count": 1,
                    "atlas_states": 178,

                    "frame_calibration":
                        "outer-fold train only",

                    "test_labels_used":
                        False,

                    "atanas_train_atlas_built":
                        False,

                    "shared_train_ids":
                        len(shared),

                    "alignment_rmse":
                        float(rmse),

                    "det_R":
                        float(
                            np.linalg.det(R)
                        ),
                },
                f,
                indent=2,
            )

        prepare_fold(
            fold,
            R,
        )

    print("=" * 90)
    print("DONE")
    print("CRF TEMPLATE COUNT = 1")
    print("=" * 90)


if __name__ == "__main__":
    main()
