from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

DATA = Path("nuclr/Data/Dunn_001623/cv5_grouped_v1")
ATLAS = Path(
    "CRF_Cell_ID/sample_run/"
    "data_neuron_relationship_annotation_updated.mat"
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


def cid(x):
    return str(x).strip().rstrip("?")


def norm_cloud(X):
    X = np.asarray(X, float)

    X = X - X.mean(0)

    rms = np.sqrt(
        np.mean(
            np.sum(X * X, axis=1)
        )
    )

    return X / max(rms, 1e-12)


# ------------------------------------------------------------
# Official atlas
# ------------------------------------------------------------

S = loadmat(
    ATLAS,
    squeeze_me=True,
    struct_as_record=False,
)

names = [
    mstr(x)
    for x in np.asarray(
        S["Neuron_head"]
    ).reshape(-1)
]

A = np.column_stack([
    np.asarray(S["X_rot"], float).reshape(-1),
    np.asarray(S["Y_rot"], float).reshape(-1),
    np.asarray(S["Z_rot"], float).reshape(-1),
])

A = norm_cloud(A)

atlas = {
    n: A[i]
    for i, n in enumerate(names)
}


def fit_one_worm(p):
    z = np.load(
        p,
        allow_pickle=True,
    )

    X0 = np.asarray(
        z["xyz"],
        float,
    )

    ids = np.asarray(
        z["cell_id"]
    ).astype(str)

    clean = np.asarray(
        z["clean_mask"],
        bool,
    )

    valid = np.asarray(
        z["valid_xyz_mask"],
        bool,
    )

    finite = np.isfinite(
        X0
    ).all(1)

    nonzero = ~np.all(
        np.isclose(X0, 0),
        axis=1,
    )

    use = clean & valid & finite & nonzero

    src = []
    dst = []
    shared = []

    for i in np.where(use)[0]:

        name = cid(
            ids[i]
        )

        if name not in atlas:
            continue

        src.append(
            X0[i]
        )

        dst.append(
            atlas[name]
        )

        shared.append(name)

    if len(src) < 6:
        return None

    X = norm_cloud(
        np.asarray(src)
    )

    Y = norm_cloud(
        np.asarray(dst)
    )

    U, sv, Vt = np.linalg.svd(
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

    return {
        "file": p.name,
        "n": len(src),
        "R": R,
        "det": np.linalg.det(R),
        "rmse": rmse,
    }


for fold in range(5):

    files = sorted(
        (
            DATA /
            f"fold_{fold}" /
            "train"
        ).glob("*.npz")
    )

    results = []

    for p in files:
        r = fit_one_worm(p)

        if r is not None:
            results.append(r)

    print()
    print("=" * 100)
    print(f"FOLD {fold} — TRAIN WORMS")
    print("=" * 100)

    print(
        "usable worms =",
        len(results),
    )

    rmses = np.array([
        r["rmse"]
        for r in results
    ])

    dets = np.array([
        r["det"]
        for r in results
    ])

    print(
        "per-worm RMSE "
        f"mean={rmses.mean():.4f} "
        f"median={np.median(rmses):.4f} "
        f"min={rmses.min():.4f} "
        f"max={rmses.max():.4f}"
    )

    print(
        "det +1:",
        int((dets > 0).sum()),
        "det -1:",
        int((dets < 0).sum())
    )

    print()

    for r in results[:20]:

        print(
            f"{r['file'][:40]:40s} "
            f"N={r['n']:3d} "
            f"det={r['det']:+.0f} "
            f"RMSE={r['rmse']:.4f}"
        )

    # --------------------------------------------------------
    # How different are worm-specific transformations?
    # Frobenius distance, valid for both rotations/reflections.
    # --------------------------------------------------------

    Rs = [
        r["R"]
        for r in results
    ]

    distances = []

    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):

            distances.append(
                np.linalg.norm(
                    Rs[i] - Rs[j],
                    ord="fro"
                )
            )

    distances = np.asarray(
        distances
    )

    print()

    print(
        "pairwise R Frobenius distance: "
        f"mean={distances.mean():.4f} "
        f"median={np.median(distances):.4f} "
        f"max={distances.max():.4f}"
    )
