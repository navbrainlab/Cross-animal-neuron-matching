from pathlib import Path
from itertools import permutations, product

import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist


DATA_ROOT = Path(
    "nuclr/Data/Dunn_001623/cv5_grouped_v1"
)

ATLAS_FILE = Path(
    "CRF_Cell_ID/sample_run/"
    "data_neuron_relationship_annotation_updated.mat"
)


# ============================================================
# Utilities
# ============================================================

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


def normalize_cloud(X):
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    X = X - X.mean(
        axis=0,
        keepdims=True,
    )

    rms = np.sqrt(
        np.mean(
            np.sum(
                X * X,
                axis=1,
            )
        )
    )

    if rms < 1e-12:
        raise RuntimeError(
            "Degenerate cloud"
        )

    return X / rms


def pca_basis(X):
    """
    Returns eigenvectors as columns.
    No labels involved.
    """

    X = normalize_cloud(X)

    C = X.T @ X / len(X)

    vals, vecs = np.linalg.eigh(C)

    order = np.argsort(
        vals
    )[::-1]

    vals = vals[order]
    vecs = vecs[:, order]

    return vecs, vals


# ============================================================
# Signed permutations
# ============================================================

def signed_permutation_matrices():
    """
    All 3! * 2^3 = 48 possible PCA axis
    permutation/sign configurations.

    Reflections are intentionally permitted because
    RLD acquisition frames may differ in handedness.
    """

    mats = []

    for perm in permutations(
        range(3)
    ):
        P = np.zeros(
            (3, 3),
            dtype=float,
        )

        for i, j in enumerate(perm):
            P[i, j] = 1.0

        for signs in product(
            [-1.0, 1.0],
            repeat=3,
        ):
            S = np.diag(signs)

            M = P @ S

            mats.append(
                (
                    perm,
                    signs,
                    M,
                )
            )

    assert len(mats) == 48

    return mats


CANDIDATES = (
    signed_permutation_matrices()
)


# ============================================================
# Atlas
# ============================================================

S = loadmat(
    ATLAS_FILE,
    squeeze_me=True,
    struct_as_record=False,
)

ATLAS_NAMES = [
    mstr(x)
    for x in np.asarray(
        S["Neuron_head"]
    ).reshape(-1)
]

ATLAS_XYZ = np.column_stack([
    np.asarray(
        S["X_rot"],
        float,
    ).reshape(-1),

    np.asarray(
        S["Y_rot"],
        float,
    ).reshape(-1),

    np.asarray(
        S["Z_rot"],
        float,
    ).reshape(-1),
])

ATLAS_XYZ = normalize_cloud(
    ATLAS_XYZ
)

ATLAS_PCA, ATLAS_EVALS = (
    pca_basis(
        ATLAS_XYZ
    )
)

ATLAS_LOOKUP = {
    name: ATLAS_XYZ[i]
    for i, name
    in enumerate(ATLAS_NAMES)
}


# ============================================================
# Label-free geometry
# ============================================================

def valid_geometry(z):
    xyz = np.asarray(
        z["xyz"],
        dtype=float,
    )

    valid = np.asarray(
        z["valid_xyz_mask"],
        dtype=bool,
    )

    finite = np.isfinite(
        xyz
    ).all(axis=1)

    nonzero = ~np.all(
        np.isclose(
            xyz,
            0.0,
        ),
        axis=1,
    )

    mask = (
        valid &
        finite &
        nonzero
    )

    return xyz, mask


def trimmed_oneway_chamfer(
    X,
    A,
    keep_fraction=0.80,
):
    """
    Query -> atlas only.

    We deliberately do not penalize atlas neurons that
    are absent from the query recording.
    """

    D = cdist(
        X,
        A,
        metric="euclidean",
    )

    nearest = D.min(
        axis=1
    )

    k = max(
        3,
        int(
            np.ceil(
                keep_fraction *
                len(nearest)
            )
        ),
    )

    keep = np.partition(
        nearest,
        k - 1,
    )[:k]

    return float(
        np.mean(keep)
    )


def label_free_frame(X):
    """
    IMPORTANT:
    Uses XYZ only.

    No:
      cell_id
      clean_mask
      labeled_mask
      correspondence
    """

    Xn = normalize_cloud(X)

    query_pca, evals = (
        pca_basis(Xn)
    )

    best = None

    for perm, signs, M in CANDIDATES:

        # Row-vector convention:
        #
        # X_query
        # -> query PCA coordinates
        # -> signed/permuted canonical coordinates
        # -> official atlas coordinate system

        R = (
            query_pca
            @ M
            @ ATLAS_PCA.T
        )

        Xt = Xn @ R

        score = (
            trimmed_oneway_chamfer(
                Xt,
                ATLAS_XYZ,
                keep_fraction=0.80,
            )
        )

        item = {
            "score": score,
            "R": R,
            "perm": perm,
            "signs": signs,
            "det": float(
                np.linalg.det(R)
            ),
            "evals": evals,
        }

        if (
            best is None
            or score <
            best["score"]
        ):
            best = item

    return best


# ============================================================
# Oracle evaluation
#
# NOTE:
# This section uses train identities ONLY to evaluate whether
# the label-free orientation is sensible.
#
# It is NOT part of the orientation algorithm.
# ============================================================

def oracle_shared_pairs(z):
    xyz = np.asarray(
        z["xyz"],
        dtype=float,
    )

    ids = np.asarray(
        z["cell_id"]
    ).astype(str)

    clean = np.asarray(
        z["clean_mask"],
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

    use = (
        clean &
        valid &
        finite &
        nonzero
    )

    src = []
    dst = []

    for i in np.where(use)[0]:

        name = cid(
            ids[i]
        )

        if name not in ATLAS_LOOKUP:
            continue

        src.append(
            xyz[i]
        )

        dst.append(
            ATLAS_LOOKUP[name]
        )

    if len(src) < 6:
        return None

    X = normalize_cloud(
        np.asarray(src)
    )

    Y = normalize_cloud(
        np.asarray(dst)
    )

    return X, Y


def oracle_frame(X, Y):
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

    return R, rmse


def evaluate_selected_frame(
    z,
    R_selected,
):
    pairs = oracle_shared_pairs(z)

    if pairs is None:
        return None

    X, Y = pairs

    pred = X @ R_selected

    rmse = np.sqrt(
        np.mean(
            np.sum(
                (pred - Y) ** 2,
                axis=1,
            )
        )
    )

    R_oracle, oracle_rmse = (
        oracle_frame(
            X,
            Y,
        )
    )

    R_distance = np.linalg.norm(
        R_selected -
        R_oracle,
        ord="fro",
    )

    return {
        "selected_rmse": float(
            rmse
        ),
        "oracle_rmse": float(
            oracle_rmse
        ),
        "R_distance": float(
            R_distance
        ),
        "n_shared": int(
            len(X)
        ),
        "oracle_det": float(
            np.linalg.det(
                R_oracle
            )
        ),
    }


# ============================================================
# Train-only audit
# ============================================================

for fold in range(5):

    train_dir = (
        DATA_ROOT /
        f"fold_{fold}" /
        "train"
    )

    files = sorted(
        train_dir.glob(
            "*.npz"
        )
    )

    rows = []

    for p in files:

        z = np.load(
            p,
            allow_pickle=True,
        )

        xyz, geom = (
            valid_geometry(z)
        )

        if geom.sum() < 6:
            continue

        # ----------------------------------------------------
        # SELECT FRAME:
        # XYZ ONLY
        # ----------------------------------------------------

        chosen = (
            label_free_frame(
                xyz[geom]
            )
        )

        # ----------------------------------------------------
        # EVALUATE FRAME:
        # train GT only
        # ----------------------------------------------------

        audit = (
            evaluate_selected_frame(
                z,
                chosen["R"],
            )
        )

        if audit is None:
            continue

        rows.append({
            "file": p.name,
            "chamfer": chosen[
                "score"
            ],
            "selected_det":
                chosen["det"],
            **audit,
        })

    print()
    print("=" * 110)
    print(
        f"FOLD {fold} — "
        "LABEL-FREE FRAME AUDIT"
    )
    print("=" * 110)

    if not rows:
        print("NO USABLE WORMS")
        continue

    sel_rmse = np.array([
        r["selected_rmse"]
        for r in rows
    ])

    ora_rmse = np.array([
        r["oracle_rmse"]
        for r in rows
    ])

    rdist = np.array([
        r["R_distance"]
        for r in rows
    ])

    det_match = np.array([
        np.sign(
            r["selected_det"]
        )
        ==
        np.sign(
            r["oracle_det"]
        )
        for r in rows
    ])

    print(
        "usable worms       =",
        len(rows)
    )

    print(
        "LABEL-FREE RMSE    : "
        f"mean={sel_rmse.mean():.4f} "
        f"median={np.median(sel_rmse):.4f}"
    )

    print(
        "ORACLE RMSE        : "
        f"mean={ora_rmse.mean():.4f} "
        f"median={np.median(ora_rmse):.4f}"
    )

    print(
        "R distance         : "
        f"mean={rdist.mean():.4f} "
        f"median={np.median(rdist):.4f}"
    )

    print(
        "det sign agreement : "
        f"{100*det_match.mean():.2f}%"
    )

    print()

    for r in rows[:20]:

        print(
            f"{r['file'][:43]:43s} "
            f"N={r['n_shared']:2d} "
            f"Chamfer={r['chamfer']:.4f} "
            f"RMSE={r['selected_rmse']:.4f} "
            f"Oracle={r['oracle_rmse']:.4f} "
            f"Rdist={r['R_distance']:.3f} "
            f"det={r['selected_det']:+.0f}/"
            f"{r['oracle_det']:+.0f}"
        )

print()
print("=" * 110)
print("TRAIN-ONLY LABEL-FREE FRAME AUDIT COMPLETE")
print("=" * 110)
