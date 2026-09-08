import os
import os.path as osp
import glob
import hashlib
from collections import defaultdict

import numpy as np
import torch

from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def stable_cell_id(x):
    """
    Deterministically map pair-local cell ID to int64.
    Used as GT only, never as model input.
    """
    s = str(x).strip()

    if s.lower() in {
        "",
        "nan",
        "none",
        "unknown",
        "unk",
        "?",
        "-1",
    }:
        return -1

    h = hashlib.blake2b(
        s.encode("utf-8"),
        digest_size=8,
    ).digest()

    value = int.from_bytes(
        h,
        byteorder="little",
        signed=False,
    )

    return value & ((1 << 63) - 1)


def normalize_cloud(xyz):
    # Exact normalization used by the existing semantic
    # GeoTransformer experiment.
    xyz = xyz.astype(np.float32).copy()

    center = np.median(
        xyz,
        axis=0,
        keepdims=True,
    )
    xyz = xyz - center

    radius = np.linalg.norm(xyz, axis=1)

    if len(radius) > 0:
        scale = np.percentile(radius, 90)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    xyz = xyz / scale

    return xyz.astype(np.float32)


def _scalar_string(x):
    a = np.asarray(x)

    if a.ndim == 0:
        x = a.item()
    elif a.size == 1:
        x = a.reshape(-1)[0]

    if isinstance(x, bytes):
        x = x.decode()

    return str(x)


class ZebrafishPairDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_root,
        subset,
        min_shared=20,
        normalize=True,
    ):
        super().__init__()

        assert subset in ["train", "val", "test"]

        self.dataset_root = dataset_root
        self.subset = subset
        self.min_shared = min_shared
        self.normalize = normalize

        split_root = osp.join(
            dataset_root,
            subset,
        )

        self.files = sorted(
            glob.glob(
                osp.join(
                    split_root,
                    "**",
                    "*.npz",
                ),
                recursive=True,
            )
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No NPZ files found under: {split_root}"
            )

        # --------------------------------------------------------------
        # Read pair metadata.
        # --------------------------------------------------------------
        self.meta = []

        for path in self.files:
            with np.load(
                path,
                allow_pickle=True,
            ) as z:

                if "pair_id" not in z:
                    raise KeyError(
                        f"{path}: pair_id missing"
                    )

                if "side" not in z:
                    raise KeyError(
                        f"{path}: side missing"
                    )

                pair_id = _scalar_string(
                    z["pair_id"]
                )

                side = _scalar_string(
                    z["side"]
                ).lower()

            if side not in {"q", "r"}:
                raise RuntimeError(
                    f"{path}: invalid side={side}"
                )

            self.meta.append({
                "pair_id": pair_id,
                "side": side,
            })

        # Cache supervised IDs.
        self.id_cache = []

        for path in self.files:
            _, ids = self._load_cloud(path)
            self.id_cache.append(ids)

        # --------------------------------------------------------------
        # STRICT PAIR-LOCAL PROTOCOL
        #
        # Only q/r belonging to the same physical longitudinal pair
        # may be matched.
        #
        # Both directions are included for train/val so the model does
        # not learn a privileged temporal direction.
        # --------------------------------------------------------------
        groups = defaultdict(dict)

        for index, meta in enumerate(self.meta):
            pair_id = meta["pair_id"]
            side = meta["side"]

            if side in groups[pair_id]:
                raise RuntimeError(
                    f"Duplicate {side} for pair {pair_id}"
                )

            groups[pair_id][side] = index

        self.pairs = []
        physical_pairs = 0

        for pair_id in sorted(groups):
            sides = groups[pair_id]

            if set(sides) != {"q", "r"}:
                raise RuntimeError(
                    f"Incomplete pair {pair_id}: "
                    f"{sorted(sides)}"
                )

            qi = sides["q"]
            ri = sides["r"]

            q_ids = self.id_cache[qi]
            r_ids = self.id_cache[ri]

            q_valid = set(
                q_ids[q_ids >= 0].tolist()
            )
            r_valid = set(
                r_ids[r_ids >= 0].tolist()
            )

            n_shared = len(
                q_valid.intersection(r_valid)
            )

            if n_shared < self.min_shared:
                continue

            physical_pairs += 1

            # q -> r
            self.pairs.append(
                (qi, ri, n_shared)
            )

            # r -> q
            self.pairs.append(
                (ri, qi, n_shared)
            )

        if not self.pairs:
            raise RuntimeError(
                f"No usable {subset} q/r pairs. "
                f"files={len(self.files)} "
                f"min_shared={self.min_shared}"
            )

        print(
            f"[ZebrafishPairDataset] "
            f"subset={subset} "
            f"files={len(self.files)} "
            f"physical_pairs={physical_pairs} "
            f"ordered_pairs={len(self.pairs)} "
            f"min_shared={self.min_shared}"
        )

    def _load_cloud(self, path):

        with np.load(
            path,
            allow_pickle=True,
        ) as z:

            xyz = np.asarray(
                z["xyz"],
                dtype=np.float32,
            )

            raw_ids = np.asarray(
                z["cell_id"]
            ).reshape(-1)

            n = len(raw_ids)

            if xyz.shape[0] != n:
                raise RuntimeError(
                    f"{path}: xyz={xyz.shape}, "
                    f"cell_id={raw_ids.shape}"
                )

            # Candidate population:
            # same convention as locked MPRT.
            finite = np.isfinite(
                xyz
            ).all(axis=1)

            if "valid_xyz_mask" in z:
                candidate = (
                    finite
                    & np.asarray(
                        z["valid_xyz_mask"],
                        dtype=bool,
                    ).reshape(-1)
                )
            else:
                candidate = finite

            def mask_or_true(key):
                if key in z:
                    return np.asarray(
                        z[key],
                        dtype=bool,
                    ).reshape(-1)

                return np.ones(
                    n,
                    dtype=bool,
                )

            labeled = mask_or_true(
                "labeled_mask"
            )
            certain = mask_or_true(
                "certain_mask"
            )
            clean = mask_or_true(
                "clean_mask"
            )

        xyz = xyz[candidate]
        raw_ids = raw_ids[candidate]

        supervised = (
            labeled
            & certain
            & clean
        )[candidate]

        ids = np.full(
            len(raw_ids),
            -1,
            dtype=np.int64,
        )

        for k, cid in enumerate(raw_ids):
            if supervised[k]:
                ids[k] = stable_cell_id(cid)

        if self.normalize:
            xyz = normalize_cloud(xyz)

        return xyz, ids

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        ref_idx, src_idx, n_shared = (
            self.pairs[index]
        )

        ref_path = self.files[ref_idx]
        src_path = self.files[src_idx]

        ref_points, ref_ids = (
            self._load_cloud(ref_path)
        )

        src_points, src_ids = (
            self._load_cloud(src_path)
        )

        ref_feats = np.ones(
            (ref_points.shape[0], 1),
            dtype=np.float32,
        )

        src_feats = np.ones(
            (src_points.shape[0], 1),
            dtype=np.float32,
        )

        return {
            "ref_points": ref_points,
            "src_points": src_points,

            "ref_feats": ref_feats,
            "src_feats": src_feats,

            "ref_ids": ref_ids,
            "src_ids": src_ids,

            # Identity supervision only.
            "transform": np.eye(
                4,
                dtype=np.float32,
            ),

            "ref_name": osp.basename(
                ref_path
            ),
            "src_name": osp.basename(
                src_path
            ),

            "pair_index": int(index),
            "num_shared": int(n_shared),
        }


def build_dataset(cfg, subset):
    return ZebrafishPairDataset(
        cfg.data.dataset_root,
        subset,
        min_shared=cfg.data.min_shared,
        normalize=cfg.data.normalize,
    )


def train_valid_data_loader(cfg, distributed):

    train_dataset = build_dataset(cfg, "train")

    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    print("neighbor_limits:", neighbor_limits)

    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    valid_dataset = build_dataset(cfg, "val")

    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg):

    train_dataset = build_dataset(cfg, "train")

    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = build_dataset(cfg, "test")

    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits
