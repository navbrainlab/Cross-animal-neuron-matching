import os
import os.path as osp
import glob
import hashlib

import numpy as np
import torch

from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def stable_cell_id(x):
    """
    Deterministically convert a string cell identity to int64.
    Used only as GT label, never as model input.
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

    value = int.from_bytes(h, byteorder="little", signed=False)

    # keep inside signed int64
    value = value & ((1 << 63) - 1)

    return value


def normalize_cloud(xyz):
    xyz = xyz.astype(np.float32).copy()

    center = np.median(xyz, axis=0, keepdims=True)
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


class RLDPairDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_root,
        subset,
        min_shared=5,
        normalize=True,
    ):
        super().__init__()

        assert subset in ["train", "val", "test"]

        self.dataset_root = dataset_root
        self.subset = subset
        self.min_shared = min_shared
        self.normalize = normalize

        split_root = osp.join(dataset_root, subset)

        self.files = sorted(
            glob.glob(
                osp.join(split_root, "**", "*.npz"),
                recursive=True,
            )
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No NPZ files found under: {split_root}"
            )

        # cache IDs so pair construction is cheap
        self.id_cache = []

        for path in self.files:
            _, ids = self._load_cloud(path)
            self.id_cache.append(ids)

        self.pairs = []

        # Ordered pairs:
        # worm A -> worm B and worm B -> worm A are both evaluated.
        for i in range(len(self.files)):
            ids_i = self.id_cache[i]
            valid_i = set(ids_i[ids_i >= 0].tolist())

            for j in range(len(self.files)):
                if i == j:
                    continue

                ids_j = self.id_cache[j]
                valid_j = set(ids_j[ids_j >= 0].tolist())

                n_shared = len(valid_i.intersection(valid_j))

                if n_shared >= self.min_shared:
                    self.pairs.append((i, j, n_shared))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No usable {subset} animal pairs. "
                f"files={len(self.files)}, min_shared={self.min_shared}"
            )

        print(
            f"[RLDPairDataset] subset={subset} "
            f"worms={len(self.files)} "
            f"ordered_pairs={len(self.pairs)} "
            f"min_shared={self.min_shared}"
        )

    def _load_cloud(self, path):
        with np.load(path, allow_pickle=False) as z:
            xyz = z["xyz"].astype(np.float32)

            if "cell_id" not in z:
                raise KeyError(
                    f"{path} does not contain cell_id"
                )

            raw_ids = z["cell_id"]

            if "labeled_mask" in z:
                labeled_mask = z["labeled_mask"].astype(bool)
            else:
                labeled_mask = np.ones(
                    len(raw_ids),
                    dtype=bool,
                )

        if xyz.shape[0] != len(raw_ids):
            raise RuntimeError(
                f"xyz/cell_id size mismatch in {path}: "
                f"{xyz.shape[0]} vs {len(raw_ids)}"
            )

        finite_mask = np.isfinite(xyz).all(axis=1)

        xyz = xyz[finite_mask]
        raw_ids = raw_ids[finite_mask]
        labeled_mask = labeled_mask[finite_mask]

        ids = np.full(
            len(raw_ids),
            -1,
            dtype=np.int64,
        )

        for k, cid in enumerate(raw_ids):
            if labeled_mask[k]:
                ids[k] = stable_cell_id(cid)

        if self.normalize:
            xyz = normalize_cloud(xyz)

        return xyz, ids

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ref_idx, src_idx, n_shared = self.pairs[index]

        ref_path = self.files[ref_idx]
        src_path = self.files[src_idx]

        ref_points, ref_ids = self._load_cloud(ref_path)
        src_points, src_ids = self._load_cloud(src_path)

        ref_feats = np.ones(
            (ref_points.shape[0], 1),
            dtype=np.float32,
        )

        src_feats = np.ones(
            (src_points.shape[0], 1),
            dtype=np.float32,
        )

        # registration transform is intentionally NOT used for supervision.
        # Identity is retained only for compatibility with generic utilities.
        transform = np.eye(4, dtype=np.float32)

        return {
            "ref_points": ref_points,
            "src_points": src_points,

            "ref_feats": ref_feats,
            "src_feats": src_feats,

            "ref_ids": ref_ids,
            "src_ids": src_ids,

            "transform": transform,

            "ref_name": osp.basename(ref_path),
            "src_name": osp.basename(src_path),

            "pair_index": int(index),
            "num_shared": int(n_shared),
        }


def build_dataset(cfg, subset):
    return RLDPairDataset(
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
