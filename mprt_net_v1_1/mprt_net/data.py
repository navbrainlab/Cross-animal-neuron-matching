from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F


_INVALID_IDS = {"", "nan", "none", "null", "unknown", "unk"}


@dataclass
class WormSample:
    uid: str
    xyz: torch.Tensor
    activity: torch.Tensor
    cell_ids: tuple[str, ...]
    supervised_mask: torch.Tensor
    source_path: str

    @property
    def num_nodes(self) -> int:
        return int(self.xyz.shape[0])

    def to(self, device: torch.device | str) -> "WormSample":
        return WormSample(
            uid=self.uid,
            xyz=self.xyz.to(device),
            activity=self.activity.to(device),
            cell_ids=self.cell_ids,
            supervised_mask=self.supervised_mask.to(device),
            source_path=self.source_path,
        )

    def subset(self, keep: torch.Tensor) -> "WormSample":
        keep = keep.to(device="cpu", dtype=torch.bool)
        indices = keep.nonzero(as_tuple=False).flatten().tolist()
        tensor_keep = keep.to(self.xyz.device)
        return WormSample(
            uid=self.uid,
            xyz=self.xyz[tensor_keep],
            activity=self.activity[tensor_keep],
            cell_ids=tuple(self.cell_ids[i] for i in indices),
            supervised_mask=self.supervised_mask[tensor_keep],
            source_path=self.source_path,
        )


@dataclass
class PairTargets:
    """Categorical row and column targets for an augmented assignment.

    ``-1`` means unknown and is ignored.  For rows, ``num_b`` is the dustbin.
    For columns, ``num_a`` is the dustbin.
    """

    row_target: torch.Tensor
    col_target: torch.Tensor
    num_direct_matches: int
    num_synthetic_unmatched: int

    def to(self, device: torch.device | str) -> "PairTargets":
        return PairTargets(
            row_target=self.row_target.to(device),
            col_target=self.col_target.to(device),
            num_direct_matches=self.num_direct_matches,
            num_synthetic_unmatched=self.num_synthetic_unmatched,
        )


def _as_bool(z: np.lib.npyio.NpzFile, key: str, n: int, default: bool) -> np.ndarray:
    if key not in z.files:
        return np.full(n, default, dtype=bool)
    value = np.asarray(z[key], dtype=bool)
    if value.shape != (n,):
        raise ValueError(f"{key} must have shape ({n},), got {value.shape}")
    return value


def _valid_id(value: str) -> bool:
    return value.strip().lower() not in _INVALID_IDS


def _resample_activity(activity: torch.Tensor, length: int | None) -> torch.Tensor:
    if length is None or length <= 0 or activity.shape[1] == length:
        return activity
    # Treat neurons as channels so every trace is interpolated independently.
    return F.interpolate(
        activity.unsqueeze(0), size=length, mode="linear", align_corners=False
    ).squeeze(0)


def load_worm(path: str | Path, activity_length: int | None = 512) -> WormSample:
    """Load one Atanas or RLD NPZ without using labels as input features.

    All neurons with finite coordinates are retained as population context.
    ``supervised_mask`` only controls which identities may create targets.
    """

    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        required = {"activity_raw", "xyz", "cell_id"}
        missing = required.difference(z.files)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")

        xyz = np.asarray(z["xyz"], dtype=np.float32)
        activity = np.asarray(z["activity_raw"], dtype=np.float32)
        cell_ids_raw = np.asarray(z["cell_id"]).astype(str)

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be [N, 3], got {xyz.shape} in {path}")
        n = xyz.shape[0]
        if activity.ndim != 2:
            raise ValueError(f"activity_raw must be 2-D, got {activity.shape}")
        if activity.shape[0] != n and activity.shape[1] == n:
            activity = activity.T
        if activity.shape[0] != n or cell_ids_raw.shape != (n,):
            raise ValueError(
                f"Inconsistent node counts in {path}: xyz={xyz.shape}, "
                f"activity={activity.shape}, cell_id={cell_ids_raw.shape}"
            )

        finite_xyz = np.isfinite(xyz).all(axis=1)
        finite_xyz &= _as_bool(z, "valid_xyz_mask", n, True)
        if finite_xyz.sum() < 2:
            raise ValueError(f"Fewer than two valid neurons in {path}")

        labeled = _as_bool(z, "labeled_mask", n, True)
        certain = _as_bool(z, "certain_mask", n, True)
        clean = _as_bool(z, "clean_mask", n, True)
        id_valid = np.asarray([_valid_id(v) for v in cell_ids_raw], dtype=bool)
        supervised = labeled & certain & clean & id_valid

        uid = str(z["recording_uid"].item()) if "recording_uid" in z.files else path.stem

    xyz = xyz[finite_xyz]
    activity = activity[finite_xyz]
    cell_ids = tuple(v.strip() for v in cell_ids_raw[finite_xyz])
    supervised = supervised[finite_xyz]

    # Non-finite activity is never allowed to poison correlations.  The two
    # supplied datasets are finite, but this makes the contract explicit.
    finite = np.isfinite(activity)
    if not finite.all():
        row_median = np.nanmedian(np.where(finite, activity, np.nan), axis=1)
        row_median = np.nan_to_num(row_median, nan=0.0)
        activity = np.where(finite, activity, row_median[:, None])

    xyz_t = torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32))
    activity_t = torch.from_numpy(np.ascontiguousarray(activity, dtype=np.float32))
    activity_t = _resample_activity(activity_t, activity_length)

    return WormSample(
        uid=uid,
        xyz=xyz_t,
        activity=activity_t,
        cell_ids=cell_ids,
        supervised_mask=torch.from_numpy(supervised.astype(bool, copy=False)),
        source_path=str(path),
    )


def unique_identity_map(sample: WormSample) -> dict[str, int]:
    candidates = [
        identity
        for identity, valid in zip(sample.cell_ids, sample.supervised_mask.tolist())
        if valid
    ]
    counts = Counter(candidates)
    return {
        identity: index
        for index, (identity, valid) in enumerate(
            zip(sample.cell_ids, sample.supervised_mask.tolist())
        )
        if valid and counts[identity] == 1
    }


def _drop_mask(
    n: int,
    probability: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if probability <= 0:
        return torch.ones(n, dtype=torch.bool)
    keep = torch.rand(n, generator=generator) >= probability
    if int(keep.sum()) < min(2, n):
        # Recover nodes deterministically under the same random draw.
        keep[: min(2, n)] = True
    return keep


def build_pair_targets(
    sample_a: WormSample,
    sample_b: WormSample,
    synthetic_drop_probability: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[WormSample, WormSample, PairTargets]:
    """Create sparse supervision and optional known dustbin examples.

    A label absent from the other recording is *not* treated as unmatched:
    the homolog may simply be one of the unlabeled neurons.  Dustbin targets
    are created only when a previously observed matching node is deliberately
    removed by this function.
    """

    if not 0.0 <= synthetic_drop_probability < 1.0:
        raise ValueError("synthetic_drop_probability must be in [0, 1)")

    map_a = unique_identity_map(sample_a)
    map_b = unique_identity_map(sample_b)
    common = sorted(set(map_a).intersection(map_b))

    keep_a = _drop_mask(sample_a.num_nodes, synthetic_drop_probability, generator)
    keep_b = _drop_mask(sample_b.num_nodes, synthetic_drop_probability, generator)
    old_to_new_a = {
        old: new for new, old in enumerate(keep_a.nonzero(as_tuple=False).flatten().tolist())
    }
    old_to_new_b = {
        old: new for new, old in enumerate(keep_b.nonzero(as_tuple=False).flatten().tolist())
    }

    kept_a = sample_a.subset(keep_a)
    kept_b = sample_b.subset(keep_b)
    row_target = torch.full((kept_a.num_nodes,), -1, dtype=torch.long)
    col_target = torch.full((kept_b.num_nodes,), -1, dtype=torch.long)

    direct = 0
    synthetic_unmatched = 0
    for identity in common:
        old_a, old_b = map_a[identity], map_b[identity]
        present_a, present_b = old_a in old_to_new_a, old_b in old_to_new_b
        if present_a and present_b:
            new_a, new_b = old_to_new_a[old_a], old_to_new_b[old_b]
            row_target[new_a] = new_b
            col_target[new_b] = new_a
            direct += 1
        elif present_a and not present_b:
            row_target[old_to_new_a[old_a]] = kept_b.num_nodes
            synthetic_unmatched += 1
        elif present_b and not present_a:
            col_target[old_to_new_b[old_b]] = kept_a.num_nodes
            synthetic_unmatched += 1

    targets = PairTargets(
        row_target=row_target,
        col_target=col_target,
        num_direct_matches=direct,
        num_synthetic_unmatched=synthetic_unmatched,
    )
    return kept_a, kept_b, targets


def split_files(root: str | Path, split: str) -> list[Path]:
    directory = Path(root) / split
    if not directory.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {directory}")
    files = sorted(directory.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files in {directory}")
    return files


def _file_unique_ids(path: Path) -> frozenset[str]:
    with np.load(path, allow_pickle=False) as z:
        ids = np.asarray(z["cell_id"]).astype(str)
        n = len(ids)
        mask = (
            _as_bool(z, "labeled_mask", n, True)
            & _as_bool(z, "certain_mask", n, True)
            & _as_bool(z, "clean_mask", n, True)
        )
    values = [value.strip() for value, valid in zip(ids, mask) if valid and _valid_id(value)]
    counts = Counter(values)
    return frozenset(value for value, count in counts.items() if count == 1)


class PairIndex:
    """All within-split animal pairs with enough shared clean identities."""

    def __init__(self, root: str | Path, split: str, min_shared: int = 2):
        self.files = split_files(root, split)
        identities = {path: _file_unique_ids(path) for path in self.files}
        self.pairs: list[tuple[Path, Path]] = []
        self.shared_counts: list[int] = []
        for a, b in combinations(self.files, 2):
            count = len(identities[a].intersection(identities[b]))
            if count >= min_shared:
                self.pairs.append((a, b))
                self.shared_counts.append(count)
        if not self.pairs:
            raise ValueError(
                f"No {split} pairs under {root} have at least {min_shared} shared identities"
            )

    def shuffled(self, generator: torch.Generator) -> list[tuple[Path, Path]]:
        order = torch.randperm(len(self.pairs), generator=generator).tolist()
        return [self.pairs[i] for i in order]


class WormCache:
    def __init__(self, activity_length: int | None = 512, max_items: int = 32):
        self.activity_length = activity_length
        self.max_items = max_items
        self._items: OrderedDict[str, WormSample] = OrderedDict()

    def get(self, path: str | Path) -> WormSample:
        key = str(path)
        if key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return value
        value = load_worm(path, activity_length=self.activity_length)
        self._items[key] = value
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return value


def iter_pairs(
    index: PairIndex,
    cache: WormCache,
    generator: torch.Generator,
    limit: int | None = None,
) -> Iterator[tuple[WormSample, WormSample]]:
    pairs: Sequence[tuple[Path, Path]] = index.shuffled(generator)
    if limit is not None and limit > 0:
        if limit <= len(pairs):
            pairs = pairs[:limit]
        else:
            repeats = (limit + len(pairs) - 1) // len(pairs)
            pairs = (list(pairs) * repeats)[:limit]
    for path_a, path_b in pairs:
        yield cache.get(path_a), cache.get(path_b)
