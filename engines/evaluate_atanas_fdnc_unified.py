#!/usr/bin/env python3
"""Evaluate frozen position-only fDNC on aligned whole-worm datasets.

Each ``full`` NPZ supplies the complete aligned point cloud through ``xyz``.
The complete point cloud is passed to fDNC, including unlabeled neurons, so the
model retains full spatial context.  Identity supervision and metrics are
restricted to rows where ``clean_mask`` is true.

The script performs two leakage-safe protocols:

1. Pairwise held-out evaluation over ordered worm pairs within val/test.
2. Multi-template identity recognition using train worms as labeled templates.
   Coordinate normalization and aggregation are selected only on validation,
   then locked and evaluated once on test.

No fDNC weight is updated and no test label is used for configuration selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:  # pragma: no cover
    raise ImportError("scipy is required for Hungarian evaluation") from exc


# -----------------------------------------------------------------------------
# Data and normalization
# -----------------------------------------------------------------------------

@dataclass
class Worm:
    path: Path
    name: str
    xyz_raw: np.ndarray
    labels: list[Optional[str]]

    @property
    def n(self) -> int:
        return int(self.xyz_raw.shape[0])


def identity_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null", "unknown", "?"}:
        return None
    return text


def load_worms(root: Path, supervision_mask_key: str = "clean_mask") -> list[Worm]:
    """Load full point clouds while masking labels outside the clean GT subset.

    ``xyz`` is never filtered: every aligned neuron remains in the fDNC input and
    in the candidate set.  Only labels used for supervision/evaluation are masked.
    """
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files under {root}")
    worms: list[Worm] = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            required = {"xyz", "cell_id", supervision_mask_key}
            missing = required - set(data.files)
            if missing:
                raise KeyError(
                    f"{path}: missing {sorted(missing)}; keys={list(data.files)}"
                )
            xyz = np.asarray(data["xyz"], dtype=np.float32)
            raw_labels = np.asarray(data["cell_id"]).reshape(-1)
            clean_mask = np.asarray(
                data[supervision_mask_key], dtype=bool
            ).reshape(-1)

        if xyz.ndim != 2 or xyz.shape[1] < 3:
            raise ValueError(f"{path}: xyz must be [N,>=3], got {xyz.shape}")
        xyz = xyz[:, :3]
        n = xyz.shape[0]
        if n != len(raw_labels) or n != len(clean_mask):
            raise ValueError(
                f"{path}: xyz/label/mask mismatch "
                f"{n} vs {len(raw_labels)} vs {len(clean_mask)}"
            )
        if not np.isfinite(xyz).all():
            raise ValueError(f"{path}: xyz contains NaN/Inf")

        labels: list[Optional[str]] = []
        for value, keep in zip(raw_labels, clean_mask):
            labels.append(identity_text(value) if bool(keep) else None)

        clean_count = sum(label is not None for label in labels)
        if clean_count == 0:
            raise ValueError(f"{path}: no clean identities after {supervision_mask_key}")
        worms.append(Worm(path=path, name=path.stem, xyz_raw=xyz, labels=labels))
        print(
            f"loaded {root.name}/{path.name}: "
            f"full_points={n}, clean_gt={clean_count}"
        )
    return worms


def normalize_xyz(xyz: np.ndarray, mode: str, scale: float) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).copy()
    center = np.median(xyz, axis=0, keepdims=True)
    centered = xyz - center
    if mode == "none":
        return xyz
    if mode == "center":
        return centered
    if mode == "median_scale":
        if scale <= 0:
            raise ValueError("scale must be positive")
        return centered / float(scale)
    if mode == "median_rms":
        radius = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
        return centered / max(radius, 1e-6)
    raise ValueError(mode)


@dataclass(frozen=True)
class NormConfig:
    mode: str
    scale: float

    @property
    def name(self) -> str:
        if self.mode == "median_scale":
            return f"median_scale_{self.scale:g}"
        return self.mode


def parse_norm_configs(text: str) -> list[NormConfig]:
    out: list[NormConfig] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            mode, value = item.split(":", 1)
            out.append(NormConfig(mode.strip(), float(value)))
        else:
            out.append(NormConfig(item, 200.0))
    if not out:
        raise ValueError("No normalization configs")
    return out


# -----------------------------------------------------------------------------
# fDNC model compatible with original NIT_Registration checkpoints
# -----------------------------------------------------------------------------

class STNkd(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).flatten().view(1, -1)
        x = x + identity.repeat(batch_size, 1)
        return x.view(-1, self.k, self.k)


class PointTransFeat(nn.Module):
    def __init__(self, rotate: bool = False, feature_transform: bool = False,
                 input_dim: int = 3, hidden_d: int = 128):
        super().__init__()
        self.hidden_d = hidden_d
        self.rotate = rotate
        self.feature_transform = feature_transform
        if rotate:
            self.stn = STNkd(k=input_dim)
        self.conv1 = nn.Conv1d(input_dim, hidden_d, 1)
        self.bn1 = nn.BatchNorm1d(hidden_d)
        if feature_transform:
            self.fstn = STNkd(k=hidden_d)

    def forward(self, x: Tensor) -> Tensor:
        if self.rotate:
            transform = self.stn(x)
            x = torch.bmm(x.transpose(2, 1), transform).transpose(2, 1)
        x = self.bn1(self.conv1(x))
        if self.feature_transform:
            transform_feat = self.fstn(x)
            x = torch.bmm(x.transpose(2, 1), transform_feat).transpose(2, 1)
        return x


class FdncRegistrationBackbone(nn.Module):
    def __init__(self, input_dim: int = 3, n_hidden: int = 128, n_layer: int = 6,
                 p_rotate: bool = False, feat_trans: bool = False):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_hidden = int(n_hidden)
        self.n_layer = int(n_layer)
        self.p_rotate = bool(p_rotate)
        self.feat_trans = bool(feat_trans)
        self.point_f = PointTransFeat(
            rotate=self.p_rotate, feature_transform=self.feat_trans,
            input_dim=self.input_dim, hidden_d=self.n_hidden,
        )
        self.enc_l = nn.TransformerEncoderLayer(d_model=self.n_hidden, nhead=8)
        self.model = nn.TransformerEncoder(self.enc_l, self.n_layer)
        self.fc_outlier = nn.Linear(self.n_hidden, 1)


def extract_state_dict(checkpoint: Any) -> Dict[str, Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model_state", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(v, Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Could not locate state dict")


def strip_common_prefixes(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    prefixes = ("module.", "fdnc.", "backbone.")
    cleaned: Dict[str, Tensor] = {}
    for key, value in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned


def load_checkpoint(path: Path, device: torch.device, default_hidden: int,
                    default_layers: int) -> FdncRegistrationBackbone:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError, ValueError, pickle.UnpicklingError):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(args, dict):
        args = {}
    model = FdncRegistrationBackbone(
        input_dim=int(args.get("input_dim", 3)),
        n_hidden=int(args.get("n_hidden", default_hidden)),
        n_layer=int(args.get("n_layer", default_layers)),
        p_rotate=bool(args.get("p_rotate", False)),
        feat_trans=bool(args.get("feat_trans", args.get("f_trans", False))),
    )
    state = strip_common_prefixes(extract_state_dict(checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    missing_core = [k for k in incompatible.missing_keys
                    if k.startswith(("point_f.", "model.", "fc_outlier."))]
    if missing_core:
        raise RuntimeError(f"Incompatible fDNC checkpoint; missing core keys: {missing_core[:20]}")
    print(f"Loaded fDNC: {path}")
    print(f"  hidden={model.n_hidden}, layers={model.n_layer}, "
          f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}")
    return model.to(device).eval()


@torch.no_grad()
def fdnc_score_matrix(model: FdncRegistrationBackbone,
                      ref_xyz: np.ndarray, target_xyz: np.ndarray,
                      device: torch.device) -> np.ndarray:
    """Return rows=target neurons, cols=reference neurons (outlier removed)."""
    n_ref, n_target = len(ref_xyz), len(target_xyz)
    max_n = max(n_ref, n_target)
    xyz = torch.zeros(2, max_n, 3, dtype=torch.float32, device=device)
    xyz[0, :n_ref] = torch.from_numpy(ref_xyz).to(device)
    xyz[1, :n_target] = torch.from_numpy(target_xyz).to(device)
    spatial = model.point_f(xyz.transpose(2, 1)).transpose(2, 1)

    padded = torch.zeros(2, max_n, model.n_hidden, dtype=spatial.dtype, device=device)
    padded[0, :n_ref] = spatial[0, :n_ref]
    padded[1, :n_target] = spatial[1, :n_target]
    padding_mask = torch.ones(2, max_n, dtype=torch.bool, device=device)
    padding_mask[0, :n_ref] = False
    padding_mask[1, :n_target] = False

    repeated_ref = padded[0:1, :n_ref].repeat(2, 1, 1) + 1.0
    ref_mask = torch.zeros(2, n_ref, dtype=torch.bool, device=device)
    transformer_input = torch.cat([repeated_ref, padded], dim=1)
    transformer_mask = torch.cat([ref_mask, padding_mask], dim=1)
    encoded = model.model(
        transformer_input.transpose(0, 1),
        src_key_padding_mask=transformer_mask,
    ).transpose(0, 1)
    ref_embedding = encoded[1, :n_ref]
    target_embedding = encoded[1, n_ref:n_ref + n_target]
    similarity = target_embedding @ ref_embedding.transpose(0, 1)
    return similarity.float().cpu().numpy()


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def unique_identity_map(labels: Sequence[Optional[str]]) -> dict[str, int]:
    positions: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        if label is not None:
            positions.setdefault(label, []).append(i)
    return {label: idxs[0] for label, idxs in positions.items() if len(idxs) == 1}


@dataclass
class PairAccumulator:
    ranks: list[int]
    hungarian_hits: int = 0
    hungarian_queries: int = 0
    pair_rows: list[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.pair_rows is None:
            self.pair_rows = []

    def add(self, scores: np.ndarray, target: Worm, ref: Worm) -> None:
        tmap = unique_identity_map(target.labels)
        rmap = unique_identity_map(ref.labels)
        common = sorted(set(tmap) & set(rmap))
        local_ranks = []
        for identity in common:
            qi, ri = tmap[identity], rmap[identity]
            row = scores[qi]
            rank = 1 + int(np.sum(row > row[ri]))
            self.ranks.append(rank); local_ranks.append(rank)

        hits = 0
        if common:
            rows, cols = linear_sum_assignment(-scores)
            assignment = {int(r): int(c) for r, c in zip(rows, cols)}
            for identity in common:
                qi = tmap[identity]
                self.hungarian_queries += 1
                pred = assignment.get(qi, -1)
                if pred >= 0 and ref.labels[pred] == identity:
                    self.hungarian_hits += 1; hits += 1
        arr = np.asarray(local_ranks, dtype=np.float64)
        self.pair_rows.append({
            "target": target.name, "reference": ref.name,
            "queries": len(common),
            "top1": float(np.mean(arr <= 1)) if len(arr) else math.nan,
            "top5": float(np.mean(arr <= 5)) if len(arr) else math.nan,
            "mrr": float(np.mean(1.0 / arr)) if len(arr) else math.nan,
            "hungarian_top1": hits / len(common) if common else math.nan,
        })

    def result(self) -> dict[str, Any]:
        ranks = np.asarray(self.ranks, dtype=np.float64)
        if not len(ranks):
            return {"queries": 0}
        pair_top1 = np.asarray([r["top1"] for r in self.pair_rows if np.isfinite(r["top1"])])
        return {
            "queries": int(len(ranks)),
            "ranking_top1": float(np.mean(ranks <= 1)),
            "ranking_top3": float(np.mean(ranks <= 3)),
            "ranking_top5": float(np.mean(ranks <= 5)),
            "ranking_top10": float(np.mean(ranks <= 10)),
            "mrr": float(np.mean(1.0 / ranks)),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "hungarian_top1": self.hungarian_hits / max(self.hungarian_queries, 1),
            "num_ordered_pairs": len(self.pair_rows),
            "pair_macro_top1": float(np.mean(pair_top1)) if len(pair_top1) else math.nan,
            "pair_std_top1": float(np.std(pair_top1)) if len(pair_top1) else math.nan,
        }


def retrieval_metrics(score_rows: np.ndarray, true_cols: np.ndarray) -> dict[str, float]:
    true_scores = score_rows[np.arange(len(true_cols)), true_cols]
    ranks = 1 + np.sum(score_rows > true_scores[:, None], axis=1)
    return {
        "queries": int(len(ranks)),
        "top1": float(np.mean(ranks <= 1)),
        "top3": float(np.mean(ranks <= 3)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }


def stable_softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = x / max(float(temperature), 1e-8)
    z = z - np.max(z)
    e = np.exp(np.clip(z, -80, 80))
    return e / max(float(e.sum()), 1e-12)


def row_zscore(x: np.ndarray) -> np.ndarray:
    return (x - float(np.mean(x))) / max(float(np.std(x)), 1e-6)


# -----------------------------------------------------------------------------
# Score cache
# -----------------------------------------------------------------------------

def cache_path(cache_root: Path, norm: NormConfig, ref: Worm, target: Worm) -> Path:
    token = f"{norm.name}::{ref.path.resolve()}::{target.path.resolve()}".encode()
    digest = hashlib.sha256(token).hexdigest()[:16]
    return cache_root / norm.name / f"{target.name}__to__{ref.name}__{digest}.npy"


def get_scores(model: FdncRegistrationBackbone, ref: Worm, target: Worm,
               norm: NormConfig, device: torch.device, cache_root: Path) -> np.ndarray:
    path = cache_path(cache_root, norm, ref, target)
    if path.exists():
        scores = np.load(path)
        if scores.shape == (target.n, ref.n):
            return scores
    ref_xyz = normalize_xyz(ref.xyz_raw, norm.mode, norm.scale)
    target_xyz = normalize_xyz(target.xyz_raw, norm.mode, norm.scale)
    scores = fdnc_score_matrix(model, ref_xyz, target_xyz, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, scores.astype(np.float32))
    return scores


# -----------------------------------------------------------------------------
# Pairwise and multi-template evaluation
# -----------------------------------------------------------------------------

def evaluate_pairwise(model: FdncRegistrationBackbone, worms: Sequence[Worm],
                      norm: NormConfig, device: torch.device,
                      cache_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    acc = PairAccumulator(ranks=[])
    for ti, target in enumerate(worms):
        for ri, ref in enumerate(worms):
            if ti == ri:
                continue
            scores = get_scores(model, ref, target, norm, device, cache_root)
            acc.add(scores, target, ref)
    return acc.result(), acc.pair_rows


def global_template_identities(templates: Sequence[Worm]) -> tuple[list[str], dict[str, int]]:
    names = sorted({x for worm in templates for x in worm.labels if x is not None})
    return names, {name: i for i, name in enumerate(names)}


def template_identity_scores(scores: np.ndarray, labels: Sequence[Optional[str]],
                             id_to_col: dict[str, int], n_ids: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.full((scores.shape[0], n_ids), np.nan, dtype=np.float64)
    available = np.zeros(n_ids, dtype=bool)
    positions: dict[str, list[int]] = {}
    for j, label in enumerate(labels):
        if label is not None:
            positions.setdefault(label, []).append(j)
    for label, idxs in positions.items():
        col = id_to_col[label]
        # Duplicates are uncommon; max keeps the strongest candidate for that identity.
        out[:, col] = np.max(scores[:, idxs], axis=1)
        available[col] = True
    return out, available


def build_evidence(model: FdncRegistrationBackbone, templates: Sequence[Worm],
                   target: Worm, norm: NormConfig, device: torch.device,
                   cache_root: Path, identities: list[str], id_to_col: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    per_template = []
    availability = []
    for idx, ref in enumerate(templates, start=1):
        scores = get_scores(model, ref, target, norm, device, cache_root)
        idscores, avail = template_identity_scores(scores, ref.labels, id_to_col, len(identities))
        per_template.append(idscores); availability.append(avail)
        print(f"      target={target.name} template {idx:02d}/{len(templates):02d} {ref.name}")
    return np.stack(per_template, axis=0), np.stack(availability, axis=0)


def aggregate_methods(evidence: np.ndarray, availability: np.ndarray,
                      softmax_temps: Sequence[float], rrf_ks: Sequence[float],
                      tie_epsilon: float) -> dict[str, np.ndarray]:
    # evidence [R, N, C], NaN where identity is unavailable in a template.
    r_count, n, c = evidence.shape
    available_count = availability.sum(axis=0).astype(np.float64)
    denom = np.maximum(available_count, 1.0)
    methods: dict[str, np.ndarray] = {}

    majority = np.zeros((n, c), dtype=np.float64)
    confidence = np.zeros((n, c), dtype=np.float64)
    confidence_z = np.zeros((n, c), dtype=np.float64)
    raw_sum = np.zeros((n, c), dtype=np.float64)
    z_sum = np.zeros((n, c), dtype=np.float64)
    soft_sums = {t: np.zeros((n, c), dtype=np.float64) for t in softmax_temps}
    rrf_sums = {k: np.zeros((n, c), dtype=np.float64) for k in rrf_ks}

    for r in range(r_count):
        cols = np.flatnonzero(availability[r])
        if len(cols) == 0:
            continue
        for i in range(n):
            vals = evidence[r, i, cols]
            order = np.argsort(-vals, kind="stable")
            winner = cols[order[0]]
            majority[i, winner] += 1.0
            top1 = vals[order[0]]
            top2 = vals[order[1]] if len(order) > 1 else top1
            confidence[i, winner] += max(float(top1 - top2), 0.0)
            zvals = row_zscore(vals)
            zorder = np.argsort(-zvals, kind="stable")
            zwinner = cols[zorder[0]]
            ztop2 = zvals[zorder[1]] if len(zorder) > 1 else zvals[zorder[0]]
            confidence_z[i, zwinner] += max(float(zvals[zorder[0]] - ztop2), 0.0)
            raw_sum[i, cols] += vals
            z_sum[i, cols] += zvals
            for t in softmax_temps:
                soft_sums[t][i, cols] += stable_softmax(vals, t)
            ranks = np.empty(len(cols), dtype=np.int64)
            ranks[order] = np.arange(1, len(cols) + 1)
            for k in rrf_ks:
                rrf_sums[k][i, cols] += 1.0 / (float(k) + ranks)

    z_mean = z_sum / denom[None, :]
    raw_mean = raw_sum / denom[None, :]
    tie = tie_epsilon * z_mean
    methods["majority_vote"] = majority + tie
    methods["majority_vote_availnorm"] = majority / denom[None, :] + tie
    methods["confidence_vote"] = confidence + tie
    methods["confidence_vote_availnorm"] = confidence / denom[None, :] + tie
    methods["confidence_zscore_vote"] = confidence_z + tie
    methods["mean_score"] = raw_mean
    methods["zscore_mean"] = z_mean
    for t, values in soft_sums.items():
        methods[f"softmax_sum_t{t:g}"] = values + tie
        methods[f"softmax_availnorm_t{t:g}"] = values / denom[None, :] + tie
    for k, values in rrf_sums.items():
        methods[f"rrf_k{k:g}"] = values + tie
        methods[f"rrf_availnorm_k{k:g}"] = values / denom[None, :] + tie
    return methods


def evaluate_multitemplate(model: FdncRegistrationBackbone, templates: Sequence[Worm],
                           targets: Sequence[Worm], norm: NormConfig,
                           methods_to_run: Optional[set[str]], device: torch.device,
                           cache_root: Path, softmax_temps: Sequence[float],
                           rrf_ks: Sequence[float], tie_epsilon: float,
                           save_predictions: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    identities, id_to_col = global_template_identities(templates)
    accum: dict[str, dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []

    for target_index, target in enumerate(targets, start=1):
        print(f"    multi-template target {target_index:02d}/{len(targets):02d}: {target.name}")
        evidence, availability = build_evidence(
            model, templates, target, norm, device, cache_root, identities, id_to_col
        )
        method_scores = aggregate_methods(evidence, availability, softmax_temps, rrf_ks, tie_epsilon)
        if methods_to_run is not None:
            method_scores = {k: v for k, v in method_scores.items() if k in methods_to_run}
        available_any = availability.any(axis=0)
        target_unique = unique_identity_map(target.labels)
        covered = [(label, qi, id_to_col[label]) for label, qi in target_unique.items()
                   if label in id_to_col and available_any[id_to_col[label]]]
        total_labeled = len(target_unique)

        for method, scores in method_scores.items():
            state = accum.setdefault(method, {
                "score_rows": [], "true_cols": [], "total_target_neurons": 0,
                "covered_queries": 0, "assignment_hits": 0, "assignment_queries": 0,
            })
            state["total_target_neurons"] += total_labeled
            if covered:
                qidx = np.asarray([x[1] for x in covered], dtype=np.int64)
                true_cols = np.asarray([x[2] for x in covered], dtype=np.int64)
                state["score_rows"].append(scores[qidx])
                state["true_cols"].append(true_cols)
                state["covered_queries"] += len(covered)

            rows, cols = linear_sum_assignment(-scores)
            assignment = {int(r): int(c) for r, c in zip(rows, cols)}
            for label, qi, true_col in covered:
                state["assignment_queries"] += 1
                if assignment.get(qi, -1) == true_col:
                    state["assignment_hits"] += 1

            # Save only top prediction for compactness.
            if save_predictions is not None:
                for label, qi, true_col in covered:
                    order = np.argsort(-scores[qi], kind="stable")
                    prediction_rows.append({
                        "method": method, "target_worm": target.name,
                        "target_neuron_index": qi, "true_identity": label,
                        "pred_identity": identities[int(order[0])],
                        "true_rank": int(1 + np.sum(scores[qi] > scores[qi, true_col])),
                    })

    results: dict[str, dict[str, Any]] = {}
    for method, state in accum.items():
        if state["score_rows"]:
            rows = np.concatenate(state["score_rows"], axis=0)
            true = np.concatenate(state["true_cols"], axis=0)
            metrics = retrieval_metrics(rows, true)
        else:
            metrics = {"queries": 0}
        total = int(state["total_target_neurons"])
        covered = int(state["covered_queries"])
        metrics.update({
            "total_target_neurons": total,
            "coverage": covered / max(total, 1),
            "assignment_queries": int(state["assignment_queries"]),
            "assignment_top1": state["assignment_hits"] / max(state["assignment_queries"], 1),
        })
        results[method] = metrics

    if save_predictions is not None:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        if prediction_rows:
            with save_predictions.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
                writer.writeheader(); writer.writerows(prediction_rows)
    return results


# -----------------------------------------------------------------------------
# CLI and orchestration
# -----------------------------------------------------------------------------

def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen fDNC baseline on Atanas xyz splits")
    # Kept optional at argparse level so --self-test can run standalone.
    p.add_argument("--train-root", type=Path)
    p.add_argument("--val-root", type=Path)
    p.add_argument("--test-root", type=Path)
    p.add_argument("--fdnc-checkpoint", type=Path)
    p.add_argument("--save-dir", type=Path)
    p.add_argument(
        "--supervision-mask-key",
        default="clean_mask",
        help=(
            "Boolean NPZ mask selecting reliable identities for supervision and "
            "metrics. xyz rows are never removed."
        ),
    )
    p.add_argument("--normalizations", default="median_scale:200,median_rms",
                   help="Comma list, e.g. median_scale:200,median_rms")
    p.add_argument("--softmax-temperatures", default="0.1,0.5,1.0")
    p.add_argument("--rrf-k", default="0,60")
    p.add_argument("--tie-epsilon", type=float, default=1e-4)
    p.add_argument("--selection-metric", choices=["top1", "assignment_top1"], default="top1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-hidden", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--skip-pairwise", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def synthetic_self_test() -> None:
    rng = np.random.default_rng(1)
    evidence = rng.normal(size=(4, 7, 9))
    availability = rng.random((4, 9)) > 0.2
    for r in range(4):
        evidence[r, :, ~availability[r]] = np.nan
    methods = aggregate_methods(evidence, availability, [0.1, 1.0], [0.0, 60.0], 1e-4)
    assert "confidence_vote" in methods and all(x.shape == (7, 9) for x in methods.values())
    print(f"self-test passed: {len(methods)} aggregation methods")


def main() -> None:
    args = parse_args()
    if args.self_test:
        synthetic_self_test()
        return
    missing = [
        name for name in ("train_root", "val_root", "test_root", "fdnc_checkpoint", "save_dir")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit("Missing required arguments: " + ", ".join("--" + x.replace("_", "-") for x in missing))
    args.save_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.save_dir / "score_cache"
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train = load_worms(args.train_root, args.supervision_mask_key)
    val = load_worms(args.val_root, args.supervision_mask_key)
    test = load_worms(args.test_root, args.supervision_mask_key)
    print(f"Loaded train={len(train)}, val={len(val)}, test={len(test)} worms")
    print(
        "Protocol: complete xyz point clouds are model inputs; "
        f"only {args.supervision_mask_key}=True rows contribute GT metrics."
    )
    model = load_checkpoint(args.fdnc_checkpoint, device, args.n_hidden, args.n_layer)
    norm_configs = parse_norm_configs(args.normalizations)
    softmax_temps = parse_float_list(args.softmax_temperatures)
    rrf_ks = parse_float_list(args.rrf_k)

    comparison_rows = []
    val_results_all: dict[str, Any] = {}
    for norm in norm_configs:
        print("=" * 100)
        print(f"Validation normalization: {norm.name}")
        pair_metrics = None
        if not args.skip_pairwise:
            pair_metrics, pair_rows = evaluate_pairwise(model, val, norm, device, cache_root)
            write_json(args.save_dir / f"val_pairwise_{norm.name}.json", pair_metrics)
            if pair_rows:
                with (args.save_dir / f"val_pairwise_pairs_{norm.name}.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
                    writer.writeheader(); writer.writerows(pair_rows)
        multi = evaluate_multitemplate(
            model, train, val, norm, None, device, cache_root,
            softmax_temps, rrf_ks, args.tie_epsilon,
            save_predictions=None,
        )
        val_results_all[norm.name] = {"pairwise": pair_metrics, "multitemplate": multi}
        for method, metrics in multi.items():
            comparison_rows.append({
                "normalization": norm.name,
                "method": method,
                "coverage": metrics.get("coverage", math.nan),
                "top1": metrics.get("top1", math.nan),
                "top5": metrics.get("top5", math.nan),
                "mrr": metrics.get("mrr", math.nan),
                "assignment_top1": metrics.get("assignment_top1", math.nan),
            })

    comparison_rows.sort(key=lambda r: (-float(r[args.selection_metric]), r["normalization"], r["method"]))
    if not comparison_rows:
        raise RuntimeError("No validation result")
    best = comparison_rows[0]
    write_json(args.save_dir / "validation_results.json", val_results_all)
    with (args.save_dir / "validation_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader(); writer.writerows(comparison_rows)

    best_norm = next(x for x in norm_configs if x.name == best["normalization"])
    locked = {
        "fdnc_checkpoint": str(args.fdnc_checkpoint),
        "normalization": best_norm.name,
        "normalization_mode": best_norm.mode,
        "normalization_scale": best_norm.scale,
        "method": best["method"],
        "selection_split": str(args.val_root),
        "selection_metric": args.selection_metric,
        "validation_metrics": best,
        "num_templates": len(train),
        "template_worms": [w.name for w in train],
        "point_cloud_input": "full xyz",
        "supervision_mask_key": args.supervision_mask_key,
        "softmax_temperatures_scanned": softmax_temps,
        "rrf_k_scanned": rrf_ks,
        "tie_epsilon": args.tie_epsilon,
    }
    write_json(args.save_dir / "locked_config.json", locked)
    print("=" * 100)
    print("Locked validation-selected fDNC config")
    print(json.dumps(locked, indent=2))

    print("=" * 100)
    print("Evaluating locked fDNC configuration exactly once on test worms")
    test_pairwise = None
    if not args.skip_pairwise:
        test_pairwise, test_pair_rows = evaluate_pairwise(model, test, best_norm, device, cache_root)
        if test_pair_rows:
            with (args.save_dir / "test_pairwise_pairs.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(test_pair_rows[0].keys()))
                writer.writeheader(); writer.writerows(test_pair_rows)
    test_multi_all = evaluate_multitemplate(
        model, train, test, best_norm, {best["method"]}, device, cache_root,
        softmax_temps, rrf_ks, args.tie_epsilon,
        save_predictions=args.save_dir / "test_multitemplate_predictions.csv",
    )
    test_multi = test_multi_all[best["method"]]
    final = {
        "checkpoint": str(args.fdnc_checkpoint),
        "normalization": best_norm.name,
        "locked_method": best["method"],
        "num_train_templates": len(train),
        "num_val_worms": len(val),
        "num_test_worms": len(test),
        "point_cloud_input": "full xyz",
        "supervision_mask_key": args.supervision_mask_key,
        "pairwise_test": test_pairwise,
        "multitemplate_test": test_multi,
        "warning": "Configuration selected on val and evaluated once on test. Do not retune from test results.",
    }
    write_json(args.save_dir / "test_results_locked.json", final)
    print("=" * 100)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
