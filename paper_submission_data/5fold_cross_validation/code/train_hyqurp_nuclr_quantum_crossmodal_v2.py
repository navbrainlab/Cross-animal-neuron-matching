#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HyQuRP-inspired quantum position encoder + frozen NuCLR embeddings + standard Transformer.

This model intentionally does NOT use the fDNC PointNet, fDNC Transformer, or fDNC
checkpoint.  It keeps the strong experimental protocol/data utilities from the existing
repo, but the trainable architecture is:

    raw XYZ
      -> per-worm center / median-distance normalization
      -> HyQuRP-inspired 2-qubit singlet geometric encoder (or classical matched control)
      -> per-neuron position code

    frozen/precomputed NuCLR neuron embedding
      -> LayerNorm

    [position code ; NuCLR embedding]
      -> exact original Linear concat baseline
      -> optional bounded cross-modal interaction residual
         (gated MLP / low-rank bilinear / 8-qubit quantum AG fusion)
      -> learned worm/segment embedding
      -> ordinary torch.nn.TransformerEncoder over [worm A ; worm B]
      -> directional dot-product matcher + outlier logit

Important terminology
---------------------
This is "HyQuRP-inspired", not a verbatim implementation of the full HyQuRP point-cloud
classifier.  The original HyQuRP uses 2N qubits for N points and dual-equivariant
cross-pair quantum gates.  Here a fixed 2-qubit register is shared across neurons so the
model remains practical for worms with O(100) neurons.  The geometric encoding itself
uses the HyQuRP selective singlet idea and E(p)=exp(i p.sigma / Theta).

No relative-geometry bias is injected into Transformer attention.
The quantum cross-modal circuit acts neuron-wise on HyQuRP geometry x NuCLR activity;
it is distinct from the previous pairwise QGeo attention-bias component.
No additional domain/stability loss is used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

# Repo utilities live under engines/ in the NuCLR checkout.  Make this script
# runnable from the repo root without requiring PYTHONPATH edits.
_THIS_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _THIS_DIR / "engines"
for _path in (_THIS_DIR, _ENGINE_DIR):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Existing repo utilities only.  No fDNC model/checkpoint is imported or instantiated.
import train_fdnc_nuclr_fusion_benchmark as base
import train_fdnc_nuclr_quantum_relative_geometry_from_v2 as v2data


EPS = 1.0e-8


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--train_list", default="")
    p.add_argument("--val_list", default="")
    p.add_argument("--test_list", default="")
    p.add_argument("--external_test_list", default="")
    p.add_argument("--external_test_name", default="RLD_zero_shot")
    p.add_argument("--save_dir", type=Path, default=Path("runs/hyqurp_nuclr_transformer"))

    p.add_argument("--source_data_root", type=Path, default=None)
    p.add_argument("--xyz_key", default="xyz")
    p.add_argument("--activity_dim", type=int, default=256)

    p.add_argument(
        "--position_encoder",
        choices=["hyqurp", "mlp", "none"],
        default="hyqurp",
        help="hyqurp: quantum position encoder; mlp: matched classical control; none: activity-only.",
    )
    p.add_argument("--position_dim", type=int, default=32)
    p.add_argument("--quantum_theta", type=float, default=1.7)
    p.add_argument("--quantum_chunk", type=int, default=512)

    # Cross-modal fusion ablation.  `concat` exactly reproduces the original model.
    p.add_argument(
        "--fusion",
        choices=["concat", "gated_mlp", "bilinear", "quantum_crossmodal"],
        default="quantum_crossmodal",
    )
    p.add_argument("--fusion_alpha_max", type=float, default=0.25)
    p.add_argument("--classical_fusion_rank", type=int, default=8)
    p.add_argument("--qfusion_qubits", type=int, default=8)
    p.add_argument("--qfusion_layers", type=int, default=2)
    p.add_argument("--qfusion_chunk", type=int, default=256)
    p.add_argument(
        "--init_concat_checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional best.pt from the original concat model. Compatible weights are "
            "loaded before optimization; new fusion-adapter parameters remain freshly initialized."
        ),
    )

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--temperature", type=float, default=0.10)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--max_pairs_per_epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--outlier_weight", type=float, default=0.25)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--selection_metric", choices=["top1", "mrr", "assignment_top1"], default="top1")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_threads", type=int, default=4)
    p.add_argument("--smoke_test", action="store_true")

    return p.parse_args()


# =============================================================================
# Data
# =============================================================================


@dataclass
class WormRecord:
    worm_id: str
    embedding_path: str
    source_path: str
    xyz: Tensor          # [N,3], raw source coordinates
    nuclr_emb: Tensor    # [N,Da], frozen/precomputed NuCLR representation
    labels: Tensor       # [N], -1 means not part of source label vocabulary


def _scalar_text(data: Any, key: str) -> str:
    if key not in data.files:
        return ""
    raw = np.asarray(data[key])
    if raw.size == 0:
        return ""
    value = raw.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)


def _read_worm_id(data: Any, path: str) -> str:
    if "worm_id" in data.files:
        return _scalar_text(data, "worm_id")
    return Path(path).stem


def load_worm(
    path: str,
    label_to_int: Dict[Union[int, str], int],
    args: argparse.Namespace,
) -> WormRecord:
    embedding_path = Path(path)

    with np.load(embedding_path, allow_pickle=True) as data:
        required = {"nuclr_emb", "labels"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path}: missing required keys {sorted(missing)}")

        nuclr = np.asarray(data["nuclr_emb"], dtype=np.float32)
        raw_labels = np.asarray(data["labels"]).reshape(-1)
        worm_id = _read_worm_id(data, path)
        stored_source = _scalar_text(data, "source_path")

        embedded_xyz = (
            np.asarray(data[args.xyz_key], dtype=np.float32)
            if args.xyz_key in data.files
            else None
        )

    if nuclr.ndim != 2 or nuclr.shape[1] != args.activity_dim:
        raise ValueError(
            f"{path}: nuclr_emb must be [N,{args.activity_dim}], got {nuclr.shape}"
        )
    if nuclr.shape[0] != len(raw_labels):
        raise ValueError(
            f"{path}: nuclr rows={nuclr.shape[0]} labels={len(raw_labels)}"
        )
    if not np.isfinite(nuclr).all():
        raise ValueError(f"{path}: nuclr_emb contains NaN/Inf")

    source_path = ""
    if embedded_xyz is not None:
        xyz = embedded_xyz
        source_path = str(embedding_path.resolve())
    else:
        source = v2data.resolve_source_path(
            embedding_path,
            stored_source,
            args.source_data_root,
        )
        source_path = str(source)
        with np.load(source, allow_pickle=True) as raw:
            if args.xyz_key not in raw.files:
                raise KeyError(
                    f"{source}: missing xyz key {args.xyz_key!r}; available={raw.files}"
                )
            xyz = np.asarray(raw[args.xyz_key], dtype=np.float32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"{path}: xyz must be [N,3], got {xyz.shape}")
    if xyz.shape[0] != len(raw_labels):
        raise ValueError(
            f"{path}: xyz rows={xyz.shape[0]} labels={len(raw_labels)}"
        )
    if not np.isfinite(xyz).all():
        raise ValueError(f"{path}: xyz contains NaN/Inf")

    labels = np.full(len(raw_labels), -1, dtype=np.int64)
    for i, value in enumerate(raw_labels):
        token = base.normalize_label_token(value)
        if token is not None and token in label_to_int:
            labels[i] = label_to_int[token]

    return WormRecord(
        worm_id=worm_id,
        embedding_path=str(embedding_path.resolve()),
        source_path=source_path,
        xyz=torch.from_numpy(xyz),
        nuclr_emb=torch.from_numpy(nuclr),
        labels=torch.from_numpy(labels),
    )


def load_split(
    paths: Sequence[str],
    label_to_int: Dict[Union[int, str], int],
    args: argparse.Namespace,
) -> List[WormRecord]:
    records = [load_worm(path, label_to_int, args) for path in paths]
    ids = [r.worm_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate worm IDs: {ids}")
    return records


def validate_disjoint(*splits: Sequence[WormRecord]) -> None:
    sets = [set(r.worm_id for r in split) for split in splits]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sorted(sets[i] & sets[j])
            if overlap:
                raise ValueError(f"Worm leakage between split {i} and {j}: {overlap}")


# =============================================================================
# Geometry normalization
# =============================================================================


def normalize_worm_xyz(xyz: Tensor) -> Tensor:
    """Translation + robust scale normalization, independently for each worm.

    No target-domain population statistics are stored.  The normalization is computed
    independently from the current worm, which is appropriate for zero-shot evaluation.
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be [N,3], got {tuple(xyz.shape)}")

    centered = xyz - xyz.mean(dim=0, keepdim=True)
    if xyz.shape[0] >= 2:
        scale = torch.pdist(centered).median().clamp_min(1.0e-4)
    else:
        scale = centered.new_tensor(1.0)
    return centered / scale


# =============================================================================
# HyQuRP-inspired local quantum position encoder
# =============================================================================


class HyQuRPLocalPositionEncoder(nn.Module):
    """Fixed-register HyQuRP-inspired encoder for one neuron at a time.

    For each normalized 3-D point p:
      1) prepare two-qubit singlet |psi-> = (|01>-|10>)/sqrt(2)
      2) selectively encode p on the first qubit using
             E(p) = exp(i * p.sigma / Theta)
      3) apply one trainable isotropic Heisenberg interaction
             exp(-i * phi * (XX+YY+ZZ)/2)
      4) measure all nine two-qubit Pauli correlations
             <XX>, <XY>, ..., <ZZ>
      5) classical readout -> position_dim

    The same quantum circuit is shared by every neuron, so reordering neurons simply
    reorders the produced codes.  This is a practical local adaptation; it is not the
    full 2N-qubit HyQuRP architecture.
    """

    def __init__(
        self,
        out_dim: int = 32,
        theta_scale: float = 1.7,
        chunk_size: int = 512,
    ) -> None:
        super().__init__()
        if theta_scale <= 0:
            raise ValueError("theta_scale must be > 0")

        self.out_dim = int(out_dim)
        self.theta_scale = float(theta_scale)
        self.chunk_size = int(chunk_size)

        # One SU(2)-invariant trainable interaction for the local two-qubit register.
        self.heisenberg_phi = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        I = torch.eye(2, dtype=torch.complex64)
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

        self.register_buffer("I2", I, persistent=False)
        self.register_buffer("paulis", torch.stack([X, Y, Z], dim=0), persistent=False)

        H = torch.kron(X, X) + torch.kron(Y, Y) + torch.kron(Z, Z)
        self.register_buffer("heisenberg_H", H, persistent=False)

        corr_ops = []
        for A in (X, Y, Z):
            for B in (X, Y, Z):
                corr_ops.append(torch.kron(A, B))
        self.register_buffer("corr_ops", torch.stack(corr_ops, dim=0), persistent=False)

        singlet = torch.zeros(4, dtype=torch.complex64)
        singlet[1] = 1.0 / math.sqrt(2.0)
        singlet[2] = -1.0 / math.sqrt(2.0)
        self.register_buffer("singlet", singlet, persistent=False)

        self.readout = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def _encoding_unitary(self, p: Tensor) -> Tensor:
        """Closed-form exp(i p.sigma / Theta), returns [B,2,2]."""
        dtype = p.dtype
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        paulis = self.paulis.to(device=p.device, dtype=complex_dtype)
        I2 = self.I2.to(device=p.device, dtype=complex_dtype)

        radius = torch.linalg.norm(p, dim=-1)  # [B]
        direction = p / radius[:, None].clamp_min(EPS)
        direction = torch.where(
            (radius > EPS)[:, None],
            direction,
            torch.zeros_like(direction),
        )

        axis = torch.einsum("bc,cij->bij", direction.to(complex_dtype), paulis)
        angle = radius / self.theta_scale

        return (
            torch.cos(angle)[:, None, None].to(complex_dtype) * I2[None]
            + 1j * torch.sin(angle)[:, None, None].to(complex_dtype) * axis
        )

    def _forward_chunk(self, p: Tensor) -> Tensor:
        complex_dtype = torch.complex128 if p.dtype == torch.float64 else torch.complex64

        # [B,4] row representation of a column state.
        state = self.singlet.to(device=p.device, dtype=complex_dtype)[None].expand(p.shape[0], -1)

        # Apply E(p) on qubit 0.  Reshape amplitudes as A[q0,q1], so E acts by left multiplication.
        E = self._encoding_unitary(p)
        amp = state.reshape(-1, 2, 2)
        amp = torch.bmm(E, amp)
        state = amp.reshape(-1, 4)

        # Trainable isotropic Heisenberg interaction.  This commutes with global U⊗U rotations.
        H = self.heisenberg_H.to(device=p.device, dtype=complex_dtype)
        phi = self.heisenberg_phi.to(dtype=p.dtype)
        U = torch.matrix_exp((-0.5j * phi).to(complex_dtype) * H)
        state = torch.einsum("ij,bj->bi", U, state)

        # Nine pair correlations: [XX,XY,XZ,YX,...,ZZ].
        O = self.corr_ops.to(device=p.device, dtype=complex_dtype)
        obs = torch.einsum("bi,oij,bj->bo", state.conj(), O, state).real.to(dtype=p.dtype)
        return self.readout(obs)

    def forward(self, xyz: Tensor) -> Tensor:
        xyz = normalize_worm_xyz(xyz)
        outputs = []
        for start in range(0, xyz.shape[0], self.chunk_size):
            outputs.append(self._forward_chunk(xyz[start : start + self.chunk_size]))
        return torch.cat(outputs, dim=0) if outputs else xyz.new_zeros((0, self.out_dim))


class ClassicalPositionEncoder(nn.Module):
    """Parameter-light classical control receiving exactly the same normalized XYZ."""

    def __init__(self, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, xyz: Tensor) -> Tensor:
        return self.net(normalize_worm_xyz(xyz))


class ZeroPositionEncoder(nn.Module):
    def __init__(self, out_dim: int = 32) -> None:
        super().__init__()
        self.out_dim = int(out_dim)

    def forward(self, xyz: Tensor) -> Tensor:
        return xyz.new_zeros((xyz.shape[0], self.out_dim))




# =============================================================================
# Cross-modal fusion adapters
# =============================================================================


class BoundedResidualGate(nn.Module):
    """Scalar residual gate alpha = alpha_max * tanh(beta), initialized at zero."""

    def __init__(self, alpha_max: float = 0.25) -> None:
        super().__init__()
        if alpha_max <= 0:
            raise ValueError("fusion_alpha_max must be > 0")
        self.alpha_max = float(alpha_max)
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self) -> Tensor:
        return self.alpha_max * torch.tanh(self.beta)


class GatedMLPResidual(nn.Module):
    """Small classical nonlinear residual control on [geometry; activity]."""

    def __init__(self, position_dim: int, activity_dim: int, d_model: int, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("classical_fusion_rank must be >= 1")
        in_dim = int(position_dim + activity_dim)
        self.value = nn.Sequential(
            nn.Linear(in_dim, rank),
            nn.GELU(),
            nn.Linear(rank, d_model, bias=False),
        )
        self.gate = nn.Linear(in_dim, 1)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, position: Tensor, activity: Tensor) -> Tensor:
        x = torch.cat([position, activity], dim=-1)
        return torch.sigmoid(self.gate(x)) * self.value(x)


class BilinearResidual(nn.Module):
    """Low-rank classical multiplicative geometry x activity control."""

    def __init__(self, position_dim: int, activity_dim: int, d_model: int, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("classical_fusion_rank must be >= 1")
        self.activity = nn.Linear(activity_dim, rank, bias=False)
        self.position = nn.Linear(position_dim, rank, bias=False)
        self.out = nn.Linear(rank, d_model, bias=False)

    def forward(self, position: Tensor, activity: Tensor) -> Tensor:
        a = torch.tanh(self.activity(activity))
        g = torch.tanh(self.position(position))
        return self.out(a * g)


class QuantumCrossModalResidual(nn.Module):
    """Neuron-wise activity-geometry quantum interaction block.

    The register is split into two equal halves:
        qubits [0, ..., m-1]     : NuCLR activity latent
        qubits [m, ..., 2m-1]    : HyQuRP geometry latent

    Each variational layer performs data re-uploading and trainable local rotations,
    followed by *cross-modal* RZZ entanglers.  No neuron-neuron geometry is supplied;
    this block models the two modalities of the SAME neuron.

    Measurements contain local <Z> plus paired/shifted cross-register <ZZ>, making the
    returned feature explicitly sensitive to non-separable activity x geometry terms.
    """

    def __init__(
        self,
        position_dim: int,
        activity_dim: int,
        d_model: int,
        qubits: int = 8,
        layers: int = 2,
        chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if qubits < 4 or qubits % 2 != 0:
            raise ValueError("qfusion_qubits must be an even integer >= 4")
        if layers < 1:
            raise ValueError("qfusion_layers must be >= 1")
        if chunk_size < 1:
            raise ValueError("qfusion_chunk must be >= 1")

        self.qubits = int(qubits)
        self.half = self.qubits // 2
        self.layers = int(layers)
        self.chunk_size = int(chunk_size)
        self.state_dim = 1 << self.qubits

        # Classical compression only maps each modality into its own quantum register.
        self.activity_angles = nn.Linear(activity_dim, self.half)
        self.position_angles = nn.Linear(position_dim, self.half)

        # Data re-uploading scale/shift and trainable local rotations.
        self.data_scale = nn.Parameter(torch.ones(self.layers, self.qubits))
        self.data_shift = nn.Parameter(torch.zeros(self.layers, self.qubits))
        self.theta_ry = nn.Parameter(0.02 * torch.randn(self.layers, self.qubits))
        self.theta_rz = nn.Parameter(0.02 * torch.randn(self.layers, self.qubits))

        # Cross-modal entanglement only: A_k-G_k and A_k-G_{k+1}.
        self.rzz_paired = nn.Parameter(0.05 * torch.randn(self.layers, self.half))
        self.rzz_shifted = nn.Parameter(0.05 * torch.randn(self.layers, self.half))

        # local Z (2m) + paired AG ZZ (m) + shifted AG ZZ (m) = 4m = 2*qubits.
        self.measurement_dim = 2 * self.qubits
        self.readout = nn.Sequential(
            nn.LayerNorm(self.measurement_dim),
            nn.Linear(self.measurement_dim, d_model, bias=False),
        )

        basis = torch.arange(self.state_dim, dtype=torch.long)
        signs = []
        z_signs = []
        for q in range(self.qubits):
            bit = (basis >> (self.qubits - 1 - q)) & 1
            z = 1.0 - 2.0 * bit.float()
            z_signs.append(z)
            signs.append(z)
        for k in range(self.half):
            signs.append(z_signs[k] * z_signs[self.half + k])
        for k in range(self.half):
            signs.append(z_signs[k] * z_signs[self.half + ((k + 1) % self.half)])
        self.register_buffer("measurement_signs", torch.stack(signs, dim=0), persistent=False)

    def _pair_indices(self, qubit: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        idx = torch.arange(self.state_dim, device=device)
        mask = 1 << (self.qubits - 1 - qubit)
        idx0 = idx[(idx & mask) == 0]
        return idx0, idx0 | mask

    def _ry(self, state: Tensor, qubit: int, angle: Tensor) -> Tensor:
        idx0, idx1 = self._pair_indices(qubit, state.device)
        a = state.index_select(1, idx0)
        b = state.index_select(1, idx1)
        angle = angle.reshape(-1, 1).to(dtype=state.real.dtype)
        c = torch.cos(0.5 * angle).to(state.dtype)
        s = torch.sin(0.5 * angle).to(state.dtype)
        out = state.clone()
        out[:, idx0] = c * a - s * b
        out[:, idx1] = s * a + c * b
        return out

    def _rz(self, state: Tensor, qubit: int, angle: Tensor) -> Tensor:
        idx0, idx1 = self._pair_indices(qubit, state.device)
        angle = angle.reshape(-1, 1).to(dtype=state.real.dtype)
        p0 = torch.exp((-0.5j * angle).to(state.dtype))
        p1 = torch.exp((+0.5j * angle).to(state.dtype))
        out = state.clone()
        out[:, idx0] = state.index_select(1, idx0) * p0
        out[:, idx1] = state.index_select(1, idx1) * p1
        return out

    def _rzz(self, state: Tensor, q1: int, q2: int, angle: Tensor) -> Tensor:
        basis = torch.arange(self.state_dim, device=state.device)
        b1 = (basis >> (self.qubits - 1 - q1)) & 1
        b2 = (basis >> (self.qubits - 1 - q2)) & 1
        zz = (1.0 - 2.0 * b1.float()) * (1.0 - 2.0 * b2.float())
        angle = angle.reshape(-1, 1).to(dtype=state.real.dtype)
        phase = torch.exp((-0.5j * angle * zz[None]).to(state.dtype))
        return state * phase

    def _forward_chunk(self, position: Tensor, activity: Tensor) -> Tensor:
        dtype = activity.dtype
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        batch = activity.shape[0]

        a = math.pi * torch.tanh(self.activity_angles(activity))
        g = math.pi * torch.tanh(self.position_angles(position))
        data = torch.cat([a, g], dim=-1)

        state = torch.zeros(batch, self.state_dim, dtype=cdtype, device=activity.device)
        state[:, 0] = 1.0 + 0.0j

        for layer in range(self.layers):
            encoded = self.data_scale[layer][None] * data + self.data_shift[layer][None]
            for q in range(self.qubits):
                # Data re-uploading followed by local trainable rotations.
                state = self._ry(state, q, encoded[:, q])
                state = self._ry(state, q, self.theta_ry[layer, q].expand(batch))
                state = self._rz(state, q, self.theta_rz[layer, q].expand(batch))

            for k in range(self.half):
                state = self._rzz(
                    state,
                    k,
                    self.half + k,
                    self.rzz_paired[layer, k].expand(batch),
                )
            for k in range(self.half):
                state = self._rzz(
                    state,
                    k,
                    self.half + ((k + 1) % self.half),
                    self.rzz_shifted[layer, k].expand(batch),
                )

        prob = state.real.square() + state.imag.square()
        obs = prob @ self.measurement_signs.to(device=state.device, dtype=prob.dtype).transpose(0, 1)
        return self.readout(obs.to(dtype=dtype))

    def forward(self, position: Tensor, activity: Tensor) -> Tensor:
        if position.shape[0] != activity.shape[0]:
            raise ValueError("position/activity neuron count mismatch in quantum fusion")
        outputs = []
        for start in range(0, position.shape[0], self.chunk_size):
            outputs.append(
                self._forward_chunk(
                    position[start : start + self.chunk_size],
                    activity[start : start + self.chunk_size],
                )
            )
        return torch.cat(outputs, dim=0) if outputs else activity.new_zeros((0, self.readout[-1].out_features))


class CrossModalFusion(nn.Module):
    """Exact original concat baseline + optional bounded interaction residual."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.mode = str(args.fusion)
        self.position_dim = int(args.position_dim)
        self.activity_dim = int(args.activity_dim)
        self.d_model = int(args.d_model)

        # This is the exact fusion used by the original 57.37% HyQuRP model.
        self.base_projection = nn.Linear(
            self.position_dim + self.activity_dim,
            self.d_model,
            bias=False,
        )
        nn.init.xavier_uniform_(self.base_projection.weight)

        self.residual_gate = BoundedResidualGate(args.fusion_alpha_max)
        self.adapter: Optional[nn.Module]
        if self.mode == "concat":
            self.adapter = None
        elif self.mode == "gated_mlp":
            self.adapter = GatedMLPResidual(
                self.position_dim,
                self.activity_dim,
                self.d_model,
                int(args.classical_fusion_rank),
            )
        elif self.mode == "bilinear":
            self.adapter = BilinearResidual(
                self.position_dim,
                self.activity_dim,
                self.d_model,
                int(args.classical_fusion_rank),
            )
        elif self.mode == "quantum_crossmodal":
            self.adapter = QuantumCrossModalResidual(
                self.position_dim,
                self.activity_dim,
                self.d_model,
                qubits=int(args.qfusion_qubits),
                layers=int(args.qfusion_layers),
                chunk_size=int(args.qfusion_chunk),
            )
        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")

    def alpha(self) -> Tensor:
        if self.adapter is None:
            return self.base_projection.weight.new_tensor(0.0)
        return self.residual_gate()

    def forward(self, position: Tensor, activity: Tensor) -> Tensor:
        base = self.base_projection(torch.cat([position, activity], dim=-1))
        if self.adapter is None:
            return base
        residual = self.adapter(position, activity)
        return base + self.alpha() * residual


# =============================================================================
# HyQuRP + NuCLR + ordinary Transformer
# =============================================================================


class HyQuRPNuCLRTransformer(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.activity_dim = int(args.activity_dim)
        self.position_dim = int(args.position_dim)
        self.d_model = int(args.d_model)
        self.temperature = float(args.temperature)

        if args.position_encoder == "hyqurp":
            self.position_encoder = HyQuRPLocalPositionEncoder(
                out_dim=args.position_dim,
                theta_scale=args.quantum_theta,
                chunk_size=args.quantum_chunk,
            )
        elif args.position_encoder == "mlp":
            self.position_encoder = ClassicalPositionEncoder(args.position_dim)
        elif args.position_encoder == "none":
            self.position_encoder = ZeroPositionEncoder(args.position_dim)
        else:
            raise ValueError(args.position_encoder)

        self.activity_norm = nn.LayerNorm(self.activity_dim, elementwise_affine=False)
        self.position_norm = nn.LayerNorm(self.position_dim)

        # Keep the original concat projection as the base path, then optionally add
        # a bounded cross-modal interaction residual.  At alpha=0 the new model is
        # exactly the old HyQuRP + NuCLR concat fusion.
        self.cross_modal_fusion = CrossModalFusion(args)
        self.fusion_norm = nn.LayerNorm(self.d_model)

        # Only identifies which worm/role a neuron belongs to.  No neuron-index positional
        # encoding is added, so the Transformer stays permutation-equivariant within a worm.
        self.segment_embedding = nn.Parameter(torch.zeros(2, self.d_model))
        nn.init.normal_(self.segment_embedding, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=args.n_heads,
            dim_feedforward=args.ff_dim,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=args.n_layers)
        self.final_norm = nn.LayerNorm(self.d_model)

        self.query_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.outlier_head = nn.Linear(self.d_model, 1)

    def neuron_tokens(self, xyz: Tensor, activity: Tensor) -> Tensor:
        if activity.ndim != 2 or activity.shape[-1] != self.activity_dim:
            raise ValueError(
                f"activity must be [N,{self.activity_dim}], got {tuple(activity.shape)}"
            )
        if xyz.shape[0] != activity.shape[0]:
            raise ValueError("xyz/activity neuron count mismatch")

        qpos = self.position_norm(self.position_encoder(xyz))
        act = self.activity_norm(activity)
        token = self.cross_modal_fusion(qpos, act)
        return self.fusion_norm(token)

    def directional_logits_from_tokens(
        self,
        reference_tokens: Tensor,
        target_tokens: Tensor,
    ) -> Tensor:
        n_ref = reference_tokens.shape[0]
        n_tgt = target_tokens.shape[0]

        seq = torch.cat(
            [
                reference_tokens + self.segment_embedding[0],
                target_tokens + self.segment_embedding[1],
            ],
            dim=0,
        )

        encoded = self.transformer(seq.unsqueeze(0)).squeeze(0)
        encoded = self.final_norm(encoded)
        ref = encoded[:n_ref]
        tgt = encoded[n_ref : n_ref + n_tgt]

        # Directional target -> reference scores.
        q = F.normalize(self.query_proj(tgt), dim=-1)
        k = F.normalize(self.key_proj(ref), dim=-1)
        similarity = (q @ k.transpose(0, 1)) / max(self.temperature, 1.0e-4)
        outlier = self.outlier_head(tgt)
        return torch.cat([similarity, outlier], dim=-1)

    def score_pair(
        self,
        record_a: WormRecord,
        record_b: WormRecord,
        device: torch.device,
    ) -> Dict[str, Tensor]:
        xyz_a = record_a.xyz.to(device, non_blocking=True)
        xyz_b = record_b.xyz.to(device, non_blocking=True)
        act_a = record_a.nuclr_emb.to(device, non_blocking=True)
        act_b = record_b.nuclr_emb.to(device, non_blocking=True)

        # Position/activity encoding is worm-local and shared.
        token_a = self.neuron_tokens(xyz_a, act_a)
        token_b = self.neuron_tokens(xyz_b, act_b)

        # Two directional pair-conditioned passes, matching the existing protocol.
        logits_ba = self.directional_logits_from_tokens(token_a, token_b)
        logits_ab = self.directional_logits_from_tokens(token_b, token_a)

        return {
            "logits_ba": logits_ba,
            "logits_ab": logits_ab,
            "tokens_a": token_a,
            "tokens_b": token_b,
        }


# =============================================================================
# Matching objective
# =============================================================================


def unique_label_map(labels: Tensor) -> Dict[int, int]:
    positions: Dict[int, List[int]] = {}
    for index, value in enumerate(labels.detach().cpu().tolist()):
        value = int(value)
        if value >= 0:
            positions.setdefault(value, []).append(index)
    return {label: rows[0] for label, rows in positions.items() if len(rows) == 1}


def directed_matching_loss(
    logits: Tensor,
    query_labels: Tensor,
    reference_labels: Tensor,
    outlier_weight: float,
) -> Tuple[Tensor, Dict[str, int]]:
    reference_map = unique_label_map(reference_labels)
    outlier_index = logits.shape[1] - 1

    query_rows: List[int] = []
    targets: List[int] = []
    weights: List[float] = []
    shared = 0

    for row, value in enumerate(query_labels.detach().cpu().tolist()):
        identity = int(value)
        if identity < 0:
            continue
        target = reference_map.get(identity, outlier_index)
        query_rows.append(row)
        targets.append(target)
        is_shared = target != outlier_index
        weights.append(1.0 if is_shared else float(outlier_weight))
        shared += int(is_shared)

    if not query_rows:
        return logits.sum() * 0.0, {"queries": 0, "shared": 0}

    rows_t = torch.tensor(query_rows, dtype=torch.long, device=logits.device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=logits.device)
    weights_t = torch.tensor(weights, dtype=logits.dtype, device=logits.device)

    losses = F.cross_entropy(logits.index_select(0, rows_t), targets_t, reduction="none")
    loss = (losses * weights_t).sum() / weights_t.sum().clamp_min(EPS)
    return loss, {"queries": len(query_rows), "shared": shared}


def bidirectional_matching_loss(
    logits_ba: Tensor,
    logits_ab: Tensor,
    labels_a: Tensor,
    labels_b: Tensor,
    outlier_weight: float,
) -> Tuple[Tensor, Dict[str, int]]:
    loss_ba, info_ba = directed_matching_loss(
        logits_ba, labels_b, labels_a, outlier_weight
    )
    loss_ab, info_ab = directed_matching_loss(
        logits_ab, labels_a, labels_b, outlier_weight
    )
    return 0.5 * (loss_ba + loss_ab), {
        "queries": info_ba["queries"] + info_ab["queries"],
        "shared": info_ba["shared"] + info_ab["shared"],
    }


# =============================================================================
# Evaluation
# =============================================================================


def add_directional_metrics(
    accumulator: Any,
    logits_ba: Tensor,
    logits_ab: Tensor,
    labels_a: Tensor,
    labels_b: Tensor,
) -> None:
    scores_ba = logits_ba[:, :-1]
    scores_ab = logits_ab[:, :-1]

    accumulator.add_ranks(base.compute_ranks(scores_ba, labels_b, labels_a))
    accumulator.add_ranks(base.compute_ranks(scores_ab, labels_a, labels_b))

    hits_ba, queries_ba = base.hungarian_query_accuracy(
        scores_ba, labels_b, labels_a
    )
    hits_ab, queries_ab = base.hungarian_query_accuracy(
        scores_ab, labels_a, labels_b
    )
    accumulator.add_assignment(hits_ba + hits_ab, queries_ba + queries_ab)


@torch.no_grad()
def evaluate(
    model: HyQuRPNuCLRTransformer,
    records: Sequence[WormRecord],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    accumulator = base.MetricAccumulator()

    for ia, ib in base.all_unordered_pairs(len(records)):
        a, b = records[ia], records[ib]
        scores = model.score_pair(a, b, device)
        add_directional_metrics(
            accumulator,
            scores["logits_ba"].float(),
            scores["logits_ab"].float(),
            a.labels.to(device),
            b.labels.to(device),
        )

    return accumulator.to_dict()


# =============================================================================
# Train / checkpoint
# =============================================================================


def linear_warmup(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / float(warmup_epochs))


def train_one_epoch(
    model: HyQuRPNuCLRTransformer,
    optimizer: torch.optim.Optimizer,
    records: Sequence[WormRecord],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()

    pairs = base.all_unordered_pairs(len(records))
    random.shuffle(pairs)
    if args.max_pairs_per_epoch > 0:
        if len(pairs) >= args.max_pairs_per_epoch:
            pairs = pairs[: args.max_pairs_per_epoch]
        else:
            # Repeat shuffled pair cycles so max_pairs_per_epoch is a real step budget.
            original = list(pairs)
            while len(pairs) < args.max_pairs_per_epoch:
                extra = list(original)
                random.shuffle(extra)
                pairs.extend(extra)
            pairs = pairs[: args.max_pairs_per_epoch]

    losses: List[float] = []
    shared_counts: List[float] = []

    for ia, ib in pairs:
        a, b = records[ia], records[ib]
        labels_a = a.labels.to(device)
        labels_b = b.labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        scores = model.score_pair(a, b, device)
        loss, info = bidirectional_matching_loss(
            scores["logits_ba"],
            scores["logits_ab"],
            labels_a,
            labels_b,
            args.outlier_weight,
        )

        if info["queries"] == 0:
            continue
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss for {a.worm_id}/{b.worm_id}")

        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        shared_counts.append(float(info["shared"]))

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "shared": float(np.mean(shared_counts)) if shared_counts else 0.0,
        "steps": float(len(losses)),
    }


def save_checkpoint(
    path: Path,
    model: HyQuRPNuCLRTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    validation: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "validation": validation,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "architecture": "HyQuRP-inspired local QPE + NuCLR concat + standard joint Transformer",
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: HyQuRPNuCLRTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def metric_value(metrics: Dict[str, float], name: str) -> float:
    aliases = {
        "top1": ("top1", "ranking_top1"),
        "mrr": ("mrr",),
        "assignment_top1": ("assignment_top1", "hungarian"),
    }
    for key in aliases[name]:
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"Metric {name!r} not found in {sorted(metrics)}")




def load_compatible_concat_checkpoint(path: Path, model: HyQuRPNuCLRTransformer) -> Dict[str, Any]:
    """Load an original concat checkpoint into the new residual-fusion model.

    Old key `fusion_projection.weight` is remapped to
    `cross_modal_fusion.base_projection.weight`.  New adapter/gate parameters are left
    freshly initialized.  This is useful for a clean residual-adapter experiment on top
    of the already selected concat baseline.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("model_state", checkpoint)
    state = {}
    for key, value in raw.items():
        key = str(key)
        if key.startswith("module."):
            key = key[7:]
        if key == "fusion_projection.weight":
            key = "cross_modal_fusion.base_projection.weight"
        state[key] = value

    result = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = (
        "cross_modal_fusion.residual_gate.",
        "cross_modal_fusion.adapter.",
    )
    bad_missing = [
        key for key in result.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            "Incompatible concat checkpoint. "
            f"unexpected={result.unexpected_keys}, bad_missing={bad_missing}"
        )
    return {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_metric": checkpoint.get("best_metric"),
        "missing_new_keys": list(result.missing_keys),
    }


# =============================================================================
# Smoke test
# =============================================================================


def smoke_test(args: argparse.Namespace) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    args = argparse.Namespace(**vars(args))
    args.dropout = 0.0
    args.position_encoder = "hyqurp"
    args.fusion = "quantum_crossmodal"
    args.activity_dim = 256
    args.position_dim = 32
    args.d_model = 128
    args.n_heads = 4
    args.n_layers = 2
    args.ff_dim = 256
    args.qfusion_qubits = 8
    args.qfusion_layers = 2
    args.qfusion_chunk = 64
    args.fusion_alpha_max = 0.25

    model = HyQuRPNuCLRTransformer(args).to(device)
    model.eval()

    na, nb = 8, 9
    xyz_a = torch.randn(na, 3)
    xyz_b = torch.randn(nb, 3)
    act_a = torch.randn(na, args.activity_dim)
    act_b = torch.randn(nb, args.activity_dim)
    labels_a = torch.arange(na)
    labels_b = torch.cat([torch.arange(na), torch.tensor([99])])

    a = WormRecord("A", "", "", xyz_a, act_a, labels_a)
    b = WormRecord("B", "", "", xyz_b, act_b, labels_b)

    # alpha=0 must exactly reproduce the original concat fusion path.
    with torch.no_grad():
        xyz_d = xyz_a.to(device)
        act_d = act_a.to(device)
        qpos = model.position_norm(model.position_encoder(xyz_d))
        actn = model.activity_norm(act_d)
        manual_base = model.fusion_norm(
            model.cross_modal_fusion.base_projection(torch.cat([qpos, actn], dim=-1))
        )
        actual = model.neuron_tokens(xyz_d, act_d)
        baseline_error = float((manual_base - actual).abs().max().cpu())

    out = model.score_pair(a, b, device)
    assert out["logits_ba"].shape == (nb, na + 1)
    assert out["logits_ab"].shape == (na, nb + 1)

    # First backward: gate should receive gradient even though alpha starts at zero.
    model.train()
    out_train = model.score_pair(a, b, device)
    loss, _ = bidirectional_matching_loss(
        out_train["logits_ba"], out_train["logits_ab"],
        labels_a.to(device), labels_b.to(device), 0.25,
    )
    loss.backward()
    gate_grad = float(model.cross_modal_fusion.residual_gate.beta.grad.abs().cpu())

    # Turn on a tiny residual only for the diagnostic, then verify quantum-circuit gradients.
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        target_alpha = 0.05
        ratio = min(0.999, target_alpha / args.fusion_alpha_max)
        model.cross_modal_fusion.residual_gate.beta.fill_(math.atanh(ratio))
    out_train = model.score_pair(a, b, device)
    loss2, _ = bidirectional_matching_loss(
        out_train["logits_ba"], out_train["logits_ab"],
        labels_a.to(device), labels_b.to(device), 0.25,
    )
    loss2.backward()
    qadapter = model.cross_modal_fusion.adapter
    assert isinstance(qadapter, QuantumCrossModalResidual)
    rzz_grad = float(
        qadapter.rzz_paired.grad.abs().sum().cpu()
        + qadapter.rzz_shifted.grad.abs().sum().cpu()
    )
    angle_grad = float(
        qadapter.activity_angles.weight.grad.abs().sum().cpu()
        + qadapter.position_angles.weight.grad.abs().sum().cpu()
    )

    # Permutation equivariance test for neuron ordering inside worm A.
    model.eval()
    perm = torch.randperm(na)
    inv = torch.argsort(perm)
    a_perm = WormRecord("A", "", "", xyz_a[perm], act_a[perm], labels_a[perm])
    with torch.no_grad():
        ref = model.score_pair(a, b, device)["logits_ba"][:, :-1].cpu()
        changed = model.score_pair(a_perm, b, device)["logits_ba"][:, :-1].cpu()[:, inv]
        perm_error = float((ref - changed).abs().max())

        a_transform = WormRecord(
            "A", "", "",
            2.7 * xyz_a + torch.tensor([4.0, -3.0, 1.5]),
            act_a,
            labels_a,
        )
        transformed = model.score_pair(a_transform, b, device)["logits_ba"][:, :-1].cpu()
        norm_error = float((ref - transformed).abs().max())

    print("HyQuRP + NuCLR + Quantum Cross-Modal Fusion + Transformer smoke test passed")
    print("  logits_ba shape:", tuple(out["logits_ba"].shape))
    print("  logits_ab shape:", tuple(out["logits_ab"].shape))
    print(f"  alpha=0 concat-baseline max error: {baseline_error:.6e}")
    print(f"  residual gate grad at alpha=0:      {gate_grad:.6e}")
    print(f"  cross-modal RZZ grad sum:           {rzz_grad:.6e}")
    print(f"  modality-angle grad sum:            {angle_grad:.6e}")
    print(f"  max neuron-permutation error:       {perm_error:.6e}")
    print(f"  max translation/scale error:        {norm_error:.6e}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.num_threads))
    base.set_seed(args.seed)

    if args.smoke_test:
        smoke_test(args)
        return

    if not args.train_list or not args.val_list:
        raise ValueError("--train_list and --val_list are required unless --smoke_test is used")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = requested

    # IMPORTANT: source label vocabulary is constructed from source train+val only.
    # Test/external target data are not opened before checkpoint selection.
    train_paths = base.read_list_file(args.train_list)
    val_paths = base.read_list_file(args.val_list)
    label_to_int = base.collect_label_mapping(train_paths + val_paths)

    train_records = load_split(train_paths, label_to_int, args)
    val_records = load_split(val_paths, label_to_int, args)
    validate_disjoint(train_records, val_records)

    model = HyQuRPNuCLRTransformer(args).to(device)
    init_info = None
    if args.init_concat_checkpoint is not None:
        init_info = load_compatible_concat_checkpoint(args.init_concat_checkpoint, model)
        print("Loaded compatible concat checkpoint:", json.dumps(init_info, ensure_ascii=False))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    parameter_info = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "hyqurp_quantum": sum(
            p.numel()
            for name, p in model.named_parameters()
            if "position_encoder" in name and args.position_encoder == "hyqurp"
        ),
        "fusion_adapter": sum(
            p.numel()
            for name, p in model.named_parameters()
            if "cross_modal_fusion.adapter" in name
        ),
        "fusion_quantum": sum(
            p.numel()
            for name, p in model.named_parameters()
            if "cross_modal_fusion.adapter" in name and args.fusion == "quantum_crossmodal"
        ),
    }

    config = {
        "architecture": "HyQuRP-inspired position encoder + NuCLR + cross-modal fusion + standard Transformer",
        "position_encoder": args.position_encoder,
        "fusion": args.fusion,
        "init_concat_checkpoint": None if args.init_concat_checkpoint is None else str(args.init_concat_checkpoint),
        "init_info": init_info,
        "uses_fdnc_model": False,
        "uses_relative_geometry_attention_bias": False,
        "uses_pair_cotransformer": False,
        "uses_extra_domain_loss": False,
        "source_train_worms": [r.worm_id for r in train_records],
        "source_val_worms": [r.worm_id for r in val_records],
        "parameters": parameter_info,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.save_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 110)
    print("HyQuRP + NuCLR + Cross-Modal Fusion + Standard Transformer")
    print("device:", device)
    print("position encoder:", args.position_encoder)
    print("fusion:", args.fusion)
    print("initial fusion alpha:", float(model.cross_modal_fusion.alpha().detach().cpu()))
    print("train/val worms:", len(train_records), len(val_records))
    print("parameters:", json.dumps(parameter_info))
    print("No fDNC model/checkpoint is used.")
    print("Test/external splits remain unopened until validation selects best.pt.")
    print("=" * 110)

    initial_val = evaluate(model, val_records, device)
    best_metric = metric_value(initial_val, args.selection_metric)
    best_epoch = 0
    save_checkpoint(
        args.save_dir / "best.pt",
        model,
        optimizer,
        0,
        best_metric,
        initial_val,
        args,
    )

    history: List[Dict[str, Any]] = []
    stale = 0

    for epoch in range(1, args.epochs + 1):
        factor = linear_warmup(epoch, args.warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor

        train_stats = train_one_epoch(
            model, optimizer, train_records, device, args
        )
        validation = evaluate(model, val_records, device)
        selected = metric_value(validation, args.selection_metric)

        row = {
            "epoch": epoch,
            "train": train_stats,
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if selected > best_metric:
            best_metric = selected
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                args.save_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_metric,
                validation,
                args,
            )
        else:
            stale += 1

        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    (args.save_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Lock best source-validation checkpoint before opening test domains.
    checkpoint = load_checkpoint(args.save_dir / "best.pt", model)
    model.to(device)
    model.eval()

    summary: Dict[str, Any] = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_metric": float(checkpoint["best_metric"]),
        "validation": checkpoint["validation"],
        "position_encoder": args.position_encoder,
        "fusion": args.fusion,
        "final_fusion_alpha": float(model.cross_modal_fusion.alpha().detach().cpu()),
        "architecture": "HyQuRP-inspired position encoder + NuCLR + cross-modal fusion + standard Transformer",
    }

    if args.test_list:
        test_paths = base.read_list_file(args.test_list)
        test_records = load_split(test_paths, label_to_int, args)
        validate_disjoint(train_records, val_records, test_records)
        summary["test"] = evaluate(model, test_records, device)
        summary["test_worms"] = [r.worm_id for r in test_records]

    if args.external_test_list:
        external_paths = base.read_list_file(args.external_test_list)
        external_records = load_split(external_paths, label_to_int, args)
        # No overlap is expected for true cross-dataset zero-shot evaluation.
        validate_disjoint(train_records, val_records, external_records)
        summary[args.external_test_name] = evaluate(model, external_records, device)
        summary[f"{args.external_test_name}_worms"] = [r.worm_id for r in external_records]

    (args.save_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
