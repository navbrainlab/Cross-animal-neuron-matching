#!/usr/bin/env python3
"""Ambiguity-Aware Geometry-Activity Matching Transformer.

The two modalities first produce independent matching distributions.  Per-query
ambiguity/agreement statistics then control complementary fusion, followed by a
population Transformer and a differentiable one-to-one balancing correction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "engines"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hyqurp_nuclr_quantum_crossmodal_v2 as legacy


WormRecord = legacy.WormRecord
EPS = 1.0e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-list", required=True)
    p.add_argument("--val-list", required=True)
    p.add_argument("--test-list", default="")
    p.add_argument("--source-data-root", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--xyz-key", default="xyz")
    p.add_argument("--activity-dim", type=int, default=256)
    p.add_argument("--position-dim", type=int, default=32)
    p.add_argument("--quantum-theta", type=float, default=1.7)
    p.add_argument("--quantum-chunk", type=int, default=512)
    p.add_argument("--evidence-dim", type=int, default=64)
    p.add_argument("--ambiguity-hidden", type=int, default=64)
    p.add_argument("--ambiguity-topk", type=int, default=5)
    p.add_argument("--evidence-temperature", type=float, default=0.10)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--ff-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--sinkhorn-iters", type=int, default=3)
    p.add_argument("--sinkhorn-max-strength", type=float, default=0.25)
    p.add_argument("--disable-stage2", action="store_true")
    p.add_argument("--disable-sinkhorn", action="store_true")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--max-pairs-per-epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--outlier-weight", type=float, default=0.25)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--selection-metric", choices=["top1", "mrr", "assignment_top1"], default="top1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


class AmbiguityAwarePairFusion(nn.Module):
    """Pair-conditioned fusion driven by explicit modality uncertainty statistics."""

    feature_names = (
        "geometry_entropy",
        "activity_entropy",
        "geometry_margin",
        "activity_margin",
        "distribution_agreement",
        "js_disagreement",
        "topk_overlap",
        "joint_ambiguity",
    )

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.topk = int(args.ambiguity_topk)
        self.temperature = float(args.evidence_temperature)
        evidence_dim = int(args.evidence_dim)
        d_model = int(args.d_model)
        hidden = int(args.ambiguity_hidden)

        self.geometry_evidence = nn.Sequential(
            nn.Linear(args.position_dim, evidence_dim, bias=False),
            nn.LayerNorm(evidence_dim),
        )
        self.activity_evidence = nn.Sequential(
            nn.Linear(args.activity_dim, evidence_dim, bias=False),
            nn.LayerNorm(evidence_dim),
        )
        self.geometry_token = nn.Linear(args.position_dim, d_model, bias=False)
        self.activity_token = nn.Linear(args.activity_dim, d_model, bias=False)
        self.gate_residual = nn.Sequential(
            nn.Linear(len(self.feature_names), hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.ambiguity_token = nn.Sequential(
            nn.Linear(len(self.feature_names), hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
        )
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _normalized_entropy(p: Tensor) -> Tensor:
        denom = math.log(max(2, p.shape[-1]))
        return -(p * p.clamp_min(EPS).log()).sum(dim=-1) / denom

    def _features(self, qg: Tensor, qa: Tensor, rg: Tensor, ra: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        scale = max(self.temperature, 1.0e-4)
        logits_g = F.normalize(qg, dim=-1) @ F.normalize(rg, dim=-1).transpose(0, 1) / scale
        logits_a = F.normalize(qa, dim=-1) @ F.normalize(ra, dim=-1).transpose(0, 1) / scale
        pg, pa = logits_g.softmax(dim=-1), logits_a.softmax(dim=-1)
        hg, ha = self._normalized_entropy(pg), self._normalized_entropy(pa)

        kg = min(2, pg.shape[-1])
        top_g = pg.topk(kg, dim=-1).values
        top_a = pa.topk(kg, dim=-1).values
        mg = top_g[:, 0] - (top_g[:, 1] if kg == 2 else 0.0)
        ma = top_a[:, 0] - (top_a[:, 1] if kg == 2 else 0.0)

        agreement = torch.sqrt((pg * pa).clamp_min(EPS)).sum(dim=-1)
        mixture = 0.5 * (pg + pa)
        js = 0.5 * (
            (pg * (pg.clamp_min(EPS).log() - mixture.clamp_min(EPS).log())).sum(dim=-1)
            + (pa * (pa.clamp_min(EPS).log() - mixture.clamp_min(EPS).log())).sum(dim=-1)
        ) / math.log(2.0)

        k = min(self.topk, pg.shape[-1])
        idx_g = logits_g.topk(k, dim=-1).indices
        idx_a = logits_a.topk(k, dim=-1).indices
        overlap = (idx_g[:, :, None] == idx_a[:, None, :]).any(dim=-1).float().mean(dim=-1)
        joint = hg * ha
        features = torch.stack((hg, ha, mg, ma, agreement, js, overlap, joint), dim=-1)
        return features, logits_g, logits_a

    def directional_tokens(
        self,
        query_geometry: Tensor,
        query_activity: Tensor,
        reference_geometry: Tensor,
        reference_activity: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        qg = self.geometry_evidence(query_geometry)
        qa = self.activity_evidence(query_activity)
        rg = self.geometry_evidence(reference_geometry)
        ra = self.activity_evidence(reference_activity)
        features, logits_g, logits_a = self._features(qg, qa, rg, ra)

        # Confidence prior gives the intended rescue behavior before learning;
        # the MLP learns dataset-specific corrections from all ambiguity signals.
        confidence_prior = 2.0 * torch.stack((1.0 - features[:, 0], 1.0 - features[:, 1]), dim=-1)
        weights = (confidence_prior + self.gate_residual(features)).softmax(dim=-1)
        token = (
            weights[:, :1] * self.geometry_token(query_geometry)
            + weights[:, 1:] * self.activity_token(query_activity)
            + self.ambiguity_token(features)
        )
        diagnostics = {
            "features": features,
            "weights": weights,
            "geometry_logits": logits_g,
            "activity_logits": logits_a,
        }
        return self.output_norm(token), diagnostics

    def forward(
        self,
        geometry_a: Tensor,
        activity_a: Tensor,
        geometry_b: Tensor,
        activity_b: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        token_a, diag_a = self.directional_tokens(
            geometry_a, activity_a, geometry_b, activity_b
        )
        token_b, diag_b = self.directional_tokens(
            geometry_b, activity_b, geometry_a, activity_a
        )
        return token_a, token_b, {"a": diag_a, "b": diag_b}


class AmbiguityAwareGeometryActivityTransformer(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.activity_dim = int(args.activity_dim)
        self.d_model = int(args.d_model)
        self.temperature = float(args.temperature)
        self.sinkhorn_iters = int(args.sinkhorn_iters)
        self.sinkhorn_max_strength = float(args.sinkhorn_max_strength)
        self.stage2_enabled = not bool(getattr(args, "disable_stage2", False))
        self.sinkhorn_enabled = not bool(getattr(args, "disable_sinkhorn", False))

        self.position_encoder = legacy.HyQuRPLocalPositionEncoder(
            out_dim=args.position_dim,
            theta_scale=args.quantum_theta,
            chunk_size=args.quantum_chunk,
        )
        self.position_norm = nn.LayerNorm(args.position_dim)
        self.activity_norm = nn.LayerNorm(args.activity_dim, elementwise_affine=False)
        self.ambiguity_fusion = AmbiguityAwarePairFusion(args)

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
        self.sinkhorn_strength_logit = nn.Parameter(torch.tensor(-2.0))

    def independent_evidence(self, xyz: Tensor, activity: Tensor) -> Tuple[Tensor, Tensor]:
        if xyz.shape[0] != activity.shape[0] or activity.shape[-1] != self.activity_dim:
            raise ValueError("Invalid geometry/activity shapes")
        return self.position_norm(self.position_encoder(xyz)), self.activity_norm(activity)

    def _sinkhorn(self, scores: Tensor) -> Tensor:
        logp = scores
        for _ in range(self.sinkhorn_iters):
            logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
            logp = logp - torch.logsumexp(logp, dim=0, keepdim=True)
        return logp

    def population_scores(self, token_a: Tensor, token_b: Tensor) -> Tuple[Tensor, Tensor]:
        na, nb = token_a.shape[0], token_b.shape[0]
        seq = torch.cat(
            (token_a + self.segment_embedding[0], token_b + self.segment_embedding[1]), dim=0
        )
        encoded = self.final_norm(self.transformer(seq.unsqueeze(0)).squeeze(0))
        a, b = encoded[:na], encoded[na : na + nb]
        qa, qb = F.normalize(self.query_proj(a), dim=-1), F.normalize(self.query_proj(b), dim=-1)
        ka, kb = F.normalize(self.key_proj(a), dim=-1), F.normalize(self.key_proj(b), dim=-1)
        score_ba = qb @ ka.transpose(0, 1) / max(self.temperature, 1.0e-4)
        score_ab = qa @ kb.transpose(0, 1) / max(self.temperature, 1.0e-4)
        mutual = 0.5 * (score_ba + score_ab.transpose(0, 1))

        if self.sinkhorn_enabled:
            balanced = self._sinkhorn(mutual)
            balanced = balanced - balanced.mean(dim=1, keepdim=True)
            strength = self.sinkhorn_max_strength * torch.sigmoid(self.sinkhorn_strength_logit)
            mutual = mutual + strength * balanced
        logits_ba = torch.cat((mutual, self.outlier_head(b)), dim=-1)
        logits_ab = torch.cat((mutual.transpose(0, 1), self.outlier_head(a)), dim=-1)
        return logits_ba, logits_ab

    def score_pair(self, record_a: WormRecord, record_b: WormRecord, device: torch.device) -> Dict[str, Tensor]:
        ga, aa = self.independent_evidence(record_a.xyz.to(device), record_a.nuclr_emb.to(device))
        gb, ab = self.independent_evidence(record_b.xyz.to(device), record_b.nuclr_emb.to(device))
        if self.stage2_enabled:
            token_a, token_b, diagnostics = self.ambiguity_fusion(ga, aa, gb, ab)
        else:
            token_a = self.ambiguity_fusion.output_norm(
                0.5 * self.ambiguity_fusion.geometry_token(ga)
                + 0.5 * self.ambiguity_fusion.activity_token(aa)
            )
            token_b = self.ambiguity_fusion.output_norm(
                0.5 * self.ambiguity_fusion.geometry_token(gb)
                + 0.5 * self.ambiguity_fusion.activity_token(ab)
            )
            diagnostics = {
                "a": {
                    "features": ga.new_zeros((ga.shape[0], len(AmbiguityAwarePairFusion.feature_names))),
                    "weights": ga.new_full((ga.shape[0], 2), 0.5),
                },
                "b": {
                    "features": gb.new_zeros((gb.shape[0], len(AmbiguityAwarePairFusion.feature_names))),
                    "weights": gb.new_full((gb.shape[0], 2), 0.5),
                },
            }
        logits_ba, logits_ab = self.population_scores(token_a, token_b)
        return {
            "logits_ba": logits_ba,
            "logits_ab": logits_ab,
            "tokens_a": token_a,
            "tokens_b": token_b,
            "diagnostics": diagnostics,
        }


def evaluate(model: nn.Module, records: Sequence[WormRecord], device: torch.device) -> Dict[str, float]:
    return legacy.evaluate(model, records, device)


@torch.no_grad()
def ambiguity_diagnostics(model: AmbiguityAwareGeometryActivityTransformer, records: Sequence[WormRecord], device: torch.device) -> Dict[str, float]:
    model.eval()
    feature_sum = torch.zeros(len(AmbiguityAwarePairFusion.feature_names), dtype=torch.float64)
    weight_sum = torch.zeros(2, dtype=torch.float64)
    count = 0
    for ia, ib in legacy.base.all_unordered_pairs(len(records)):
        output = model.score_pair(records[ia], records[ib], device)
        for direction in ("a", "b"):
            diag = output["diagnostics"][direction]
            n = diag["features"].shape[0]
            feature_sum += diag["features"].double().sum(dim=0).cpu()
            weight_sum += diag["weights"].double().sum(dim=0).cpu()
            count += n
    result = {name: float(value / max(count, 1)) for name, value in zip(AmbiguityAwarePairFusion.feature_names, feature_sum)}
    result.update({
        "mean_geometry_weight": float(weight_sum[0] / max(count, 1)),
        "mean_activity_weight": float(weight_sum[1] / max(count, 1)),
        "stage2_enabled": bool(model.stage2_enabled),
        "sinkhorn_enabled": bool(model.sinkhorn_enabled),
        "sinkhorn_strength": (
            float(model.sinkhorn_max_strength * torch.sigmoid(model.sinkhorn_strength_logit).cpu())
            if model.sinkhorn_enabled else 0.0
        ),
        "neuron_pair_contexts": int(count),
    })
    return result


def metric_value(metrics: Dict[str, float], name: str) -> float:
    return legacy.metric_value(metrics, name)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float, validation: Dict[str, float], args: argparse.Namespace) -> None:
    torch.save({
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "validation": validation,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "architecture": "Ambiguity-Aware Geometry-Activity Matching Transformer",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }, path)


def smoke_test(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = AmbiguityAwareGeometryActivityTransformer(args).to(device)
    a = WormRecord("a", "", "", torch.randn(8, 3), torch.randn(8, args.activity_dim), torch.arange(8))
    b = WormRecord("b", "", "", torch.randn(9, 3), torch.randn(9, args.activity_dim), torch.cat((torch.arange(8), torch.tensor([99]))))
    output = model.score_pair(a, b, device)
    assert output["logits_ba"].shape == (9, 9)
    assert output["logits_ab"].shape == (8, 10)
    loss, _ = legacy.bidirectional_matching_loss(
        output["logits_ba"], output["logits_ab"], a.labels, b.labels, args.outlier_weight
    )
    loss.backward()
    assert model.position_encoder.heisenberg_phi.grad is not None
    if model.stage2_enabled:
        assert model.ambiguity_fusion.gate_residual[0].weight.grad is not None
    else:
        assert model.ambiguity_fusion.geometry_token.weight.grad is not None
    print(json.dumps({"smoke_test": "passed", "loss": float(loss.detach()), "parameters": sum(p.numel() for p in model.parameters())}))


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.num_threads))
    legacy.base.set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.smoke_test:
        smoke_test(args)
        return

    args.save_dir.mkdir(parents=True, exist_ok=True)
    train_paths = legacy.base.read_list_file(args.train_list)
    val_paths = legacy.base.read_list_file(args.val_list)
    label_to_int = legacy.base.collect_label_mapping(train_paths + val_paths)
    train_records = legacy.load_split(train_paths, label_to_int, args)
    val_records = legacy.load_split(val_paths, label_to_int, args)
    legacy.validate_disjoint(train_records, val_records)

    model = AmbiguityAwareGeometryActivityTransformer(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = {
        "architecture": "Ambiguity-Aware Geometry-Activity Matching Transformer",
        "compact_nuclr": "T1/ST1 D256 (precomputed and frozen)",
        "stage1": "independent HyQuRP geometry and Compact NuCLR activity evidence",
        "stage2_features": list(AmbiguityAwarePairFusion.feature_names),
        "stage3": "population Transformer + mutual scoring + learnable Sinkhorn correction",
        "stage2_enabled": not args.disable_stage2,
        "sinkhorn_enabled": not args.disable_sinkhorn,
        "source_train_worms": [record.worm_id for record in train_records],
        "source_val_worms": [record.worm_id for record in val_records],
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    initial = evaluate(model, val_records, device)
    best_metric, best_epoch = metric_value(initial, args.selection_metric), 0
    save_checkpoint(args.save_dir / "best.pt", model, optimizer, 0, best_metric, initial, args)
    history: List[Dict[str, Any]] = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        factor = legacy.linear_warmup(epoch, args.warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        train_stats = legacy.train_one_epoch(model, optimizer, train_records, device, args)
        validation = evaluate(model, val_records, device)
        selected = metric_value(validation, args.selection_metric)
        row = {"epoch": epoch, "train": train_stats, "validation": validation}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if selected > best_metric:
            best_metric, best_epoch, stale = selected, epoch, 0
            save_checkpoint(args.save_dir / "best.pt", model, optimizer, epoch, best_metric, validation, args)
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break
    (args.save_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checkpoint = torch.load(args.save_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    summary: Dict[str, Any] = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_metric": float(checkpoint["best_metric"]),
        "validation": checkpoint["validation"],
        "validation_ambiguity": ambiguity_diagnostics(model, val_records, device),
        "architecture": checkpoint["architecture"],
    }
    if args.test_list:
        test_paths = legacy.base.read_list_file(args.test_list)
        test_records = legacy.load_split(test_paths, label_to_int, args)
        legacy.validate_disjoint(train_records, val_records, test_records)
        summary["test"] = evaluate(model, test_records, device)
        summary["test_ambiguity"] = ambiguity_diagnostics(model, test_records, device)
        summary["test_worms"] = [record.worm_id for record in test_records]
    (args.save_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
