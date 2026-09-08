from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .config import ModelConfig
from .data import PairIndex, WormSample, build_pair_targets, load_worm
from .losses import symmetric_focal_matching_loss
from .model import MPRTNet
from .sinkhorn import augmented_sinkhorn, brute_force_relation_cost, contracted_relation_cost


def _reorder(sample: WormSample, order: torch.Tensor) -> WormSample:
    indices = order.tolist()
    return WormSample(
        uid=sample.uid,
        xyz=sample.xyz[order],
        activity=sample.activity[order],
        cell_ids=tuple(sample.cell_ids[i] for i in indices),
        supervised_mask=sample.supervised_mask[order],
        source_path=sample.source_path,
    )


def _find_pair(root: str | Path, split: str) -> tuple[Path, Path]:
    index = PairIndex(root, split, min_shared=2)
    return index.pairs[0]


def run_checks(path_a: Path, path_b: Path, device: torch.device) -> None:
    torch.manual_seed(7)

    logits = torch.randn(5, 7, device=device)
    sinkhorn = augmented_sinkhorn(
        logits, torch.tensor(-1.0, device=device), torch.tensor(-1.0, device=device), 50
    )
    row_error = float((sinkhorn.plan.sum(1) - sinkhorn.mu).abs().max())
    col_error = float((sinkhorn.plan.sum(0) - sinkhorn.nu).abs().max())
    assert row_error < 2e-5 and col_error < 2e-5
    print(f"[PASS] augmented Sinkhorn marginals row={row_error:.2e} col={col_error:.2e}")

    relation_a = torch.randn(4, 4, 3, device=device)
    relation_b = torch.randn(5, 5, 3, device=device)
    plan = torch.rand(4, 5, device=device)
    efficient = contracted_relation_cost(relation_a, relation_b, plan)
    reference = brute_force_relation_cost(relation_a, relation_b, plan)
    relation_error = float((efficient - reference).abs().max())
    assert relation_error < 2e-5
    print(f"[PASS] O(dN^3) relation contraction error={relation_error:.2e}")

    sample_a = load_worm(path_a, activity_length=128)
    sample_b = load_worm(path_b, activity_length=128)
    sample_a, sample_b, targets = build_pair_targets(sample_a, sample_b)
    assert targets.num_direct_matches >= 1
    sample_a = sample_a.to(device)
    sample_b = sample_b.to(device)
    targets = targets.to(device)
    print(
        f"[PASS] data contract A={sample_a.uid}:{sample_a.num_nodes} "
        f"B={sample_b.uid}:{sample_b.num_nodes} matches={targets.num_direct_matches}"
    )

    config = ModelConfig(
        hidden_dim=32,
        edge_dim=16,
        relation_dim=3,
        activity_channels=8,
        num_heads=4,
        population_layers=1,
        dropout=0.0,
        sinkhorn_iterations=30,
        transport_steps=1,
    )
    model = MPRTNet(config).to(device)
    model.eval()
    output = model(sample_a, sample_b)
    assert torch.isfinite(output.plan).all()
    assert output.plan.shape == (sample_a.num_nodes + 1, sample_b.num_nodes + 1)
    print(f"[PASS] forward plan shape={tuple(output.plan.shape)} finite=True")

    order = torch.randperm(sample_b.num_nodes, device=device)
    permuted_b = _reorder(sample_b, order)
    with torch.no_grad():
        permuted_output = model(sample_a, permuted_b)
    equivariance_error = float(
        (
            permuted_output.plan[:-1, :-1]
            - output.plan[:-1, :-1][:, order]
        )
        .abs()
        .max()
    )
    assert equivariance_error < 2e-5
    print(f"[PASS] node-permutation equivariance error={equivariance_error:.2e}")

    model.train()
    model.zero_grad(set_to_none=True)
    output = model(sample_a, sample_b)
    loss = symmetric_focal_matching_loss(output, targets, gamma=2.0).total
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "relation_projector" in name and parameter.grad is not None
    ]
    assert math.isfinite(float(loss))
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0
    print(f"[PASS] backward loss={float(loss):.6f} relation gradients finite/nonzero")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuRID numerical and data-contract checks")
    parser.add_argument("--dataset-root")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-a")
    parser.add_argument("--sample-b")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.sample_a and args.sample_b:
        path_a, path_b = Path(args.sample_a), Path(args.sample_b)
    elif args.dataset_root:
        path_a, path_b = _find_pair(args.dataset_root, args.split)
    else:
        parser.error("Provide either --dataset-root or both --sample-a and --sample-b")
    run_checks(path_a, path_b, torch.device(args.device))
    print("All NeuRID checks passed.")


if __name__ == "__main__":
    main()
