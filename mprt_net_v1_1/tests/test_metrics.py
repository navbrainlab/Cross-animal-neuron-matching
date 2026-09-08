import torch

from mprt_net.metrics import _directional_ranks


def test_real_and_dustbin_metrics_are_separate():
    probabilities = torch.tensor(
        [
            [0.60, 0.20, 0.20],
            [0.30, 0.10, 0.60],
        ]
    )
    targets = torch.tensor([0, 0])
    result = _directional_ranks(probabilities, targets).compute()
    assert result["queries"] == 2
    assert result["top1_real"] == 1.0
    assert result["top1_with_dustbin"] == 0.5
    assert result["dustbin_top1_rate"] == 0.5
    assert result["top1_real"] >= result["top1_with_dustbin"]


def test_unknown_and_dustbin_targets_are_excluded_from_benchmark_metrics():
    probabilities = torch.tensor(
        [
            [0.6, 0.2, 0.2],
            [0.2, 0.2, 0.6],
            [0.1, 0.7, 0.2],
        ]
    )
    # -1 is unknown; 2 is the dustbin target for two real candidates.
    targets = torch.tensor([0, -1, 2])
    result = _directional_ranks(probabilities, targets).compute()
    assert result["queries"] == 1
    assert result["top1_real"] == 1.0
