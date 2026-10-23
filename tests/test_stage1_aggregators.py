"""
Unit tests for Stage 1 Server Aggregation Rules.
Mandatory acceptance checks:
1. Krum rejecting an obvious outlier.
2. Trimmed Mean matching a numpy reference implementation.
3. Coordinate Median matching numpy reference.
4. Bulyan Byzantine robustness.
5. FedAvgMean weighted and unweighted equivalence.
"""

import numpy as np
import pytest
import torch

from flaas.fl.server_rules import (
    FedAvgMean,
    TrimmedMean,
    CoordinateMedian,
    Krum,
    Bulyan,
)


def numpy_trimmed_mean(arr: np.ndarray, beta: float) -> np.ndarray:
    """Numpy reference implementation of coordinate-wise trimmed mean."""
    n, d = arr.shape
    k = int(beta * n)
    if k == 0:
        return np.mean(arr, axis=0)
    if 2 * k >= n:
        k = max(0, (n - 1) // 2)
    sorted_arr = np.sort(arr, axis=0)
    trimmed = sorted_arr[k : n - k, :]
    return np.mean(trimmed, axis=0)


def test_krum_rejects_obvious_outlier():
    """
    Test that Krum rejects an obvious adversarial outlier vector
    and selects an honest update.
    """
    torch.manual_seed(42)
    dim = 20

    # 5 honest clients with small random updates centered near 1.0
    honest_updates = [torch.ones(dim) + 0.05 * torch.randn(dim) for _ in range(5)]

    # 1 obvious malicious outlier with extreme values
    malicious_update = torch.full((dim,), -100.0)

    all_updates = honest_updates + [malicious_update]

    # With n=6 and f=1: num_neighbors = 6 - 1 - 2 = 3
    krum = Krum(f=1, multi_krum=False)
    aggregated = krum.aggregate(all_updates)

    # Assert the selected update is one of the honest updates (mean close to 1.0)
    # and strictly not the outlier (-100.0)
    assert aggregated.mean().item() > 0.5
    assert not torch.allclose(aggregated, malicious_update)


def test_trimmed_mean_matches_numpy_reference():
    """
    Test that PyTorch TrimmedMean matches an independent NumPy reference implementation.
    """
    torch.manual_seed(123)
    np.random.seed(123)

    num_clients = 10
    dim = 50
    beta_values = [0.1, 0.2, 0.3]

    for beta in beta_values:
        data_np = np.random.randn(num_clients, dim).astype(np.float32)
        updates_torch = [torch.from_numpy(data_np[i]) for i in range(num_clients)]

        # Numpy reference
        expected_np = numpy_trimmed_mean(data_np, beta)

        # PyTorch TrimmedMean
        aggregator = TrimmedMean(beta=beta)
        actual_torch = aggregator.aggregate(updates_torch)

        np.testing.assert_allclose(
            actual_torch.numpy(),
            expected_np,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"TrimmedMean failed to match numpy reference for beta={beta}",
        )


def test_coordinate_median_matches_numpy():
    """
    Test that CoordinateMedian matches numpy.median for both odd and even client counts.
    """
    for num_clients in [5, 6, 11, 12]:
        data_np = np.random.randn(num_clients, 30).astype(np.float32)
        updates = [torch.from_numpy(data_np[i]) for i in range(num_clients)]

        expected_np = np.median(data_np, axis=0)

        aggregator = CoordinateMedian()
        actual = aggregator.aggregate(updates)

        np.testing.assert_allclose(
            actual.numpy(),
            expected_np,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"CoordinateMedian failed for num_clients={num_clients}",
        )


def test_fedavg_mean_weighted_and_unweighted():
    """
    Test FedAvgMean with and without sample weights.
    """
    updates = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([3.0, 4.0, 5.0]),
        torch.tensor([2.0, 6.0, 4.0]),
    ]

    aggregator = FedAvgMean()

    # Unweighted: (1+3+2)/3 = 2.0, (2+4+6)/3 = 4.0, (3+5+4)/3 = 4.0
    res_unweighted = aggregator.aggregate(updates)
    expected_unweighted = torch.tensor([2.0, 4.0, 4.0])
    assert torch.allclose(res_unweighted, expected_unweighted)

    # Weighted: weights [1, 2, 1], sum=4
    # dim 0: (1*1 + 2*3 + 1*2)/4 = 9/4 = 2.25
    # dim 1: (1*2 + 2*4 + 1*6)/4 = 16/4 = 4.0
    # dim 2: (1*3 + 2*5 + 1*4)/4 = 17/4 = 4.25
    res_weighted = aggregator.aggregate(updates, weights=[1.0, 2.0, 1.0])
    expected_weighted = torch.tensor([2.25, 4.0, 4.25])
    assert torch.allclose(res_weighted, expected_weighted)


def test_bulyan_mitigates_colluding_adversaries():
    """
    Test Bulyan with honest clients and Byzantine adversaries.
    """
    dim = 15
    # 8 honest clients around +2.0
    honest = [torch.full((dim,), 2.0) + 0.05 * torch.randn(dim) for _ in range(8)]
    # 2 malicious colluding clients with extreme negative values
    malicious = [torch.full((dim,), -50.0) for _ in range(2)]

    all_updates = honest + malicious
    bulyan = Bulyan(f=1)
    res = bulyan.aggregate(all_updates)

    # Aggregated result must remain close to honest average (+2.0)
    assert 1.5 < res.mean().item() < 2.5
