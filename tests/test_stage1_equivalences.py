"""
Equivalence unit tests:
1. FedProx with mu=0.0 must equal FedAvg numerically (atol 1e-6, same seed, same data).
2. SCAFFOLD with zero control variates on round 1 must equal FedAvg numerically.
3. FedNova with equal local steps must scale to FedAvg numerically.
4. Server-level: Full one-round FedNova with equal steps must equal FedAvg global weights numerically.
5. Server rule: TrimmedMean(beta=0.0) must equal unweighted FedAvgMean numerically.
6. Server rule: MultiKrum selecting all clients (m=n, f=0) must equal unweighted FedAvgMean numerically.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from flaas.fl.client_algos import (
    FedAvgClient,
    FedProxClient,
    SCAFFOLDClient,
    FedNovaClient,
)
from flaas.fl.server_rules import FedAvgMean, TrimmedMean, Krum, state_dict_to_vector, vector_to_state_dict
from flaas.fl.server_optimizers import ServerSGD
from flaas.fl.models import CompactCNN


@pytest.fixture
def fixed_batch():
    torch.manual_seed(999)
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=8, shuffle=False)


def test_fedprox_mu_zero_equals_fedavg(fixed_batch):
    """
    FedProx with mu=0 must equal FedAvg numerically (atol 1e-6).
    Objective: Loss + (0/2)*||w - w_global||^2 == Loss.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    init_model = CompactCNN(in_channels=1, num_classes=10)
    initial_sd = {k: v.clone() for k, v in init_model.state_dict().items()}

    torch.manual_seed(100)
    client_avg = FedAvgClient(
        client_id=0,
        model_fn=lambda: CompactCNN(in_channels=1, num_classes=10),
        device=device,
        learning_rate=0.05,
        weight_decay=0.0,
    )
    res_avg = client_avg.train(initial_sd, fixed_batch, epochs=2)

    torch.manual_seed(100)
    client_prox = FedProxClient(
        client_id=0,
        model_fn=lambda: CompactCNN(in_channels=1, num_classes=10),
        device=device,
        learning_rate=0.05,
        weight_decay=0.0,
        mu=0.0,
    )
    res_prox = client_prox.train(initial_sd, fixed_batch, epochs=2)

    assert torch.allclose(res_avg.delta, res_prox.delta, atol=1e-6), (
        f"Max difference between FedAvg and FedProx(mu=0): "
        f"{(res_avg.delta - res_prox.delta).abs().max().item()}"
    )
    assert abs(res_avg.loss - res_prox.loss) < 1e-6


def test_scaffold_zero_control_variates_equals_fedavg(fixed_batch):
    """
    SCAFFOLD with zero control variates on round 1 must equal FedAvg numerically (atol 1e-6).
    Gradient correction: g - c_i + c = g - 0 + 0 = g.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    init_model = CompactCNN(in_channels=1, num_classes=10)
    initial_sd = {k: v.clone() for k, v in init_model.state_dict().items()}

    torch.manual_seed(100)
    client_avg = FedAvgClient(
        client_id=0,
        model_fn=lambda: CompactCNN(in_channels=1, num_classes=10),
        device=device,
        learning_rate=0.05,
        weight_decay=0.0,
    )
    res_avg = client_avg.train(initial_sd, fixed_batch, epochs=2)

    torch.manual_seed(100)
    client_scaffold = SCAFFOLDClient(
        client_id=0,
        model_fn=lambda: CompactCNN(in_channels=1, num_classes=10),
        device=device,
        learning_rate=0.05,
        weight_decay=0.0,
    )
    res_scaffold = client_scaffold.train(
        initial_sd,
        fixed_batch,
        epochs=2,
        server_control_variate=None,
    )

    assert torch.allclose(res_avg.delta, res_scaffold.delta, atol=1e-6), (
        f"Max difference between FedAvg and SCAFFOLD(c=0): "
        f"{(res_avg.delta - res_scaffold.delta).abs().max().item()}"
    )
    assert abs(res_avg.loss - res_scaffold.loss) < 1e-6


def test_fednova_equal_steps_equals_fedavg(fixed_batch):
    """
    FedNova with equal local steps must equal FedAvg when scaled by tau_i (atol 1e-6).
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    init_model = CompactCNN(in_channels=1, num_classes=10)
    initial_sd = {k: v.clone() for k, v in init_model.state_dict().items()}

    torch.manual_seed(100)
    client_avg = FedAvgClient(
        client_id=0,
        model_fn=lambda: CompactCNN(in_channels=1, num_classes=10),
        device=device,
        learning_rate=0.05,
        weight_decay=0.0,
    )
    res_avg = client_avg.train(initial_sd, fixed_batch, epochs=2)

    torch.manual_seed(100)
    client_nova = FedNovaClient(
        client_id=0,
        model_fn=lambda: CompactCNN(in_channels=1, num_classes=10),
        device=device,
        learning_rate=0.05,
        weight_decay=0.0,
    )
    res_nova = client_nova.train(initial_sd, fixed_batch, epochs=2)

    tau = res_nova.extra_metrics["tau_i"]
    reconstructed_delta = res_nova.delta * tau

    assert torch.allclose(res_avg.delta, reconstructed_delta, atol=1e-6), (
        f"Max difference between FedAvg and FedNova*tau: "
        f"{(res_avg.delta - reconstructed_delta).abs().max().item()}"
    )
    assert abs(res_avg.loss - res_nova.loss) < 1e-6


def test_full_round_fednova_equals_fedavg_at_server():
    """
    Full one-round server-level test:
    When clients have equal local steps tau, aggregating FedNova with effective
    step size tau_eff produces global model weights numerically equal to FedAvg (atol 1e-6).
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    init_model = CompactCNN(in_channels=1, num_classes=10)
    initial_sd = {k: v.clone() for k, v in init_model.state_dict().items()}
    w_global_vec = state_dict_to_vector(initial_sd)

    # 3 clients with identical batch sizes
    torch.manual_seed(10)
    data = [
        DataLoader(TensorDataset(torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))), batch_size=8)
        for _ in range(3)
    ]

    # 1. FedAvg 1-round execution
    avg_deltas = []
    for cid in range(3):
        torch.manual_seed(cid + 50)
        c = FedAvgClient(cid, lambda: CompactCNN(1, 10), device, learning_rate=0.05)
        res = c.train(initial_sd, data[cid], epochs=1)
        avg_deltas.append(res.delta)

    rule_mean = FedAvgMean()
    agg_avg = rule_mean.aggregate(avg_deltas)
    w_avg_next = w_global_vec + agg_avg

    # 2. FedNova 1-round execution
    nova_deltas = []
    tau_list = []
    for cid in range(3):
        torch.manual_seed(cid + 50)
        c = FedNovaClient(cid, lambda: CompactCNN(1, 10), device, learning_rate=0.05)
        res = c.train(initial_sd, data[cid], epochs=1)
        nova_deltas.append(res.delta)
        tau_list.append(res.extra_metrics["tau_i"])

    assert all(t == tau_list[0] for t in tau_list), "Local steps should be identical"
    tau_eff = tau_list[0]

    # In FedNova, server aggregates normalized updates and scales by tau_eff
    agg_nova = rule_mean.aggregate(nova_deltas)
    w_nova_next = w_global_vec + tau_eff * agg_nova

    assert torch.allclose(w_avg_next, w_nova_next, atol=1e-6), (
        f"Server-level FedAvg and FedNova weights differ by max "
        f"{(w_avg_next - w_nova_next).abs().max().item()}"
    )


def test_trimmed_mean_beta_zero_equals_mean():
    """
    Server rule equivalence:
    TrimmedMean(beta=0.0) must equal unweighted FedAvgMean numerically (atol 1e-6).
    """
    torch.manual_seed(77)
    updates = [torch.randn(20) for _ in range(6)]

    mean_agg = FedAvgMean()
    trimmed_agg = TrimmedMean(beta=0.0)

    res_mean = mean_agg.aggregate(updates)
    res_trimmed = trimmed_agg.aggregate(updates)

    assert torch.allclose(res_mean, res_trimmed, atol=1e-6), (
        f"TrimmedMean(beta=0) and Mean differ by max {(res_mean - res_trimmed).abs().max().item()}"
    )


def test_multikrum_all_selected_equals_mean():
    """
    Server rule equivalence:
    MultiKrum with f=0 and m=n must select all clients and equal unweighted FedAvgMean numerically (atol 1e-6).
    """
    torch.manual_seed(88)
    n = 6
    updates = [torch.randn(20) for _ in range(n)]

    mean_agg = FedAvgMean()
    krum_all_agg = Krum(f=0, multi_krum=True, m=n)

    res_mean = mean_agg.aggregate(updates)
    res_krum = krum_all_agg.aggregate(updates)

    assert torch.allclose(res_mean, res_krum, atol=1e-6), (
        f"MultiKrum(m=n, f=0) and Mean differ by max {(res_mean - res_krum).abs().max().item()}"
    )
