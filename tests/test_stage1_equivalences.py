"""
Equivalence unit tests:
1. FedProx with mu=0.0 must equal FedAvg exactly (same seed, same data).
2. SCAFFOLD with zero control variates on round 1 must equal FedAvg exactly.
3. FedNova with equal local steps must scale to FedAvg exactly.
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
    FedProx with mu=0 must equal FedAvg exactly (same seed).
    Objective: Loss + (0/2)*||w - w_global||^2 == Loss.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    init_model = CompactCNN(in_channels=1, num_classes=10)
    initial_sd = {k: v.clone() for k, v in init_model.state_dict().items()}

    # Run FedAvg
    torch.manual_seed(100)
    client_avg = FedAvgClient(
        client_id=0,
        model_fn=lambda: CompactCNN(in_channels=1, num_classes=10),
        device=device,
        learning_rate=0.05,
        weight_decay=0.0,
    )
    res_avg = client_avg.train(initial_sd, fixed_batch, epochs=2)

    # Run FedProx with mu=0.0
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
    SCAFFOLD with zero control variates on round 1 must equal FedAvg exactly.
    Gradient correction: g - c_i + c = g - 0 + 0 = g.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    init_model = CompactCNN(in_channels=1, num_classes=10)
    initial_sd = {k: v.clone() for k, v in init_model.state_dict().items()}

    # Run FedAvg
    torch.manual_seed(100)
    client_avg = FedAvgClient(
        client_id=0,
        model_fn=lambda: CompactCNN(in_channels=1, num_classes=10),
        device=device,
        learning_rate=0.05,
        weight_decay=0.0,
    )
    res_avg = client_avg.train(initial_sd, fixed_batch, epochs=2)

    # Run SCAFFOLD with c_i=None (initial zero) and server_control_variate=None (zero)
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
    FedNova with equal local steps must equal FedAvg when scaled by tau_i.
    FedNova delta = (w_local - w_global) / tau_i.
    Therefore, delta_nova * tau_i == delta_avg.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    init_model = CompactCNN(in_channels=1, num_classes=10)
    initial_sd = {k: v.clone() for k, v in init_model.state_dict().items()}

    # Run FedAvg
    torch.manual_seed(100)
    client_avg = FedAvgClient(
        client_id=0,
        model_fn=lambda: CompactCNN(in_channels=1, num_classes=10),
        device=device,
        learning_rate=0.05,
        weight_decay=0.0,
    )
    res_avg = client_avg.train(initial_sd, fixed_batch, epochs=2)

    # Run FedNova
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
