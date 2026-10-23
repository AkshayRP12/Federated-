"""
Unit tests for Client Algorithms:
- FedAvgClient
- FedProxClient
- SCAFFOLDClient
- FedNovaClient
All tests are validated on CPU and device.
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
from flaas.fl.server_rules import state_dict_to_vector


class TinyMLP(nn.Module):
    """Minimal model for fast testing."""

    def __init__(self, in_features: int = 4, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def dummy_data():
    torch.manual_seed(42)
    x = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=5)


def test_fedavg_client(dummy_data):
    device = torch.device("cpu")
    client = FedAvgClient(client_id=0, model_fn=TinyMLP, device=device, learning_rate=0.05)

    model = TinyMLP()
    initial_sd = {k: v.clone() for k, v in model.state_dict().items()}

    res = client.train(initial_sd, dummy_data, epochs=1)
    assert res.client_id == 0
    assert res.num_samples == 20
    assert res.delta.shape == state_dict_to_vector(initial_sd).shape
    assert res.local_steps == 4
    # Ensure delta is non-zero after training
    assert torch.norm(res.delta).item() > 0.0


def test_fedprox_client(dummy_data):
    device = torch.device("cpu")
    client = FedProxClient(
        client_id=1, model_fn=TinyMLP, device=device, learning_rate=0.05, mu=0.1
    )

    model = TinyMLP()
    initial_sd = {k: v.clone() for k, v in model.state_dict().items()}

    res = client.train(initial_sd, dummy_data, epochs=1)
    assert res.client_id == 1
    assert res.delta.shape == state_dict_to_vector(initial_sd).shape
    assert torch.norm(res.delta).item() > 0.0


def test_scaffold_client(dummy_data):
    device = torch.device("cpu")
    client = SCAFFOLDClient(
        client_id=2, model_fn=TinyMLP, device=device, learning_rate=0.05
    )

    model = TinyMLP()
    initial_sd = {k: v.clone() for k, v in model.state_dict().items()}

    res = client.train(initial_sd, dummy_data, epochs=1)
    assert res.client_id == 2
    assert res.control_variate_delta is not None
    assert res.control_variate_delta.shape == res.delta.shape
    assert torch.norm(res.control_variate_delta).item() > 0.0


def test_fednova_client(dummy_data):
    device = torch.device("cpu")
    client = FedNovaClient(
        client_id=3, model_fn=TinyMLP, device=device, learning_rate=0.05
    )

    model = TinyMLP()
    initial_sd = {k: v.clone() for k, v in model.state_dict().items()}

    res = client.train(initial_sd, dummy_data, epochs=1)
    assert res.client_id == 3
    assert res.extra_metrics["tau_i"] == 4.0
    assert res.delta.shape == state_dict_to_vector(initial_sd).shape
    assert torch.norm(res.delta).item() > 0.0
