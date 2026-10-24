"""
Stage 3 Unit Tests: Domain Adapters, Frozen Profiles (kb-v1),
GPU Dataset Preloading, and Simulator Optimization Primitives.
Runs on CPU and CUDA.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flaas.data.domain_adapters import (
    HealthcareAdapter,
    FinanceAdapter,
    IoTEdgeAdapter,
    MobileAdapter,
    AutonomousVehiclesAdapter,
    GPUDataset,
    get_domain_adapter,
)
from flaas.fl.models import (
    MedCNN,
    FraudMLP,
    HARMLP,
    MobileCNN,
    TrafficSignCNN,
)
from flaas.profiles.kb_v1 import KB_V1_PROFILES, get_profile
from flaas.fl.simulator import FLSimulator
from flaas.fl.client_algos import FedAvgClient
from flaas.fl.server_rules import FedAvgMean
from flaas.fl.server_optimizers import ServerSGD
from flaas.compression.compressors import ThresholdedTernaryCompressor


def test_kb_v1_profiles_contain_all_five_domains():
    """Verify that kb-v1 frozen knowledge base contains all 5 required domains."""
    domains = ["healthcare", "finance", "iot_edge", "mobile_apps", "autonomous_vehicles"]
    for d in domains:
        profile = get_profile(d)
        assert profile.domain_id == d
        assert profile.num_classes > 0
        assert profile.target_accuracy > 0.0
        assert profile.heterogeneity_alpha > 0.0
        assert profile.privacy_budget_epsilon > 0.0


def test_healthcare_adapter_and_model():
    """Verify Healthcare adapter loads data and MedCNN executes forward pass."""
    adapter = HealthcareAdapter()
    train_ds, test_ds = adapter.load_data()
    meta = adapter.get_metadata()

    assert meta["domain_name"] == "healthcare"
    assert meta["dataset_source"] in ["real", "synthetic_fallback"]
    assert len(train_ds) > 0
    assert len(test_ds) > 0

    x, y = train_ds[0]
    assert x.shape == (3, 28, 28)

    model = MedCNN(in_channels=3, num_classes=8)
    out = model(x.unsqueeze(0))
    assert out.shape == (1, 8)


def test_finance_adapter_and_model():
    """Verify Finance adapter loads tabular data and FraudMLP executes forward pass."""
    adapter = FinanceAdapter()
    train_ds, test_ds = adapter.load_data()
    meta = adapter.get_metadata()

    assert meta["domain_name"] == "finance"
    assert meta["dataset_source"] in ["real", "synthetic_fallback"]

    x, y = train_ds[0]
    assert x.shape == (29,)

    model = FraudMLP(in_features=29, num_classes=2)
    out = model(x.unsqueeze(0))
    assert out.shape == (1, 2)


def test_iot_edge_adapter_and_model():
    """Verify IoT Edge adapter loads 561-feature sensor data and HARMLP executes."""
    adapter = IoTEdgeAdapter()
    train_ds, test_ds = adapter.load_data()
    meta = adapter.get_metadata()

    assert meta["domain_name"] == "iot_edge"
    assert meta["dataset_source"] in ["real", "synthetic_fallback"]

    x, y = train_ds[0]
    assert x.shape == (561,)

    model = HARMLP(in_features=561, num_classes=6)
    out = model(x.unsqueeze(0))
    assert out.shape == (1, 6)


def test_mobile_adapter_and_model():
    """Verify Mobile adapter loads FashionMNIST/EMNIST and MobileCNN executes."""
    adapter = MobileAdapter()
    train_ds, test_ds = adapter.load_data()
    meta = adapter.get_metadata()

    assert meta["domain_name"] == "mobile_apps"
    assert meta["dataset_source"] in ["real", "synthetic_fallback"]

    x, y = train_ds[0]
    assert x.shape == (1, 28, 28)

    model = MobileCNN(in_channels=1, num_classes=10)
    out = model(x.unsqueeze(0))
    assert out.shape == (1, 10)


def test_autonomous_vehicles_adapter_and_model():
    """Verify Autonomous Vehicles adapter loads GTSRB and TrafficSignCNN executes."""
    adapter = AutonomousVehiclesAdapter()
    train_ds, test_ds = adapter.load_data()
    meta = adapter.get_metadata()

    assert meta["domain_name"] == "autonomous_vehicles"
    assert meta["dataset_source"] in ["real", "synthetic_fallback"]

    x, y = train_ds[0]
    assert x.shape == (3, 32, 32)

    model = TrafficSignCNN(in_channels=3, num_classes=43)
    out = model(x.unsqueeze(0))
    assert out.shape == (1, 43)


def test_gpu_dataset_preloading():
    """Verify GPUDataset tensor preloading and batch indexing."""
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    ds = GPUDataset(x, y)

    assert len(ds) == 100
    item_x, item_y = ds[5]
    assert item_x.shape == (10,)

    batch_x, batch_y = ds.get_batch(torch.tensor([0, 1, 2]))
    assert batch_x.shape == (3, 10)
    assert batch_y.shape == (3,)


def test_simulator_early_stopping_at_target_accuracy():
    """Verify simulator terminates early when target_accuracy threshold is reached."""
    # Synthetic separable dataset where model achieves 100% quickly
    x = torch.cat([torch.randn(50, 4) + 5.0, torch.randn(50, 4) - 5.0], dim=0)
    y = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)], dim=0)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    client_algos = {
        0: FedAvgClient(0, lambda: nn.Linear(4, 2), torch.device("cpu"), learning_rate=0.2),
        1: FedAvgClient(1, lambda: nn.Linear(4, 2), torch.device("cpu"), learning_rate=0.2),
    }
    client_loaders = {0: loader, 1: loader}

    sim = FLSimulator(
        model_fn=lambda: nn.Linear(4, 2),
        client_algos=client_algos,
        client_dataloaders=client_loaders,
        test_dataloader=loader,
        server_rule=FedAvgMean(),
        server_optimizer=ServerSGD(lr=1.0),
        device=torch.device("cpu"),
        seed=42,
    )

    # Ask for 20 rounds, but target_accuracy = 80.0%
    res = sim.run(
        num_rounds=20,
        epochs_per_round=1,
        target_accuracy=80.0,
        verbose=False,
    )

    assert res["early_stopped"] is True
    assert res["rounds_completed"] < 20


def test_ternary_2bit_packing_compressor():
    """Verify ThresholdedTernaryCompressor represents ternary 2-bit packing (16x compression)."""
    compressor = ThresholdedTernaryCompressor(threshold_factor=1.0)
    delta = torch.tensor([-0.8, -0.05, 0.0, 0.05, 0.9, 0.0, -0.5, 0.3])
    comp, wire_bytes = compressor.compress(delta)

    # 8 coordinates * 2 bits = 16 bits = 2 bytes + 4 bytes scale = 6 bytes
    assert wire_bytes == 6
