"""
Stage 1 Convergence Acceptance Test:
FedAvg on Real MNIST partitioned across clients.
Validates that:
1. Model trains on real image data (torchvision MNIST).
2. Global loss decreases from initial round.
3. Test accuracy improves significantly over initial baseline.
4. Executes properly on configured hardware device.
"""

import pytest
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset, DataLoader

from flaas.fl.device import get_device
from flaas.fl.models import MNIST_CNN
from flaas.fl.client_algos import FedAvgClient
from flaas.fl.server_rules import FedAvgMean
from flaas.fl.server_optimizers import ServerSGD
from flaas.fl.simulator import FLSimulator, seed_everything
from flaas.data.partitioners import partition_iid, get_client_dataloaders


def test_fedavg_convergence_on_real_mnist():
    """
    Acceptance test verifying FedAvg training dynamics on real MNIST.
    Uses 1,500 training samples across 3 clients for 3 fast rounds.
    """
    seed_everything(42)
    device = get_device()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = torchvision.datasets.MNIST(
        root="./data_cache",
        train=True,
        download=True,
        transform=transform,
    )
    full_test = torchvision.datasets.MNIST(
        root="./data_cache",
        train=False,
        download=True,
        transform=transform,
    )

    # Use first 1500 train samples (500 per client) and 500 test samples for fast test
    train_subset = Subset(full_train, list(range(1500)))
    test_subset = Subset(full_test, list(range(500)))

    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    num_clients = 3
    client_indices = partition_iid(train_subset, num_clients=num_clients, seed=42)
    client_dataloaders = get_client_dataloaders(train_subset, client_indices, batch_size=32)

    client_algos = {}
    for cid in range(num_clients):
        client_algos[cid] = FedAvgClient(
            client_id=cid,
            model_fn=lambda: MNIST_CNN(in_channels=1, num_classes=10),
            device=device,
            learning_rate=0.05,
            weight_decay=1e-4,
        )

    simulator = FLSimulator(
        model_fn=lambda: MNIST_CNN(in_channels=1, num_classes=10),
        client_algos=client_algos,
        client_dataloaders=client_dataloaders,
        test_dataloader=test_loader,
        server_rule=FedAvgMean(),
        server_optimizer=ServerSGD(lr=1.0),
        device=device,
        seed=42,
    )

    results = simulator.run(
        num_rounds=3,
        epochs_per_round=1,
        participation_rate=1.0,
        verbose=False,
        evaluate_initial=True,
    )

    history = results["history"]
    r0 = history[0]
    r_final = history[-1]

    # Acceptance checks:
    # 1. Round 0 is untrained random guess (< 25% accuracy)
    assert r0["accuracy"] < 0.25
    # 2. Final accuracy improves to > 70% on real data in 3 rounds
    assert r_final["accuracy"] > 0.70
    # 3. Final loss is significantly lower than initial loss
    assert r_final["loss"] < r0["loss"]
