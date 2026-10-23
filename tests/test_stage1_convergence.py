"""
Stage 1 Convergence Acceptance Test:
FedAvg on an MNIST-like synthetic multi-class dataset partitioned across clients.
Validates that:
1. Global loss decreases across rounds from initialization.
2. Global test accuracy improves from initial random baseline (~10%) to >85%.
3. Runs cleanly on the configured device (CUDA/MPS/CPU).
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from flaas.fl.device import get_device
from flaas.fl.client_algos import FedAvgClient
from flaas.fl.server_rules import FedAvgMean
from flaas.fl.server_optimizers import ServerSGD
from flaas.fl.simulator import FLSimulator, seed_everything
from flaas.fl.models import CompactCNN, generate_synthetic_mnist


def test_fedavg_convergence_acceptance():
    """
    Acceptance Test for Stage 1: FedAvg convergence.
    Partition across 5 clients, run 5 rounds, verify accuracy curve reaches > 85%.
    """
    seed_everything(42)
    device = get_device()

    num_clients = 5
    num_rounds = 5
    epochs_per_round = 2
    batch_size = 32

    x_all, y_all = generate_synthetic_mnist(num_samples=1500, num_classes=10, seed=42)

    # 1200 train samples (240 per client), 300 test samples
    x_train, y_train = x_all[:1200], y_all[:1200]
    x_test, y_test = x_all[1200:], y_all[1200:]

    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    client_dataloaders = {}
    client_algos = {}
    samples_per_client = len(x_train) // num_clients

    for cid in range(num_clients):
        c_x = x_train[cid * samples_per_client : (cid + 1) * samples_per_client]
        c_y = y_train[cid * samples_per_client : (cid + 1) * samples_per_client]
        c_loader = DataLoader(TensorDataset(c_x, c_y), batch_size=batch_size, shuffle=True)
        client_dataloaders[cid] = c_loader

        client_algos[cid] = FedAvgClient(
            client_id=cid,
            model_fn=lambda: CompactCNN(in_channels=1, num_classes=10),
            device=device,
            learning_rate=0.05,
            weight_decay=1e-4,
        )

    simulator = FLSimulator(
        model_fn=lambda: CompactCNN(in_channels=1, num_classes=10),
        client_algos=client_algos,
        client_dataloaders=client_dataloaders,
        test_dataloader=test_loader,
        server_rule=FedAvgMean(),
        server_optimizer=ServerSGD(lr=1.0),
        device=device,
        seed=42,
    )

    results = simulator.run(
        num_rounds=num_rounds,
        epochs_per_round=epochs_per_round,
        participation_rate=1.0,
        verbose=True,
        evaluate_initial=True,
    )

    history = results["history"]
    r0 = history[0]
    r_final = history[-1]

    # Acceptance criteria:
    # 1. Round 0 is uninitialized random baseline (< 35% accuracy)
    assert r0["round"] == 0
    assert r0["accuracy"] < 0.35, f"Expected initial random accuracy < 0.35, got {r0['accuracy']:.4f}"

    # 2. Final accuracy reaches benchmark >= 85%
    assert r_final["accuracy"] >= 0.85, f"Expected final accuracy >= 0.85, got {r_final['accuracy']:.4f}"

    # 3. Final loss is significantly lower than initial loss
    assert r_final["loss"] < r0["loss"], f"Final loss ({r_final['loss']:.4f}) should be lower than initial ({r0['loss']:.4f})"
