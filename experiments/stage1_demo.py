"""
Standalone verification script for Stage 1 Acceptance Checks.
Runs FedAvg on synthetic MNIST-like benchmark, prints table, and outputs JSON metrics.
"""

import json
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

from flaas.fl.device import get_device
from flaas.fl.models import CompactCNN, generate_synthetic_mnist
from flaas.fl.client_algos import FedAvgClient
from flaas.fl.server_rules import FedAvgMean
from flaas.fl.server_optimizers import ServerSGD
from flaas.fl.simulator import FLSimulator, seed_everything


def main():
    seed_everything(42)
    device = get_device()
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 65)
    print("       FLaaS Stage 1 Acceptance Run: FedAvg Convergence")
    print(f"       Device in use: {device} [{dev_name}]")
    print("=" * 65)

    num_clients = 5
    num_rounds = 5
    epochs_per_round = 2
    batch_size = 32

    x_all, y_all = generate_synthetic_mnist(num_samples=1500, num_classes=10, seed=42)
    x_train, y_train = x_all[:1200], y_all[:1200]
    x_test, y_test = x_all[1200:], y_all[1200:]

    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    client_dataloaders = {}
    client_algos = {}
    samples_per_client = len(x_train) // num_clients

    for cid in range(num_clients):
        c_x = x_train[cid * samples_per_client : (cid + 1) * samples_per_client]
        c_y = y_train[cid * samples_per_client : (cid + 1) * samples_per_client]
        client_dataloaders[cid] = DataLoader(TensorDataset(c_x, c_y), batch_size=batch_size, shuffle=True)
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

    print("\nExecuting 5 Federated Training Rounds (5 clients, E=2 local epochs):")
    results = simulator.run(
        num_rounds=num_rounds,
        epochs_per_round=epochs_per_round,
        participation_rate=1.0,
        verbose=True,
    )

    history = results["history"]
    print("\n" + "-" * 65)
    print(f"{'Round':<8}{'Test Loss':<14}{'Test Accuracy':<16}{'Macro-F1':<14}{'Time (s)':<10}")
    print("-" * 65)
    for r in history:
        print(f"{r['round']:<8}{r['loss']:<14.4f}{r['accuracy']*100:<16.2f}{r['macro_f1']:<14.4f}{r['round_time']:<10.2f}")
    print("-" * 65)

    os.makedirs("results", exist_ok=True)
    summary_path = os.path.join("results", "stage1_fedavg_curve.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRun trace successfully saved to: {summary_path}")


if __name__ == "__main__":
    main()
