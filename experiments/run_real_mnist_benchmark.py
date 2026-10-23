"""
Real MNIST Acceptance Benchmark:
Compares FedAvg across 30 rounds in two settings:
1. IID (10 clients, E=2 local epochs)
2. Non-IID Dirichlet alpha=0.1 (10 clients, E=2 local epochs)

Evaluation is performed after each round on the official, completely disjoint
MNIST test set (10,000 real images).
Records dataset_source="real" in all outputs.
"""

import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from flaas.fl.device import get_device
from flaas.fl.models import MNIST_CNN
from flaas.fl.client_algos import FedAvgClient
from flaas.fl.server_rules import FedAvgMean
from flaas.fl.server_optimizers import ServerSGD
from flaas.fl.simulator import FLSimulator, seed_everything
from flaas.data.partitioners import (
    partition_iid,
    partition_dirichlet,
    get_client_dataloaders,
)


def run_experiment(
    partition_type: str,
    alpha: float,
    num_clients: int = 10,
    num_rounds: int = 30,
    local_epochs: int = 2,
    batch_size: int = 64,
    lr: float = 0.05,
    seed: int = 42,
    data_dir: str = "./data_cache",
):
    seed_everything(seed)
    device = get_device()
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print(f"\n{'='*70}")
    print(f" Running Real MNIST Benchmark: {partition_type.upper()} "
          f"(alpha={alpha if partition_type == 'dirichlet' else 'N/A'})")
    print(f" Device: {device} [{dev_name}] | Clients: {num_clients} | Rounds: {num_rounds} | E: {local_epochs}")
    print(f"{'='*70}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Official MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Partition train dataset
    if partition_type == "iid":
        client_indices = partition_iid(train_dataset, num_clients=num_clients, seed=seed)
    elif partition_type == "dirichlet":
        client_indices = partition_dirichlet(
            train_dataset,
            num_clients=num_clients,
            alpha=alpha,
            num_classes=10,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown partition_type: {partition_type}")

    client_dataloaders = get_client_dataloaders(
        train_dataset,
        client_indices,
        batch_size=batch_size,
        shuffle=True,
    )

    client_algos = {}
    for cid in range(num_clients):
        client_algos[cid] = FedAvgClient(
            client_id=cid,
            model_fn=lambda: MNIST_CNN(in_channels=1, num_classes=10),
            device=device,
            learning_rate=lr,
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
        seed=seed,
    )

    results = simulator.run(
        num_rounds=num_rounds,
        epochs_per_round=local_epochs,
        participation_rate=1.0,
        verbose=True,
        evaluate_initial=True,
    )

    # Enrich metadata
    results["dataset_source"] = "real"
    results["dataset_name"] = "MNIST"
    results["total_train_samples"] = len(train_dataset)
    results["total_test_samples"] = len(test_dataset)
    results["partition_type"] = partition_type
    results["dirichlet_alpha"] = alpha if partition_type == "dirichlet" else None
    results["num_clients"] = num_clients
    results["local_epochs"] = local_epochs
    results["batch_size"] = batch_size
    results["learning_rate"] = lr
    results["seed"] = seed

    return results


def main():
    os.makedirs("results", exist_ok=True)

    # 1. Run IID benchmark
    iid_results = run_experiment(
        partition_type="iid",
        alpha=1.0,
        num_clients=10,
        num_rounds=30,
        local_epochs=2,
    )
    iid_path = os.path.join("results", "real_mnist_iid_30rounds.json")
    with open(iid_path, "w") as f:
        json.dump(iid_results, f, indent=2)
    print(f"Saved IID results to {iid_path}")

    # 2. Run Dirichlet alpha=0.1 Non-IID benchmark
    noniid_results = run_experiment(
        partition_type="dirichlet",
        alpha=0.1,
        num_clients=10,
        num_rounds=30,
        local_epochs=2,
    )
    noniid_path = os.path.join("results", "real_mnist_noniid_alpha0.1_30rounds.json")
    with open(noniid_path, "w") as f:
        json.dump(noniid_results, f, indent=2)
    print(f"Saved Non-IID results to {noniid_path}")

    # 3. Generate Comparison Plot
    rounds_iid = [r["round"] for r in iid_results["history"]]
    acc_iid = [r["accuracy"] * 100.0 for r in iid_results["history"]]
    loss_iid = [r["loss"] for r in iid_results["history"]]

    rounds_noniid = [r["round"] for r in noniid_results["history"]]
    acc_noniid = [r["accuracy"] * 100.0 for r in noniid_results["history"]]
    loss_noniid = [r["loss"] for r in noniid_results["history"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    ax1.plot(rounds_iid, acc_iid, marker="o", label="IID (10 clients)", color="#2563eb", linewidth=2)
    ax1.plot(rounds_noniid, acc_noniid, marker="s", label=r"Non-IID Dir($\alpha=0.1$)", color="#dc2626", linewidth=2)
    ax1.set_title("Real MNIST FedAvg Test Accuracy over 30 Rounds", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Communication Round", fontsize=11)
    ax1.set_ylabel("Global Test Accuracy (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss comparison
    ax2.plot(rounds_iid, loss_iid, marker="o", label="IID (10 clients)", color="#2563eb", linewidth=2)
    ax2.plot(rounds_noniid, loss_noniid, marker="s", label=r"Non-IID Dir($\alpha=0.1$)", color="#dc2626", linewidth=2)
    ax2.set_title("Real MNIST FedAvg Test Loss over 30 Rounds", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Communication Round", fontsize=11)
    ax2.set_ylabel("Global Test Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join("results", "mnist_iid_vs_noniid_convergence.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
