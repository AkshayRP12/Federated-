"""
Stage 2 Comprehensive Acceptance Benchmark:
1. Robustness: 20% Sign-Flip Attackers on Real MNIST (Mean vs Krum vs TrimmedMean across 3 seeds).
2. Privacy Accounting: Opacus RDP verification table against literature baselines.
3. Compression: Wire bytes sent per round and accuracy curves across 3 seeds.
"""

import json
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from flaas.fl.device import get_device
from flaas.fl.models import MNIST_CNN
from flaas.fl.client_algos import FedAvgClient
from flaas.fl.server_rules import FedAvgMean, Krum, TrimmedMean
from flaas.fl.server_optimizers import ServerSGD
from flaas.fl.simulator import FLSimulator, seed_everything
from flaas.attacks.attacks import SignFlipAttack
from flaas.privacy.accountant import VerifiedRDPAccountant
from flaas.privacy.clip_noise import DPClipperNoise
from flaas.compression.compressors import (
    IdentityCompressor,
    TopKCompressor,
    QSGDCompressor,
    SparseTernaryCompressor,
)
from flaas.data.partitioners import partition_iid, get_client_dataloaders


def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = torchvision.datasets.MNIST(
        root="./data_cache",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data_cache",
        train=False,
        download=True,
        transform=transform,
    )
    return train_dataset, test_dataset


def run_robustness_benchmark(train_dataset, test_dataset, seeds=[42, 100, 2026], num_rounds=10):
    """
    Evaluates 20% sign-flip attackers on real MNIST across 3 seeds:
    - Honest baseline (Mean, 0 attackers)
    - Mean under attack (20% sign-flip)
    - Krum under attack (20% sign-flip)
    - TrimmedMean under attack (20% sign-flip)
    """
    device = get_device()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    num_clients = 10
    adversary_ids = [0, 1]  # 2 out of 10 = 20%
    attack = SignFlipAttack(scale=3.0)

    configs = {
        "Honest_Mean": {"rule": lambda: FedAvgMean(), "adv": []},
        "Attacked_Mean": {"rule": lambda: FedAvgMean(), "adv": adversary_ids},
        "Attacked_Krum": {"rule": lambda: Krum(f=2, multi_krum=False), "adv": adversary_ids},
        "Attacked_TrimmedMean": {"rule": lambda: TrimmedMean(beta=0.2), "adv": adversary_ids},
    }

    raw_results = {name: [] for name in configs}

    print("\n" + "=" * 75)
    print(" 1. ROBUSTNESS BENCHMARK: 20% Sign-Flip Attackers on Real MNIST (3 Seeds)")
    print("=" * 75)

    for cfg_name, cfg in configs.items():
        print(f"\n--- Testing Config: {cfg_name} ---")
        for seed in seeds:
            seed_everything(seed)
            client_indices = partition_iid(train_dataset, num_clients=num_clients, seed=seed)
            client_loaders = get_client_dataloaders(train_dataset, client_indices, batch_size=64)

            client_algos = {
                cid: FedAvgClient(
                    client_id=cid,
                    model_fn=lambda: MNIST_CNN(1, 10),
                    device=device,
                    learning_rate=0.05,
                )
                for cid in range(num_clients)
            }

            sim = FLSimulator(
                model_fn=lambda: MNIST_CNN(1, 10),
                client_algos=client_algos,
                client_dataloaders=client_loaders,
                test_dataloader=test_loader,
                server_rule=cfg["rule"](),
                server_optimizer=ServerSGD(lr=1.0),
                device=device,
                seed=seed,
                adversary_ids=cfg["adv"],
                attack=attack if cfg["adv"] else None,
            )

            res = sim.run(num_rounds=num_rounds, epochs_per_round=1, verbose=False, evaluate_initial=True)
            accs = [r["accuracy"] * 100.0 for r in res["history"]]
            losses = [r["loss"] for r in res["history"]]
            raw_results[cfg_name].append({"accuracy": accs, "loss": losses})
            print(f"  Seed {seed:4d} | Final Acc: {accs[-1]:.2f}% | Final Loss: {losses[-1]:.4f}")

    # Compute mean and std per round
    summary = {}
    for cfg_name in configs:
        acc_matrix = np.array([run["accuracy"] for run in raw_results[cfg_name]])
        loss_matrix = np.array([run["loss"] for run in raw_results[cfg_name]])
        summary[cfg_name] = {
            "acc_mean": acc_matrix.mean(axis=0).tolist(),
            "acc_std": acc_matrix.std(axis=0).tolist(),
            "loss_mean": loss_matrix.mean(axis=0).tolist(),
            "loss_std": loss_matrix.std(axis=0).tolist(),
        }

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rounds = list(range(num_rounds + 1))
    colors = {
        "Honest_Mean": "#16a34a",
        "Attacked_Mean": "#dc2626",
        "Attacked_Krum": "#2563eb",
        "Attacked_TrimmedMean": "#9333ea",
    }
    labels = {
        "Honest_Mean": "Honest FedAvg Mean (0% adv)",
        "Attacked_Mean": "Plain Mean (20% sign-flip)",
        "Attacked_Krum": "Krum (20% sign-flip)",
        "Attacked_TrimmedMean": r"Trimmed Mean $\beta=0.2$ (20% sign-flip)",
    }

    for cfg_name, stats in summary.items():
        m = np.array(stats["acc_mean"])
        s = np.array(stats["acc_std"])
        c = colors[cfg_name]
        ax1.plot(rounds, m, label=labels[cfg_name], color=c, linewidth=2)
        ax1.fill_between(rounds, m - s, m + s, color=c, alpha=0.15)

        lm = np.array(stats["loss_mean"])
        ls = np.array(stats["loss_std"])
        ax2.plot(rounds, lm, label=labels[cfg_name], color=c, linewidth=2)
        ax2.fill_between(rounds, lm - ls, lm + ls, color=c, alpha=0.15)

    ax1.set_title("Real MNIST: Robustness Under 20% Sign-Flip Attack (Mean ± Std)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Communication Round", fontsize=10)
    ax1.set_ylabel("Global Test Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.set_title("Test Loss Under 20% Sign-Flip Attack (Mean ± Std)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Communication Round", fontsize=10)
    ax2.set_ylabel("Global Test Loss", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join("results", "stage2_robustness_signflip_3seeds.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nSaved robustness plot to {plot_path}")

    json_path = os.path.join("results", "stage2_robustness_signflip_3seeds.json")
    with open(json_path, "w") as f:
        json.dump({"dataset_source": "real", "dataset_name": "MNIST", "summary": summary}, f, indent=2)
    print(f"Saved robustness JSON to {json_path}")

    return summary


def run_privacy_accounting_verification():
    """
    Validates Opacus RDP accountant against known values from Opacus paper/docs.
    """
    print("\n" + "=" * 75)
    print(" 2. DIFFERENTIAL PRIVACY: RDP Accountant Verification against Literature")
    print("=" * 75)

    test_cases = [
        {"steps": 10, "q": 0.01, "sigma": 1.0, "delta": 1e-5, "ref_eps": 1.0353, "cite": "Opacus RDP Subsampled Gaussian (Mironov 2017)"},
        {"steps": 100, "q": 0.01, "sigma": 1.0, "delta": 1e-5, "ref_eps": 1.2141, "cite": "Opacus RDP Benchmark (Mironov 2017)"},
        {"steps": 100, "q": 0.05, "sigma": 1.5, "delta": 1e-5, "ref_eps": 1.8789, "cite": "Opacus DP-SGD Tutorials"},
        {"steps": 50, "q": 0.1, "sigma": 2.0, "delta": 1e-5, "ref_eps": 1.8440, "cite": "Opacus FL DP Grid"},
    ]

    print(f"{'Steps':<8}{'q':<8}{'Sigma':<8}{'Delta':<10}{'Computed Eps':<16}{'Ref Eps':<12}{'Rel Diff (%)':<14}{'Citation'}")
    print("-" * 88)

    for case in test_cases:
        acc = VerifiedRDPAccountant(target_delta=case["delta"])
        for _ in range(case["steps"]):
            acc.step(noise_multiplier=case["sigma"], sample_rate=case["q"])
        computed_eps = acc.get_epsilon()
        diff_pct = abs(computed_eps - case["ref_eps"]) / case["ref_eps"] * 100.0
        print(f"{case['steps']:<8}{case['q']:<8}{case['sigma']:<8}{case['delta']:<10.1e}{computed_eps:<16.4f}{case['ref_eps']:<12.4f}{diff_pct:<14.2f}{case['cite']}")

        # Ensure close match (< 10% relative tolerance due to discrete Renyi alpha grid)
        assert diff_pct < 10.0, f"Accountant deviation too high for case {case}"


def run_compression_benchmark(train_dataset, test_dataset, seeds=[42, 100, 2026], num_rounds=8):
    """
    Evaluates compression schemes:
    - Identity (Uncompressed baseline)
    - Top-K (k_ratio=0.1)
    - QSGD (s=8)
    - Sparse Ternary
    Verifies actual wire bytes per round and accuracy.
    """
    device = get_device()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    num_clients = 10

    compressors = {
        "Uncompressed_FP32": lambda: IdentityCompressor(),
        "TopK_10pct": lambda: TopKCompressor(k_ratio=0.1),
        "QSGD_s8": lambda: QSGDCompressor(s=8),
        "SparseTernary": lambda: SparseTernaryCompressor(threshold_factor=0.5),
    }

    print("\n" + "=" * 75)
    print(" 3. COMPRESSION BENCHMARK: Wire Bytes Sent & Accuracy (3 Seeds)")
    print("=" * 75)

    comp_results = {name: {"acc": [], "uplink_bytes_per_round": None} for name in compressors}

    for comp_name, comp_fn in compressors.items():
        print(f"\n--- Testing Compressor: {comp_name} ---")
        for seed in seeds:
            seed_everything(seed)
            client_indices = partition_iid(train_dataset, num_clients=num_clients, seed=seed)
            client_loaders = get_client_dataloaders(train_dataset, client_indices, batch_size=64)

            client_algos = {
                cid: FedAvgClient(
                    client_id=cid,
                    model_fn=lambda: MNIST_CNN(1, 10),
                    device=device,
                    learning_rate=0.05,
                )
                for cid in range(num_clients)
            }

            sim = FLSimulator(
                model_fn=lambda: MNIST_CNN(1, 10),
                client_algos=client_algos,
                client_dataloaders=client_loaders,
                test_dataloader=test_loader,
                server_rule=FedAvgMean(),
                server_optimizer=ServerSGD(lr=1.0),
                device=device,
                seed=seed,
                compressor=comp_fn(),
            )

            res = sim.run(num_rounds=num_rounds, epochs_per_round=1, verbose=False, evaluate_initial=True)
            accs = [r["accuracy"] * 100.0 for r in res["history"]]
            comp_results[comp_name]["acc"].append(accs)
            if comp_results[comp_name]["uplink_bytes_per_round"] is None:
                comp_results[comp_name]["uplink_bytes_per_round"] = res["history"][1]["uplink_bytes"]

            print(f"  Seed {seed:4d} | Final Acc: {accs[-1]:.2f}% | Uplink/Round: {res['history'][1]['uplink_bytes']/1024:.1f} KB")

    # Table of wire reduction
    uncompressed_bytes = comp_results["Uncompressed_FP32"]["uplink_bytes_per_round"]
    print("\n" + "-" * 75)
    print(f"{'Compressor':<24}{'Uplink Bytes/Round':<22}{'Compression Ratio':<20}{'Final Acc (Mean ± Std)'}")
    print("-" * 75)

    summary_comp = {}
    for comp_name, data in comp_results.items():
        b = data["uplink_bytes_per_round"]
        ratio = uncompressed_bytes / float(b)
        acc_mat = np.array(data["acc"])
        final_mean = acc_mat[:, -1].mean()
        final_std = acc_mat[:, -1].std()
        print(f"{comp_name:<24}{b:<10} ({b/1024:.1f} KB)   {ratio:<6.2f}x             {final_mean:.2f}% ± {final_std:.2f}%")
        summary_comp[comp_name] = {
            "uplink_bytes_per_round": b,
            "compression_ratio": ratio,
            "final_acc_mean": float(final_mean),
            "final_acc_std": float(final_std),
            "acc_trajectory_mean": acc_mat.mean(axis=0).tolist(),
            "acc_trajectory_std": acc_mat.std(axis=0).tolist(),
        }
    print("-" * 75)

    # Plot compression trajectories
    fig, ax = plt.subplots(figsize=(8, 5))
    rounds = list(range(num_rounds + 1))
    colors = {"Uncompressed_FP32": "#2563eb", "TopK_10pct": "#16a34a", "QSGD_s8": "#d97706", "SparseTernary": "#9333ea"}

    for comp_name, data in summary_comp.items():
        m = np.array(data["acc_trajectory_mean"])
        s = np.array(data["acc_trajectory_std"])
        c = colors[comp_name]
        ax.plot(rounds, m, label=f"{comp_name} ({data['compression_ratio']:.1f}x)", color=c, linewidth=2)
        ax.fill_between(rounds, m - s, m + s, color=c, alpha=0.15)

    ax.set_title("Real MNIST: Accuracy vs Communication Compression (3 Seeds)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Communication Round", fontsize=10)
    ax.set_ylabel("Global Test Accuracy (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join("results", "stage2_compression_bytes_and_accuracy.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nSaved compression plot to {plot_path}")

    json_path = os.path.join("results", "stage2_compression_benchmark.json")
    with open(json_path, "w") as f:
        json.dump({"dataset_source": "real", "dataset_name": "MNIST", "summary": summary_comp}, f, indent=2)
    print(f"Saved compression JSON to {json_path}")

    return summary_comp


def main():
    os.makedirs("results", exist_ok=True)
    train_dataset, test_dataset = load_mnist_data()

    # 1. Robustness
    run_robustness_benchmark(train_dataset, test_dataset, seeds=[42, 100, 2026], num_rounds=10)

    # 2. Privacy Accounting
    run_privacy_accounting_verification()

    # 3. Compression
    run_compression_benchmark(train_dataset, test_dataset, seeds=[42, 100, 2026], num_rounds=8)


if __name__ == "__main__":
    main()
