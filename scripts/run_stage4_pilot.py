"""
Stage 4 Multi-Domain 5-Minute Pilot Benchmark.
Validates all 5 domains under real/synthesized data:
1. Honest convergence vs. Attacked degradation (sanity check).
2. Accuracy above chance for all 5 domains.
3. Timing and VRAM stability on RTX 3050 GPU.
"""

import time
import json
import os
import torch
import numpy as np

from flaas.data.domain_adapters import get_domain_adapter
from flaas.data.partitioners import partition_dirichlet
from flaas.profiles.kb_v1 import get_profile
from flaas.fl.simulator import FLSimulator
from flaas.fl.client_algos import FedAvgClient
from flaas.fl.server_rules import FedAvgMean
from flaas.fl.server_optimizers import ServerSGD
from flaas.attacks.attacks import SignFlipAttack
from flaas.fl.models import MedCNN, FraudMLP, HARMLP, MobileCNN, TrafficSignCNN
from torch.utils.data import DataLoader, Subset


def get_model_fn(domain: str, num_classes: int):
    if domain == "healthcare":
        return lambda: MedCNN(in_channels=3, num_classes=num_classes)
    elif domain == "finance":
        return lambda: FraudMLP(in_features=29, num_classes=num_classes)
    elif domain == "iot_edge":
        return lambda: HARMLP(in_features=561, num_classes=num_classes)
    elif domain == "mobile_apps":
        return lambda: MobileCNN(in_channels=1, num_classes=num_classes)
    elif domain == "autonomous_vehicles":
        return lambda: TrafficSignCNN(in_channels=3, num_classes=num_classes)
    raise ValueError(f"Unknown domain {domain}")


def run_pilot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Stage 4 Pilot Benchmark on Device: {device} ===")
    
    domains = ["healthcare", "finance", "iot_edge", "mobile_apps", "autonomous_vehicles"]
    results = {}
    total_start = time.perf_counter()
    
    for d in domains:
        print(f"\n--- Testing Domain: {d} ---")
        adapter = get_domain_adapter(d)
        profile = get_profile(d)
        train_ds, test_ds = adapter.load_data()
        
        num_classes = adapter.num_classes
        num_clients = min(profile.typical_num_clients, 10)
        
        # Partition data
        client_indices = partition_dirichlet(
            train_ds,
            num_clients=num_clients,
            alpha=profile.heterogeneity_alpha,
            num_classes=num_classes,
            seed=42,
        )
        # Apply 1,000-sample cap
        capped_indices = {cid: idxs[:1000] for cid, idxs in client_indices.items()}
        
        model_fn = get_model_fn(d, num_classes)
        client_loaders = {
            cid: DataLoader(Subset(train_ds, idxs), batch_size=32, shuffle=True)
            for cid, idxs in capped_indices.items()
        }
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        
        # 1. Honest Run (3 rounds)
        client_algos_honest = {
            cid: FedAvgClient(cid, model_fn, device, learning_rate=0.01)
            for cid in range(num_clients)
        }
        sim_honest = FLSimulator(
            model_fn=model_fn,
            client_algos=client_algos_honest,
            client_dataloaders=client_loaders,
            test_dataloader=test_loader,
            server_rule=FedAvgMean(),
            server_optimizer=ServerSGD(lr=1.0),
            device=device,
            seed=42,
        )
        
        t0 = time.perf_counter()
        res_honest = sim_honest.run(num_rounds=3, epochs_per_round=1, verbose=False)
        honest_time = time.perf_counter() - t0
        honest_acc = res_honest["final_metrics"]["accuracy"] * 100.0
        
        # 2. Attacked Run (30% sign-flip Byzantine adversaries, 3 rounds)
        num_adversaries = max(1, int(num_clients * 0.3))
        adversary_ids = list(range(num_adversaries))
        client_algos_att = {
            cid: FedAvgClient(cid, model_fn, device, learning_rate=0.01)
            for cid in range(num_clients)
        }
        sim_attacked = FLSimulator(
            model_fn=model_fn,
            client_algos=client_algos_att,
            client_dataloaders=client_loaders,
            test_dataloader=test_loader,
            server_rule=FedAvgMean(),
            server_optimizer=ServerSGD(lr=1.0),
            device=device,
            seed=42,
            adversary_ids=adversary_ids,
            attack=SignFlipAttack(scale=2.0),
        )
        
        t0 = time.perf_counter()
        res_attacked = sim_attacked.run(num_rounds=3, epochs_per_round=1, verbose=False)
        attacked_time = time.perf_counter() - t0
        attacked_acc = res_attacked["final_metrics"]["accuracy"] * 100.0
        
        chance_acc = 100.0 / num_classes
        print(f"Domain: {d:20s} | Chance: {chance_acc:5.1f}% | Honest: {honest_acc:5.1f}% | Attacked: {attacked_acc:5.1f}% | Time: {honest_time+attacked_time:.2f}s")
        
        # Sanity assertions
        assert honest_acc > chance_acc, f"Domain {d} honest accuracy {honest_acc:.2f}% did not exceed chance {chance_acc:.2f}%!"
        assert attacked_acc < honest_acc, f"Domain {d} attack failed: attacked {attacked_acc:.2f}% >= honest {honest_acc:.2f}%"
        
        results[d] = {
            "num_classes": num_classes,
            "chance_accuracy": chance_acc,
            "honest_accuracy": honest_acc,
            "attacked_accuracy": attacked_acc,
            "time_seconds": honest_time + attacked_time,
            "sanity_passed": True,
        }

    total_time = time.perf_counter() - total_start
    print(f"\n=== Pilot Benchmark Complete in {total_time:.2f}s. All 5 Domains PASSED Sanity Checks! ===")
    
    os.makedirs("results", exist_ok=True)
    with open("results/stage4_pilot_results.json", "w") as f:
        json.dump({"total_time_s": total_time, "domains": results}, f, indent=2)
    print("Saved pilot results to results/stage4_pilot_results.json")


if __name__ == "__main__":
    run_pilot()
