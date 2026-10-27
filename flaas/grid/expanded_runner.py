"""
Expanded Grid Runner for Mobile and IoT Domains.
Executes expanded scenarios and configurations:
- IoT first, then Mobile.
- 1 seed (seed=42).
- Persists to results/stage4_grid.sqlite and results/stage4_grid.parquet.
- Does not modify frozen kb-v1 profiles.
"""

from typing import List, Dict, Any, Optional
import os
import sys
import time
import datetime
import sqlite3
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from flaas.data.domain_adapters import get_domain_adapter
from flaas.data.partitioners import partition_dirichlet, partition_iid
from flaas.fl.simulator import FLSimulator, seed_everything
from flaas.fl.client_algos import FedAvgClient, FedProxClient, SCAFFOLDClient, FedNovaClient
from flaas.fl.server_rules import FedAvgMean, TrimmedMean, Krum, Bulyan, CoordinateMedian
from flaas.fl.server_optimizers import ServerSGD, ServerAdam
from flaas.attacks.attacks import SignFlipAttack, GaussianNoiseAttack, ScaledUpdateAttack
from flaas.compression.compressors import IdentityCompressor, ThresholdedTernaryCompressor, QSGDCompressor
from flaas.privacy.clip_noise import DPClipperNoise
from flaas.privacy.accountant import VerifiedRDPAccountant
from flaas.fl.models import HARMLP, MobileCNN
from flaas.grid.expanded_scenarios import (
    IOT_SCENARIOS,
    MOBILE_SCENARIOS,
    get_expanded_configs,
    ExpandedScenarioSpec,
    ExpandedConfigSpec,
)
from flaas.grid.runner import init_database, export_to_parquet


def get_model_fn(domain_id: str, num_classes: int):
    if domain_id == "iot_edge":
        return lambda: HARMLP(in_features=561, num_classes=num_classes)
    elif domain_id == "mobile_apps":
        return lambda: MobileCNN(in_channels=1, num_classes=num_classes)
    raise ValueError(f"Expanded runner only supports iot_edge and mobile_apps, got {domain_id}")


def run_expanded_grid(
    db_path: str = "results/stage4_grid.sqlite",
    parquet_path: str = "results/stage4_grid.parquet",
    seed: int = 42,
    max_rounds: int = 10,
    resume: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.zeros(1, device=device)

    conn = init_database(db_path)
    cur = conn.cursor()

    # Domains executed sequentially: IoT first, then Mobile
    domain_plan = [
        ("iot_edge", IOT_SCENARIOS, 88.0),
        ("mobile_apps", MOBILE_SCENARIOS, 85.0),
    ]

    print("=" * 85)
    print(f"Starting FLaaS Expanded Grid Runner on [{device}] (1 seed={seed})")
    print(f"Domain Execution Order: [1. iot_edge, 2. mobile_apps]")
    print(f"Database: {db_path} | Parquet: {parquet_path}")
    print("=" * 85)

    executed_counter = 0
    start_time = time.perf_counter()

    for domain_id, scenarios, target_acc in domain_plan:
        adapter = get_domain_adapter(domain_id)
        train_ds, test_ds = adapter.load_data()
        num_classes = adapter.num_classes
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        print(f"\n>>> Running Domain: {domain_id.upper()} ({len(scenarios)} scenarios) <<<")

        for scenario in scenarios:
            num_clients = scenario.num_clients
            configs = get_expanded_configs(num_clients)

            # 1. Partition Data
            seed_everything(seed)
            if scenario.heterogeneity_alpha is None:
                client_indices = partition_iid(train_ds, num_clients=num_clients, seed=seed)
            else:
                client_indices = partition_dirichlet(
                    train_ds,
                    num_clients=num_clients,
                    alpha=scenario.heterogeneity_alpha,
                    num_classes=num_classes,
                    seed=seed,
                )

            capped_indices = {
                cid: idxs[:1000] if len(idxs) > 0 else [0]
                for cid, idxs in client_indices.items()
            }
            client_loaders = {
                cid: DataLoader(Subset(train_ds, idxs), batch_size=32, shuffle=True)
                for cid, idxs in capped_indices.items()
            }

            for config in configs:
                run_id = f"{domain_id}_{scenario.scenario_id}_{config.config_id}_s{seed}"

                # Checkpoint resume
                if resume:
                    cur.execute(
                        "SELECT 1 FROM experiment_runs WHERE domain=? AND scenario_id=? AND config_id=? AND seed=?",
                        (domain_id, scenario.scenario_id, config.config_id, seed),
                    )
                    if cur.fetchone() is not None:
                        continue

                # 2. Build Client Algorithms
                model_fn = get_model_fn(domain_id, num_classes)
                lr = config.client_algo_kwargs.get("lr", 0.01)

                client_algos = {}
                for cid in range(num_clients):
                    if config.client_algo == "FedProx":
                        mu = config.client_algo_kwargs.get("mu", 0.01)
                        client_algos[cid] = FedProxClient(cid, model_fn, device, learning_rate=lr, mu=mu)
                    elif config.client_algo == "SCAFFOLD":
                        client_algos[cid] = SCAFFOLDClient(cid, model_fn, device, learning_rate=lr)
                    elif config.client_algo == "FedNova":
                        client_algos[cid] = FedNovaClient(cid, model_fn, device, learning_rate=lr)
                    else:
                        client_algos[cid] = FedAvgClient(cid, model_fn, device, learning_rate=lr)

                # 3. Server Aggregator
                rule_name = config.aggregation_rule
                if rule_name == "TrimmedMean":
                    server_rule = TrimmedMean(beta=config.aggregation_kwargs.get("beta", 0.2))
                elif rule_name == "MultiKrum":
                    f_krum = config.aggregation_kwargs.get("f", 1)
                    m_krum = config.aggregation_kwargs.get("m", max(1, num_clients - f_krum - 2))
                    server_rule = Krum(f=f_krum, multi_krum=True, m=m_krum)
                elif rule_name == "Bulyan":
                    f_bul = config.aggregation_kwargs.get("f", 1)
                    server_rule = Bulyan(f=f_bul)
                elif rule_name == "CoordinateMedian":
                    server_rule = CoordinateMedian()
                else:
                    server_rule = FedAvgMean()

                # 4. Server Optimizer (ServerSGD vs ServerAdam)
                if config.server_optimizer == "ServerAdam":
                    server_opt = ServerAdam(
                        lr=config.server_opt_kwargs.get("lr", 1.0),
                        beta1=config.server_opt_kwargs.get("beta1", 0.9),
                        beta2=config.server_opt_kwargs.get("beta2", 0.99),
                        eps=config.server_opt_kwargs.get("eps", config.server_opt_kwargs.get("epsilon", 1e-4)),
                    )
                else:
                    server_opt = ServerSGD(lr=config.server_opt_kwargs.get("lr", 1.0))

                # 5. Compressor
                if config.compressor == "ternary_2bit":
                    compressor = ThresholdedTernaryCompressor(threshold_factor=1.0)
                elif config.compressor == "qsgd_8":
                    compressor = QSGDCompressor(s=8)
                else:
                    compressor = IdentityCompressor()

                # 6. Privacy & Accountant (DP Axis: off, eps 8, eps 4, eps 1)
                dp_clipper = None
                dp_accountant = None
                if config.dp_type == "client" or scenario.privacy_mode == "client_dp":
                    clip_norm = config.dp_kwargs.get("clip_norm", 1.0)
                    noise_mult = config.dp_kwargs.get("noise_multiplier", 0.5)
                    dp_clipper = DPClipperNoise(clip_norm=clip_norm, noise_multiplier=noise_mult)
                    dp_accountant = VerifiedRDPAccountant(target_delta=1e-5)

                # 7. Attacks
                attack = None
                adversary_ids = []
                if scenario.attack_type is not None and scenario.attack_fraction > 0.0:
                    num_adv = max(1, int(num_clients * scenario.attack_fraction))
                    adversary_ids = list(range(num_adv))
                    if scenario.attack_type == "sign_flip":
                        attack = SignFlipAttack(scale=scenario.attack_scale)
                    elif scenario.attack_type == "gaussian_noise":
                        attack = GaussianNoiseAttack(std=scenario.attack_scale)
                    elif scenario.attack_type == "scaled_update":
                        attack = ScaledUpdateAttack(scale=scenario.attack_scale)

                # 8. Instantiate FL Simulator
                sim = FLSimulator(
                    model_fn=model_fn,
                    client_algos=client_algos,
                    client_dataloaders=client_loaders,
                    test_dataloader=test_loader,
                    server_rule=server_rule,
                    server_optimizer=server_opt,
                    device=device,
                    seed=seed,
                    adversary_ids=adversary_ids,
                    attack=attack,
                    dp_clipper=dp_clipper,
                    dp_accountant=dp_accountant,
                    compressor=compressor,
                )

                # 9. Run Simulation
                t0 = time.perf_counter()
                use_poisson = scenario.participation_rate < 1.0 and (
                    scenario.privacy_mode == "client_dp" or config.dp_type == "client"
                )
                sampling_strat = "poisson" if use_poisson else "fixed"

                sim_res = sim.run(
                    num_rounds=max_rounds,
                    epochs_per_round=config.client_algo_kwargs.get("epochs", 1),
                    participation_rate=scenario.participation_rate,
                    sampling_strategy=sampling_strat,
                    target_accuracy=target_acc,
                    verbose=False,
                )
                run_duration = time.perf_counter() - t0
                executed_counter += 1

                final_met = sim_res["final_metrics"]
                final_acc = final_met.get("accuracy", 0.0) * 100.0
                final_loss = final_met.get("loss", 0.0)
                final_f1 = final_met.get("macro_f1", 0.0)
                rounds_done = sim_res["rounds_completed"]
                early_stopped = 1 if sim_res["early_stopped"] else 0
                final_eps = final_met.get("epsilon", 0.0) or 0.0
                wire_sum = sim_res["wire_summary"]

                # 10. Persist to SQLite
                timestamp = datetime.datetime.now().isoformat()
                cur.execute("""
                    INSERT OR REPLACE INTO experiment_runs (
                        run_id, domain, scenario_id, config_id, seed,
                        final_accuracy, final_loss, final_macro_f1, rounds_completed,
                        early_stopped, total_time_seconds, uplink_bytes, downlink_bytes,
                        total_wire_bytes, epsilon, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, domain_id, scenario.scenario_id, config.config_id, seed,
                    final_acc, final_loss, final_f1, rounds_done,
                    early_stopped, run_duration,
                    wire_sum.get("uplink_bytes", wire_sum.get("total_uplink_bytes", 0)),
                    wire_sum.get("downlink_bytes", wire_sum.get("total_downlink_bytes", 0)),
                    wire_sum.get("total_bytes", wire_sum.get("total_wire_bytes", 0)),
                    final_eps, timestamp,
                ))

                for r in sim_res["history"]:
                    cur.execute("""
                        INSERT OR REPLACE INTO round_metrics (
                            run_id, round_idx, accuracy, loss, macro_f1, round_time, uplink_bytes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        run_id, r["round"], r["accuracy"] * 100.0, r["loss"],
                        r["macro_f1"], r["round_time"], r["uplink_bytes"],
                    ))

                conn.commit()

                elapsed = time.perf_counter() - start_time
                rate = executed_counter / max(1e-5, elapsed)
                print(
                    f"[{executed_counter:3d}] {run_id:52s} | "
                    f"Acc: {final_acc:5.1f}% | Time: {run_duration:4.1f}s | "
                    f"Rounds: {rounds_done:02d} | EarlyStop: {early_stopped}",
                    flush=True,
                )

                if executed_counter % 10 == 0:
                    export_to_parquet(conn, parquet_path)

    export_to_parquet(conn, parquet_path)
    total_time = time.perf_counter() - start_time
    print("=" * 85)
    print(f"Expanded Grid Complete! {executed_counter} runs executed in {total_time/60:.2f} minutes.")
    print("=" * 85)
    conn.close()


if __name__ == "__main__":
    run_expanded_grid()
