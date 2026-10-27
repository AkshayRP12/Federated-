"""
Stage 4 Experiment Grid Runner & Storage Engine.
Executes multi-domain FL scenario evaluations, handles checkpointing,
early stopping, and persists results to SQLite and Parquet.
"""

from typing import Dict, Any, List, Optional
import os
import sys
import time
import argparse
import sqlite3
import datetime
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from flaas.data.domain_adapters import get_domain_adapter
from flaas.data.partitioners import partition_dirichlet, partition_iid
from flaas.profiles.kb_v1 import get_profile, KB_V1_PROFILES
from flaas.grid.scenarios import SCENARIOS, get_domain_configs, ScenarioSpec, ConfigSpec
from flaas.fl.simulator import FLSimulator, seed_everything
from flaas.fl.client_algos import FedAvgClient, FedProxClient, SCAFFOLDClient, FedNovaClient
from flaas.fl.server_rules import FedAvgMean, TrimmedMean, Krum
from flaas.fl.server_optimizers import ServerSGD, ServerAdam
from flaas.attacks.attacks import SignFlipAttack, GaussianNoiseAttack, ScaledUpdateAttack
from flaas.compression.compressors import IdentityCompressor, ThresholdedTernaryCompressor
from flaas.privacy.clip_noise import DPClipperNoise
from flaas.privacy.accountant import VerifiedRDPAccountant
from flaas.fl.models import MedCNN, FraudMLP, HARMLP, MobileCNN, TrafficSignCNN


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


def init_database(db_path: str = "results/stage4_grid.sqlite") -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiment_runs (
            run_id TEXT PRIMARY KEY,
            domain TEXT,
            scenario_id TEXT,
            config_id TEXT,
            seed INTEGER,
            final_accuracy REAL,
            final_loss REAL,
            final_macro_f1 REAL,
            rounds_completed INTEGER,
            early_stopped INTEGER,
            total_time_seconds REAL,
            uplink_bytes INTEGER,
            downlink_bytes INTEGER,
            total_wire_bytes INTEGER,
            epsilon REAL,
            timestamp TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS round_metrics (
            run_id TEXT,
            round_idx INTEGER,
            accuracy REAL,
            loss REAL,
            macro_f1 REAL,
            round_time REAL,
            uplink_bytes INTEGER,
            PRIMARY KEY (run_id, round_idx)
        )
    """)
    conn.commit()
    return conn


def export_to_parquet(conn: sqlite3.Connection, parquet_path: str = "results/stage4_grid.parquet") -> None:
    df = pd.read_sql_query("SELECT * FROM experiment_runs", conn)
    df.to_parquet(parquet_path, index=False)


def run_grid(
    domains: Optional[List[str]] = None,
    seeds: List[int] = [42],
    max_rounds: int = 20,
    resume: bool = True,
    db_path: str = "results/stage4_grid.sqlite",
    parquet_path: str = "results/stage4_grid.parquet",
    verbose_rounds: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.zeros(1, device=device)  # Pre-warm GPU context
    conn = init_database(db_path)
    cur = conn.cursor()

    core_domains = ["healthcare", "finance", "iot_edge", "mobile_apps"]
    active_domains = domains or core_domains
    print("=" * 90)
    print(f"Starting FLaaS Stage 4 Experiment Grid Runner on [{device}]")
    print(f"Domains: {active_domains}")
    print(f"Seeds: {seeds} | Max Rounds: {max_rounds} | Resume: {resume}")
    print(f"Database: {db_path} | Parquet: {parquet_path}")
    print("=" * 90)

    total_runs_planned = len(active_domains) * len(SCENARIOS) * 8 * len(seeds)
    run_counter = 0
    executed_counter = 0
    grid_start_time = time.perf_counter()

    for domain_id in active_domains:
        adapter = get_domain_adapter(domain_id)
        profile = get_profile(domain_id)
        train_ds, test_ds = adapter.load_data()
        num_classes = adapter.num_classes
        num_clients = profile.typical_num_clients
        configs = get_domain_configs(domain_id, num_clients)

        # Pre-create test loader (reused across runs)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        for scenario in SCENARIOS:
            for config in configs:
                for seed in seeds:
                    run_counter += 1
                    run_id = f"{domain_id}_{scenario.scenario_id}_{config.config_id}_s{seed}"

                    if resume:
                        cur.execute(
                            "SELECT 1 FROM experiment_runs WHERE domain=? AND scenario_id=? AND config_id=? AND seed=?",
                            (domain_id, scenario.scenario_id, config.config_id, seed),
                        )
                        if cur.fetchone() is not None:
                            continue

                    seed_everything(seed)
                    
                    # 1. Partition Data
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
                    # Apply 1,000-sample cap (with safe non-empty fallback)
                    capped_indices = {
                        cid: idxs[:1000] if len(idxs) > 0 else [0]
                        for cid, idxs in client_indices.items()
                    }

                    client_loaders = {
                        cid: DataLoader(Subset(train_ds, idxs), batch_size=config.client_algo_kwargs.get("batch_size", 32), shuffle=True)
                        for cid, idxs in capped_indices.items()
                    }

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

                    # 3. Server Aggregation Rule
                    if config.aggregation_rule == "TrimmedMean":
                        server_rule = TrimmedMean(beta=config.aggregation_kwargs.get("beta", 0.2))
                    elif config.aggregation_rule == "MultiKrum":
                        f_krum = config.aggregation_kwargs.get("f", 1)
                        m_krum = config.aggregation_kwargs.get("m", max(1, num_clients - f_krum - 2))
                        server_rule = Krum(f=f_krum, multi_krum=True, m=m_krum)
                    else:
                        server_rule = FedAvgMean()

                    # 4. Server Optimizer
                    server_optimizer = ServerSGD(lr=config.server_opt_kwargs.get("lr", 1.0))

                    # 5. Compressor
                    if config.compressor == "ternary_2bit":
                        compressor = ThresholdedTernaryCompressor(threshold_factor=1.0)
                    else:
                        compressor = IdentityCompressor()

                    # 6. Privacy & Accountant
                    dp_clipper = None
                    dp_accountant = None
                    if scenario.privacy_mode == "client_dp" or config.dp_type == "client":
                        clip_norm = config.dp_kwargs.get("clip_norm", 1.0)
                        noise_mult = config.dp_kwargs.get("noise_multiplier", 0.5)
                        dp_clipper = DPClipperNoise(clip_norm=clip_norm, noise_multiplier=noise_mult)
                        dp_accountant = VerifiedRDPAccountant(
                            target_delta=profile.privacy_delta,
                        )

                    # 7. Adversarial Attacks
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
                        server_optimizer=server_optimizer,
                        device=device,
                        seed=seed,
                        adversary_ids=adversary_ids,
                        attack=attack,
                        dp_clipper=dp_clipper,
                        dp_accountant=dp_accountant,
                        compressor=compressor,
                    )

                    # 9. Execute Run
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
                        target_accuracy=profile.target_accuracy,
                        verbose=verbose_rounds,
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

                    elapsed = time.perf_counter() - grid_start_time
                    rate = executed_counter / max(1e-5, elapsed)
                    print(
                        f"[{executed_counter:4d} runs] {run_id:50s} | "
                        f"Acc: {final_acc:5.1f}% | Time: {run_duration:4.1f}s | "
                        f"Rate: {rate:4.2f} runs/s",
                        flush=True,
                    )

                    if executed_counter % 10 == 0:
                        export_to_parquet(conn, parquet_path)

    export_to_parquet(conn, parquet_path)
    total_grid_time = time.perf_counter() - grid_start_time
    print("=" * 90)
    print(f"Stage 4 Grid Execution COMPLETE! Executed: {executed_counter} runs in {total_grid_time/60:.2f} minutes.")
    print(f"Stored SQLite: {db_path} | Parquet: {parquet_path}")
    print("=" * 90)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4 Experiment Grid Runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of seeds (e.g. 42 or 42 43 44)")
    parser.add_argument("--domains", type=str, nargs="+", default=None, help="Specific domains to run")
    parser.add_argument("--max-rounds", type=int, default=15, help="Maximum rounds per run")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume")
    parser.add_argument("--db-path", type=str, default="results/stage4_grid.sqlite")
    parser.add_argument("--parquet-path", type=str, default="results/stage4_grid.parquet")
    args = parser.parse_args()

    run_grid(
        domains=args.domains,
        seeds=args.seeds,
        max_rounds=args.max_rounds,
        resume=not args.no_resume,
        db_path=args.db_path,
        parquet_path=args.parquet_path,
    )
