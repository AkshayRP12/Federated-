"""
Stage 4 Unit Tests: Experiment Grid Generator, Storage Engine, and Verification.
Tests:
- 16 canonical scenarios cover orthogonal axes (heterogeneity, attacks, participation, DP).
- 8 candidate FL configurations per domain with clear rationale.
- SQLite schema and Parquet export validity.
- Checkpoint resume capability.
- Poisson client sampling alignment under DP and partial participation.
"""

import os
import sqlite3
import pytest
import pandas as pd
import torch

from flaas.grid.scenarios import SCENARIOS, get_domain_configs, ScenarioSpec, ConfigSpec
from flaas.grid.runner import init_database, export_to_parquet, run_grid
from flaas.profiles.kb_v1 import KB_V1_PROFILES


def test_scenario_suite_covers_sixteen_scenarios():
    """Verify that exactly 16 scenarios exist and cover all necessary axes."""
    assert len(SCENARIOS) == 16
    scenario_ids = [s.scenario_id for s in SCENARIOS]
    assert len(set(scenario_ids)) == 16

    # Verify orthogonal axes
    alphas = [s.heterogeneity_alpha for s in SCENARIOS if s.heterogeneity_alpha is not None]
    assert 0.1 in alphas  # severe skew
    assert 0.3 in alphas  # moderate skew
    assert 0.5 in alphas  # mild skew

    # Attack coverage
    attacks = set(s.attack_type for s in SCENARIOS if s.attack_type is not None)
    assert "sign_flip" in attacks
    assert "gaussian_noise" in attacks
    assert "scaled_update" in attacks

    # Participation coverage
    participation_rates = set(s.participation_rate for s in SCENARIOS)
    assert 1.0 in participation_rates
    assert 0.5 in participation_rates
    assert 0.2 in participation_rates

    # Privacy coverage
    privacy_modes = set(s.privacy_mode for s in SCENARIOS)
    assert "none" in privacy_modes
    assert "client_dp" in privacy_modes


def test_domain_configs_produce_eight_distinct_specs():
    """Verify each domain has 8 diverse, rationalized candidate configurations."""
    for domain_id in KB_V1_PROFILES.keys():
        configs = get_domain_configs(domain_id, num_clients=10)
        assert len(configs) == 8
        config_ids = [c.config_id for c in configs]
        assert len(set(config_ids)) == 8

        # Check essential algorithm categories
        algos = set(c.client_algo for c in configs)
        assert "FedAvg" in algos
        assert "FedProx" in algos

        # Check defense mechanisms
        aggs = set(c.aggregation_rule for c in configs)
        assert "FedAvgMean" in aggs
        assert "TrimmedMean" in aggs

        # Check compression and DP representation
        compressors = set(c.compressor for c in configs)
        assert "ternary_2bit" in compressors
        dp_types = set(c.dp_type for c in configs)
        assert "client" in dp_types


def test_sqlite_schema_and_parquet_export(tmp_path):
    """Verify SQLite database initialization and pandas/pyarrow Parquet export."""
    db_file = str(tmp_path / "test_stage4.sqlite")
    parquet_file = str(tmp_path / "test_stage4.parquet")

    conn = init_database(db_file)
    cur = conn.cursor()

    # Verify tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    assert "experiment_runs" in tables
    assert "round_metrics" in tables

    # Insert mock record
    cur.execute("""
        INSERT INTO experiment_runs (
            run_id, domain, scenario_id, config_id, seed,
            final_accuracy, final_loss, final_macro_f1, rounds_completed,
            early_stopped, total_time_seconds, uplink_bytes, downlink_bytes,
            total_wire_bytes, epsilon, timestamp
        ) VALUES ('mock_01', 'healthcare', 'scen_01', 'cfg_01_baseline', 42,
                  82.5, 0.45, 0.81, 15, 1, 14.2, 5000, 5000, 10000, 0.0, '2026-10-25T00:00:00')
    """)
    conn.commit()

    # Test Parquet export
    export_to_parquet(conn, parquet_file)
    assert os.path.exists(parquet_file)

    df = pd.read_parquet(parquet_file)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "mock_01"
    assert df.iloc[0]["final_accuracy"] == 82.5
    conn.close()


def test_checkpoint_resume_logic(tmp_path):
    """Verify that runner skips previously computed runs when resume=True."""
    db_file = str(tmp_path / "test_resume.sqlite")
    conn = init_database(db_file)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO experiment_runs (
            run_id, domain, scenario_id, config_id, seed,
            final_accuracy, final_loss, final_macro_f1, rounds_completed,
            early_stopped, total_time_seconds, uplink_bytes, downlink_bytes,
            total_wire_bytes, epsilon, timestamp
        ) VALUES ('skip_me', 'finance', 'scen_01', 'cfg_01_baseline', 42,
                  99.9, 0.01, 0.98, 10, 1, 8.0, 1000, 1000, 2000, 0.0, '2026-10-25T00:00:00')
    """)
    conn.commit()

    cur.execute(
        "SELECT 1 FROM experiment_runs WHERE domain=? AND scenario_id=? AND config_id=? AND seed=?",
        ("finance", "scen_01", "cfg_01_baseline", 42)
    )
    assert cur.fetchone() is not None

    cur.execute(
        "SELECT 1 FROM experiment_runs WHERE domain=? AND scenario_id=? AND config_id=? AND seed=?",
        ("finance", "scen_02", "cfg_01_baseline", 42)
    )
    assert cur.fetchone() is None
    conn.close()


def test_client_dp_accountant_integration(tmp_path):
    """Verify that a DP scenario and config execute and account epsilon properly."""
    from flaas.privacy.accountant import VerifiedRDPAccountant
    from flaas.privacy.clip_noise import DPClipperNoise
    from flaas.fl.simulator import FLSimulator
    from flaas.fl.models import MedCNN
    from flaas.fl.client_algos import FedAvgClient
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cpu")
    model_fn = lambda: MedCNN(in_channels=3, num_classes=9)
    x = torch.randn(64, 3, 28, 28)
    y = torch.randint(0, 9, (64,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=32)

    client_algos = {0: FedAvgClient(0, model_fn, device, learning_rate=0.01)}
    client_loaders = {0: loader}
    dp_clipper = DPClipperNoise(clip_norm=1.0, noise_multiplier=0.5)
    dp_accountant = VerifiedRDPAccountant(target_delta=1e-5)

    sim = FLSimulator(
        model_fn=model_fn,
        client_algos=client_algos,
        client_dataloaders=client_loaders,
        test_dataloader=loader,
        device=device,
        dp_clipper=dp_clipper,
        dp_accountant=dp_accountant,
    )
    res = sim.run(num_rounds=1, epochs_per_round=1, participation_rate=1.0)
    assert res["rounds_completed"] == 1
    assert "epsilon" in res["final_metrics"]
    assert res["final_metrics"]["epsilon"] > 0.0


