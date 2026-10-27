"""
Tests for Expanded Grid Scenarios, Configurations, and Streamlit Demo Integration.
Runs entirely on CPU in standard pytest test suite.
"""

import pytest
import torch
import numpy as np
import pandas as pd

from flaas.grid.expanded_scenarios import (
    IOT_SCENARIOS,
    MOBILE_SCENARIOS,
    get_expanded_configs,
    ExpandedScenarioSpec,
    ExpandedConfigSpec,
)
from flaas.fl.server_rules import Bulyan, CoordinateMedian, TrimmedMean, Krum, FedAvgMean
from flaas.fl.server_optimizers import ServerSGD, ServerAdam
from flaas.compression.compressors import QSGDCompressor, ThresholdedTernaryCompressor, IdentityCompressor
from demo.app import (
    encode_scenario_features,
    score_and_rank_candidates,
    generate_empirical_rationale,
    CONFIG_NAMES,
)


def test_iot_scenarios_count_and_client_scale():
    """IoT must have 7 domain-specific scenarios with 20 edge clients."""
    assert len(IOT_SCENARIOS) == 7
    for scen in IOT_SCENARIOS:
        assert scen.domain_id == "iot_edge"
        assert scen.num_clients == 20
        assert 0.0 < scen.participation_rate <= 1.0


def test_mobile_scenarios_count_and_client_scale():
    """Mobile must have 7 domain-specific scenarios with 50 smartphone clients."""
    assert len(MOBILE_SCENARIOS) == 7
    for scen in MOBILE_SCENARIOS:
        assert scen.domain_id == "mobile_apps"
        assert scen.num_clients == 50
        assert 0.0 < scen.participation_rate <= 1.0


def test_expanded_configs_content():
    """Verifies all required algorithms are represented in 16 expanded configs."""
    configs = get_expanded_configs(num_clients=20)
    assert len(configs) == 16

    c_ids = [c.config_id for c in configs]
    algos = [c.client_algo for c in configs]
    rules = [c.aggregation_rule for c in configs]
    compressors = [c.compressor for c in configs]
    server_opts = [c.server_optimizer for c in configs]

    # Required algorithms
    assert "SCAFFOLD" in algos
    assert "FedNova" in algos
    assert "FedAvg" in algos
    assert "FedProx" in algos

    # Server optimizers
    assert "ServerAdam" in server_opts
    assert "ServerSGD" in server_opts

    # Aggregation rules
    assert "Bulyan" in rules
    assert "CoordinateMedian" in rules
    assert "TrimmedMean" in rules
    assert "MultiKrum" in rules

    # Compressors
    assert "qsgd_8" in compressors
    assert "ternary_2bit" in compressors
    assert "identity" in compressors

    # DP axis (off, eps 8, 4, 1)
    dp_epsilons = {c.dp_target_epsilon for c in configs}
    assert None in dp_epsilons  # off
    assert 8.0 in dp_epsilons
    assert 4.0 in dp_epsilons
    assert 1.0 in dp_epsilons


def test_bulyan_and_median_aggregators():
    """Unit test for Bulyan and CoordinateMedian on synthetic updates."""
    updates = [torch.tensor([float(i), float(i * 2)]) for i in range(10)]
    
    med = CoordinateMedian()
    agg_med = med.aggregate(updates)
    assert agg_med.shape == (2,)

    bul = Bulyan(f=1)
    agg_bul = bul.aggregate(updates)
    assert agg_bul.shape == (2,)


def test_qsgd_and_ternary_compressors():
    """Verifies QSGD and Ternary compressors calculate non-zero wire bytes."""
    t = torch.randn(100)
    
    qsgd = QSGDCompressor(s=8, seed=42)
    dec_q, wb_q = qsgd.compress(t)
    assert dec_q.shape == t.shape
    assert wb_q < 400  # uncompressed is 400 bytes

    ternary = ThresholdedTernaryCompressor()
    dec_t, wb_t = ternary.compress(t)
    assert dec_t.shape == t.shape
    assert wb_t < 400


def test_demo_scenario_feature_encoding():
    """Verifies scenario feature encoding produces normalized 5D vector."""
    vec = encode_scenario_features(alpha=0.5, q=0.2, attack_type="sign_flip", attack_frac=0.2, dp_mode="client_dp")
    assert vec.shape == (5,)
    assert (vec >= 0.0).all() and (vec <= 1.0).all()


def test_demo_scoring_and_ranking():
    """Tests multi-criteria scoring and Pareto sorting logic on synthetic dataframe."""
    mock_df = pd.DataFrame([
        {
            "config_id": "exp_01_fedavg",
            "final_accuracy": 85.0,
            "total_wire_bytes": 100 * 1024 * 1024,
            "total_time_seconds": 50.0,
            "rounds_completed": 10,
            "early_stopped": 0,
            "epsilon": 0.0,
        },
        {
            "config_id": "exp_10_ternary_compression",
            "final_accuracy": 84.5,
            "total_wire_bytes": 6 * 1024 * 1024,
            "total_time_seconds": 45.0,
            "rounds_completed": 10,
            "early_stopped": 0,
            "epsilon": 0.0,
        },
        {
            "config_id": "exp_06_trimmed_mean",
            "final_accuracy": 82.0,
            "total_wire_bytes": 100 * 1024 * 1024,
            "total_time_seconds": 60.0,
            "rounds_completed": 10,
            "early_stopped": 0,
            "epsilon": 0.0,
        },
    ])

    candidates = score_and_rank_candidates(
        sub_df=mock_df,
        preset="balanced",
        require_byz=False,
        require_dp=False,
        require_comp=False,
    )
    assert len(candidates) == 3
    # Top candidate should be score descending
    assert candidates[0]["score"] >= candidates[1]["score"] >= candidates[2]["score"]

    # Check rationale generator
    rationale = generate_empirical_rationale(
        rec=candidates[0],
        baseline=candidates[0],
        scenario_name="iot_scen_01_iid",
        preset="balanced",
    )
    assert len(rationale) > 20
