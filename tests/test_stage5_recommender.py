"""
Stage 5 Unit Tests: FLaaS Recommender, Domain Gate, Multi-Criteria Scoring,
Pareto Front, Plain-Language Rationale, and Leave-One-Scenario-Out Evaluation.
All tests run on CPU.
"""

import pytest
import numpy as np
import pandas as pd

from flaas.recommender.domain_gate import (
    ACTIVE_DOMAINS,
    PENDING_DOMAINS,
    DATASET_LABEL,
    DomainPendingFixesError,
    UnknownDomainError,
    is_domain_active,
    validate_domain_active,
    get_domain_status,
)
from flaas.recommender.feasibility import FeasibilityFilter, QueryConstraints
from flaas.recommender.scenario_matching import ScenarioMatcher, ScenarioQuery, scenario_to_feature_vector
from flaas.recommender.multi_criteria import MultiCriteriaScorer, compute_pareto_front
from flaas.recommender.rationale import RationaleGenerator
from flaas.recommender.simple_baseline_model import SimpleKNNBaseline
from flaas.recommender.engine import RecommenderEngine, RecommendationRequest
from flaas.recommender.evaluation import LeaveOneScenarioOutEvaluator, hand_written_rule
from flaas.grid.scenarios import SCENARIOS


def test_domain_gate_policies():
    """Verify active vs pending fixes gating policy."""
    assert "mobile_apps" in ACTIVE_DOMAINS
    assert "iot_edge" in ACTIVE_DOMAINS
    assert is_domain_active("mobile_apps") is True
    assert is_domain_active("iot_edge") is True

    # Healthcare and Finance must be rejected with DomainPendingFixesError
    assert "healthcare" in PENDING_DOMAINS
    assert "finance" in PENDING_DOMAINS
    assert is_domain_active("healthcare") is False
    assert is_domain_active("finance") is False

    with pytest.raises(DomainPendingFixesError) as exc_hc:
        validate_domain_active("healthcare")
    assert "PENDING FIXES" in str(exc_hc.value)

    with pytest.raises(DomainPendingFixesError) as exc_fin:
        validate_domain_active("finance")
    assert "PENDING FIXES" in str(exc_fin.value)

    with pytest.raises(UnknownDomainError):
        validate_domain_active("unknown_domain_xyz")

    # Status check
    st_mob = get_domain_status("mobile_apps")
    assert st_mob["status"] == "active"
    st_hc = get_domain_status("healthcare")
    assert st_hc["status"] == "pending_fixes"


def test_feasibility_filter_conformance():
    """Verify that FeasibilityFilter conforms to domain's frozen kb-v1 pool."""
    # Mobile Apps: candidate_algorithms=["FedAvg", "FedProx"]
    ff_mobile = FeasibilityFilter("mobile_apps")
    all_mob = ff_mobile.filter(QueryConstraints())
    for cfg in all_mob:
        assert cfg.client_algo in ["FedAvg", "FedProx"]

    # Test Byzantine robustness constraint filter
    ff_robust = ff_mobile.filter(QueryConstraints(require_byzantine_robustness=True))
    assert len(ff_robust) > 0
    for cfg in ff_robust:
        assert cfg.aggregation_rule in ["TrimmedMean", "Krum", "MultiKrum"]

    # Test privacy constraint filter
    ff_priv = ff_mobile.filter(QueryConstraints(require_privacy=True))
    assert len(ff_priv) > 0
    for cfg in ff_priv:
        assert cfg.dp_type == "client"


def test_scenario_matcher_and_leave_one_out():
    """Verify scenario embedding, distance matching, and leave-one-out exclusion."""
    matcher = ScenarioMatcher(SCENARIOS)

    # Exact match for scen_01 (Clean IID)
    s1 = [s for s in SCENARIOS if s.scenario_id == "scen_01"][0]
    q1 = ScenarioQuery(
        heterogeneity_alpha=s1.heterogeneity_alpha,
        participation_rate=s1.participation_rate,
        attack_type=s1.attack_type,
        attack_fraction=s1.attack_fraction,
        privacy_mode=s1.privacy_mode,
        scenario_id="scen_01",
    )
    matches = matcher.find_nearest_scenarios(q1, top_k=1)
    assert matches[0][0].scenario_id == "scen_01"
    assert matches[0][1] == 0.0

    # Leave-one-out exclusion
    loso_matches = matcher.find_nearest_scenarios(q1, top_k=2, exclude_scenario_id="scen_01")
    assert loso_matches[0][0].scenario_id != "scen_01"
    assert loso_matches[0][1] > 0.0


def test_pareto_front_computation():
    """Verify mathematical Pareto optimality computation across (Acc, Wire, Time)."""
    # A dominates B if A >= B in all and strictly > in one
    # Note: Wire bytes and time are minimized, accuracy maximized
    items = [
        {"config_id": "c1", "accuracy": 90.0, "wire_bytes": 1000, "time_seconds": 10.0},  # Pareto optimal
        {"config_id": "c2", "accuracy": 85.0, "wire_bytes": 500, "time_seconds": 5.0},    # Pareto optimal (low wire/time)
        {"config_id": "c3", "accuracy": 80.0, "wire_bytes": 1200, "time_seconds": 12.0},  # Dominated by c1
    ]
    pareto_ids = compute_pareto_front(items)
    assert "c1" in pareto_ids
    assert "c2" in pareto_ids
    assert "c3" not in pareto_ids


def test_recommender_engine_end_to_end():
    """Verify end-to-end recommendation, Pareto front output, and plain-language rationale."""
    engine = RecommenderEngine(parquet_path="results/stage4_grid.parquet")

    # Mobile apps query under attack
    req = RecommendationRequest(
        domain_id="mobile_apps",
        scenario_query=ScenarioQuery(
            heterogeneity_alpha=0.5,
            participation_rate=1.0,
            attack_type="sign_flip",
            attack_fraction=0.2,
            attack_scale=2.0,
            privacy_mode="none",
        ),
        preset="balanced",
    )
    res = engine.recommend(req)

    assert res.domain_id == "mobile_apps"
    assert res.recommended_config is not None
    assert len(res.pareto_frontier) > 0
    assert len(res.ranked_candidates) > 0
    assert res.dataset_evidence == "1-seed (Seed 42)"

    # Verify plain-language rationale contains experiment numbers and 1-seed label
    rationale = res.rationale
    assert "test accuracy" in rationale
    assert "wire" in rationale or "MB" in rationale
    assert "1-seed" in rationale
    assert "seed=42" in rationale

    # Verify pending domain rejection
    req_pending = RecommendationRequest(
        domain_id="healthcare",
        scenario_query=ScenarioQuery(),
    )
    with pytest.raises(DomainPendingFixesError):
        engine.recommend(req_pending)


def test_simple_knn_baseline_regressor():
    """Verify simple baseline model regressor."""
    df = pd.read_parquet("results/stage4_grid.parquet")
    mob_df = df[df["domain"] == "mobile_apps"]
    knn = SimpleKNNBaseline(n_neighbors=2)
    knn.fit(mob_df)

    feat = scenario_to_feature_vector(0.5, 1.0, None, 0.0, 1.0, "none")
    best = knn.predict_best_config(feat, ["cfg_01_baseline", "cfg_02_fedprox"])
    assert best in ["cfg_01_baseline", "cfg_02_fedprox"]


def test_hand_written_rule():
    """Verify expert heuristic rule logic."""
    s_clean = [s for s in SCENARIOS if s.scenario_id == "scen_01"][0]
    assert hand_written_rule(s_clean, "mobile_apps") == "cfg_01_baseline"

    s_att = [s for s in SCENARIOS if s.attack_type is not None][0]
    assert hand_written_rule(s_att, "mobile_apps") == "cfg_04_trimmed_mean"
