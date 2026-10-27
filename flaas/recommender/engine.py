"""
FLaaS Recommendation Engine.
Domain-agnostic orchestrator providing multi-criteria configuration recommendation,
Pareto frontier identification, and plain-language empirical justifications.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import os
import pandas as pd
import numpy as np

from flaas.recommender.domain_gate import (
    validate_domain_active,
    is_domain_active,
    DATASET_LABEL,
)
from flaas.recommender.feasibility import FeasibilityFilter, QueryConstraints
from flaas.recommender.scenario_matching import ScenarioMatcher, ScenarioQuery
from flaas.recommender.multi_criteria import MultiCriteriaScorer, ScoredConfig
from flaas.recommender.rationale import RationaleGenerator
from flaas.profiles.kb_v1 import get_profile, DomainProfile
from flaas.grid.scenarios import get_domain_configs, SCENARIOS, ScenarioSpec, ConfigSpec


@dataclass
class RecommendationRequest:
    """User request for FL configuration recommendation."""
    domain_id: str
    scenario_query: ScenarioQuery
    constraints: Optional[QueryConstraints] = None
    preset: str = "balanced"  # 'balanced', 'accuracy_first', 'communication_first', 'low_latency'
    weights: Optional[Dict[str, float]] = None


@dataclass
class RecommendationResult:
    """Structured response from the Recommender Engine."""
    domain_id: str
    domain_name: str
    matched_scenario: ScenarioSpec
    scenario_distance: float
    recommended_config: ScoredConfig
    recommended_spec: ConfigSpec
    pareto_frontier: List[ScoredConfig]
    ranked_candidates: List[ScoredConfig]
    rationale: str
    dataset_evidence: str = DATASET_LABEL


class RecommenderEngine:
    """
    Domain-agnostic federated learning configuration recommender.
    Unites domain gating, feasibility filtering, scenario matching,
    multi-criteria scoring, and empirical rationale generation.
    """

    def __init__(
        self,
        parquet_path: str = "results/stage4_grid.parquet",
    ):
        self.parquet_path = parquet_path
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet grid dataset not found at {parquet_path}")
        self.df = pd.read_parquet(parquet_path)
        self.scenario_matcher = ScenarioMatcher(SCENARIOS)

    def recommend(
        self,
        request: RecommendationRequest,
        exclude_scenario_id: Optional[str] = None,
    ) -> RecommendationResult:
        """
        Generates configuration recommendation for the given request.
        Raises DomainPendingFixesError if domain is marked pending fixes.
        exclude_scenario_id is used for Leave-One-Scenario-Out (LOSO) evaluation.
        """
        # 1. Domain Gate Validation
        validate_domain_active(request.domain_id)

        domain_id = request.domain_id
        profile: DomainProfile = get_profile(domain_id)
        domain_df = self.df[self.df["domain"] == domain_id]

        if domain_df.empty:
            raise ValueError(f"No grid experiment data found for domain '{domain_id}' in {self.parquet_path}")

        # 2. Feasibility Filtering against domain candidate pool & constraints
        constraints = request.constraints
        if constraints is None:
            has_attack = (request.scenario_query.attack_type is not None and request.scenario_query.attack_fraction > 0.0)
            has_privacy = (request.scenario_query.privacy_mode == "client_dp")
            constraints = QueryConstraints(
                require_byzantine_robustness=has_attack,
                require_privacy=has_privacy,
            )

        feas_filter = FeasibilityFilter(domain_id)
        feasible_specs: List[ConfigSpec] = feas_filter.filter(constraints)
        feasible_ids = [c.config_id for c in feasible_specs]
        spec_map = {c.config_id: c for c in feasible_specs}

        # 3. Scenario Matching
        nearest_matches = self.scenario_matcher.find_nearest_scenarios(
            query=request.scenario_query,
            top_k=1,
            exclude_scenario_id=exclude_scenario_id,
        )
        if not nearest_matches:
            raise RuntimeError(f"No matching scenarios found for query {request.scenario_query}")

        matched_scenario, distance = nearest_matches[0]

        # 4. Multi-Criteria Scoring & Pareto Front
        scorer_weights = request.weights or MultiCriteriaScorer.PRESETS.get(
            request.preset, MultiCriteriaScorer.PRESETS["balanced"]
        )
        scorer = MultiCriteriaScorer(weights=scorer_weights)
        scored_configs = scorer.score_candidates(
            domain_df=domain_df,
            candidate_config_ids=feasible_ids,
            matched_scenario_ids=[matched_scenario.scenario_id],
        )

        if not scored_configs:
            raise RuntimeError(
                f"No empirical results found for domain '{domain_id}' under scenario '{matched_scenario.scenario_id}'"
            )

        # Enrich ScoredConfig with ConfigSpec names/algos
        for sc in scored_configs:
            if sc.config_id in spec_map:
                sp = spec_map[sc.config_id]
                sc.name = sp.name
                sc.client_algo = sp.client_algo
                sc.aggregation_rule = sp.aggregation_rule
                sc.compressor = sp.compressor
                sc.dp_type = sp.dp_type

        recommended = scored_configs[0]
        recommended_spec = spec_map[recommended.config_id]
        pareto_front = [c for c in scored_configs if c.is_pareto_optimal]

        # Baseline FedAvg for comparison (if present)
        baseline_match = [c for c in scored_configs if "cfg_01_baseline" in c.config_id]
        baseline_config = baseline_match[0] if baseline_match else None

        # 5. Plain-Language Rationale Generation
        rationale = RationaleGenerator.generate_rationale(
            recommended=recommended,
            baseline=baseline_config,
            scenario=matched_scenario,
            config_spec=recommended_spec,
            domain_name=profile.name,
        )

        return RecommendationResult(
            domain_id=domain_id,
            domain_name=profile.name,
            matched_scenario=matched_scenario,
            scenario_distance=distance,
            recommended_config=recommended,
            recommended_spec=recommended_spec,
            pareto_frontier=pareto_front,
            ranked_candidates=scored_configs,
            rationale=rationale,
            dataset_evidence=DATASET_LABEL,
        )
