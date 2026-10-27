"""
FLaaS Recommender Module.
Provides domain gating, feasibility filtering, scenario matching,
multi-criteria scoring, plain-language rationales, and evaluation.
"""

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
from flaas.recommender.scenario_matching import ScenarioMatcher, ScenarioQuery
from flaas.recommender.multi_criteria import MultiCriteriaScorer, ScoredConfig, compute_pareto_front
from flaas.recommender.rationale import RationaleGenerator
from flaas.recommender.simple_baseline_model import SimpleKNNBaseline
from flaas.recommender.engine import (
    RecommenderEngine,
    RecommendationRequest,
    RecommendationResult,
)
from flaas.recommender.evaluation import LeaveOneScenarioOutEvaluator

__all__ = [
    "ACTIVE_DOMAINS",
    "PENDING_DOMAINS",
    "DATASET_LABEL",
    "DomainPendingFixesError",
    "UnknownDomainError",
    "is_domain_active",
    "validate_domain_active",
    "get_domain_status",
    "FeasibilityFilter",
    "QueryConstraints",
    "ScenarioMatcher",
    "ScenarioQuery",
    "MultiCriteriaScorer",
    "ScoredConfig",
    "compute_pareto_front",
    "RationaleGenerator",
    "SimpleKNNBaseline",
    "RecommenderEngine",
    "RecommendationRequest",
    "RecommendationResult",
    "LeaveOneScenarioOutEvaluator",
]
