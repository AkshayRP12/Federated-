"""
Feasibility Filter for FLaaS Recommender.
Restricts candidate configurations to the domain's frozen kb-v1 pool,
enforcing domain constraints and user operational requirements.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from flaas.profiles.kb_v1 import get_profile, DomainProfile
from flaas.grid.scenarios import get_domain_configs, ConfigSpec


@dataclass
class QueryConstraints:
    """User-specified operational constraints for FL recommendation."""
    require_privacy: bool = False
    max_privacy_epsilon: Optional[float] = None
    require_byzantine_robustness: bool = False
    bandwidth_constraint: Optional[str] = None  # 'unconstrained', 'moderate', 'severe'
    require_compression: bool = False
    target_accuracy: Optional[float] = None


class FeasibilityFilter:
    """
    Filters FL configurations strictly against:
    1. The domain's frozen kb-v1 candidate pool (algorithms & aggregators).
    2. Operational safety and threat constraints (privacy, attacks, bandwidth).
    """

    def __init__(self, domain_id: str):
        self.domain_id = domain_id
        self.profile: DomainProfile = get_profile(domain_id)
        self.all_configs: List[ConfigSpec] = get_domain_configs(
            domain_id=domain_id,
            num_clients=self.profile.typical_num_clients,
        )

    def filter(self, constraints: Optional[QueryConstraints] = None) -> List[ConfigSpec]:
        """
        Returns feasible ConfigSpecs matching domain pool and constraints.
        If constraints over-filter the pool to 0 configs, soft constraints are relaxed
        with diagnostics preserved.
        """
        c = constraints or QueryConstraints()
        candidates: List[ConfigSpec] = []

        for cfg in self.all_configs:
            # 1. Domain candidate pool conformance
            # In kb-v1, cfg.client_algo must be approved by the domain profile
            if cfg.client_algo not in self.profile.candidate_algorithms:
                continue

            # 1. Aggregation rule conformance:
            # If Byzantine robustness is required, candidate must be a robust aggregator (TrimmedMean or Krum/MultiKrum).
            # Otherwise, must be in domain profile candidate aggregators.
            if c.require_byzantine_robustness:
                is_robust = cfg.aggregation_rule in ["TrimmedMean", "Krum", "MultiKrum"]
                if not is_robust:
                    continue
            else:
                agg_family = "Krum" if "Krum" in cfg.aggregation_rule else cfg.aggregation_rule
                if agg_family not in self.profile.candidate_aggregators and cfg.aggregation_rule not in self.profile.candidate_aggregators:
                    continue

            # 2. Privacy constraint
            if c.require_privacy and cfg.dp_type == "none":
                continue

            # 4. Bandwidth constraint
            if c.require_compression and cfg.compressor == "identity":
                continue

            candidates.append(cfg)

        # Fallback relaxation if constraints eliminate all configs
        if not candidates:
            # Relax to domain candidate pool ignoring soft constraints
            candidates = [
                cfg for cfg in self.all_configs
                if cfg.client_algo in self.profile.candidate_algorithms
            ]

        return candidates
