"""
Scenario Matching for FLaaS Recommender.
Embeds operational conditions into normalized feature space and retrieves
the nearest canonical grid scenarios using standardized distance metrics.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from flaas.grid.scenarios import SCENARIOS, ScenarioSpec


@dataclass
class ScenarioQuery:
    """Operational conditions describing the client environment."""
    heterogeneity_alpha: Optional[float] = None  # None for IID, or 0.5, 0.3, 0.1
    participation_rate: float = 1.0  # 0.0 - 1.0
    attack_type: Optional[str] = None  # None, 'sign_flip', 'gaussian_noise', 'scaled_update'
    attack_fraction: float = 0.0  # 0.0 - 0.5
    attack_scale: float = 1.0
    privacy_mode: str = "none"  # 'none', 'client_dp'
    scenario_id: Optional[str] = None  # If direct canonical scenario specified


def scenario_to_feature_vector(
    heterogeneity_alpha: Optional[float],
    participation_rate: float,
    attack_type: Optional[str],
    attack_fraction: float,
    attack_scale: float,
    privacy_mode: str,
) -> np.ndarray:
    """
    Encodes scenario characteristics into a normalized 5D representation:
    [skew_intensity, participation_rate, attack_intensity, attack_scale_norm, privacy_flag]
    """
    # 1. Non-IID Skew: IID = 0.0, alpha=0.5 -> 2.0, alpha=0.1 -> 10.0 (normalized by 10)
    if heterogeneity_alpha is None:
        skew_feat = 0.0
    else:
        skew_feat = min(1.0, (1.0 / max(0.01, heterogeneity_alpha)) / 10.0)

    # 2. Participation rate [0, 1]
    part_feat = float(np.clip(participation_rate, 0.0, 1.0))

    # 3. Attack intensity: fraction [0, 1]
    has_attack = 1.0 if (attack_type is not None and attack_fraction > 0.0) else 0.0
    att_frac_feat = float(np.clip(attack_fraction, 0.0, 1.0)) * has_attack

    # 4. Attack scale normalized [0, 1] (scale up to 5.0)
    att_scale_feat = float(np.clip(attack_scale / 5.0, 0.0, 1.0)) * has_attack

    # 5. Differential privacy flag
    dp_feat = 1.0 if privacy_mode == "client_dp" else 0.0

    return np.array([skew_feat, part_feat, att_frac_feat, att_scale_feat, dp_feat], dtype=np.float32)


class ScenarioMatcher:
    """Matches user operational environment to nearest canonical grid scenarios."""

    def __init__(self, canonical_scenarios: Optional[List[ScenarioSpec]] = None):
        self.scenarios = canonical_scenarios or SCENARIOS
        self.feature_matrix = np.stack([
            scenario_to_feature_vector(
                s.heterogeneity_alpha,
                s.participation_rate,
                s.attack_type,
                s.attack_fraction,
                s.attack_scale,
                s.privacy_mode,
            )
            for s in self.scenarios
        ])
        # Feature weights for distance metric:
        # [skew, participation, attack_frac, attack_scale, privacy]
        self.weights = np.array([1.5, 1.0, 2.0, 1.0, 2.5], dtype=np.float32)

    def find_nearest_scenarios(
        self,
        query: ScenarioQuery,
        top_k: int = 1,
        exclude_scenario_id: Optional[str] = None,
    ) -> List[Tuple[ScenarioSpec, float]]:
        """
        Finds the top_k nearest scenarios.
        exclude_scenario_id is used for Leave-One-Scenario-Out (LOSO) evaluation.
        Returns list of (ScenarioSpec, distance).
        """
        # If explicit canonical ID specified and not excluded
        if query.scenario_id is not None and query.scenario_id != exclude_scenario_id:
            for s in self.scenarios:
                if s.scenario_id == query.scenario_id:
                    return [(s, 0.0)]

        q_vec = scenario_to_feature_vector(
            query.heterogeneity_alpha,
            query.participation_rate,
            query.attack_type,
            query.attack_fraction,
            query.attack_scale,
            query.privacy_mode,
        )

        diffs = (self.feature_matrix - q_vec) * self.weights
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        # Filter excluded scenario
        ranked_indices = np.argsort(distances)
        results: List[Tuple[ScenarioSpec, float]] = []

        for idx in ranked_indices:
            scen = self.scenarios[idx]
            if exclude_scenario_id is not None and scen.scenario_id == exclude_scenario_id:
                continue
            results.append((scen, float(distances[idx])))
            if len(results) >= top_k:
                break

        return results
