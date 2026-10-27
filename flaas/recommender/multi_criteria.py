"""
Multi-Criteria Scoring and Pareto Front Computation for FLaaS Recommender.
Normalizes empirical metrics from nearest scenarios and ranks candidate configurations
across multi-objective trade-offs (Accuracy, Communication, Latency).
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class ScoredConfig:
    config_id: str
    name: str
    client_algo: str
    aggregation_rule: str
    compressor: str
    dp_type: str
    raw_accuracy: float
    raw_wire_bytes: int
    raw_time_seconds: float
    rounds_completed: int
    early_stopped: bool
    epsilon: float
    norm_accuracy: float
    norm_communication: float
    norm_latency: float
    composite_score: float
    is_pareto_optimal: bool
    matched_scenario_id: str


def compute_pareto_front(candidates_metrics: List[Dict[str, Any]]) -> List[str]:
    """
    Identifies Pareto-optimal configuration IDs across:
    - Objective 1: Accuracy (Maximize)
    - Objective 2: Wire Bytes (Minimize)
    - Objective 3: Time Seconds (Minimize)

    Candidate A dominates Candidate B if A is >= B in all objectives
    and strictly > in at least one.
    """
    pareto_ids = []
    n = len(candidates_metrics)

    for i in range(n):
        a = candidates_metrics[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            b = candidates_metrics[j]
            # b dominates a if b is at least as good in all, and better in at least one
            b_as_good = (
                b["accuracy"] >= a["accuracy"]
                and b["wire_bytes"] <= a["wire_bytes"]
                and b["time_seconds"] <= a["time_seconds"]
            )
            b_strictly_better = (
                b["accuracy"] > a["accuracy"]
                or b["wire_bytes"] < a["wire_bytes"]
                or b["time_seconds"] < a["time_seconds"]
            )
            if b_as_good and b_strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_ids.append(a["config_id"])

    return pareto_ids


class MultiCriteriaScorer:
    """
    Evaluates and scores configurations using empirical grid data from matched scenarios.
    """

    PRESETS = {
        "balanced": {"w_acc": 0.50, "w_comm": 0.30, "w_lat": 0.20},
        "accuracy_first": {"w_acc": 0.70, "w_comm": 0.20, "w_lat": 0.10},
        "communication_first": {"w_acc": 0.30, "w_comm": 0.55, "w_lat": 0.15},
        "low_latency": {"w_acc": 0.35, "w_comm": 0.15, "w_lat": 0.50},
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.PRESETS["balanced"]

    def score_candidates(
        self,
        domain_df: pd.DataFrame,
        candidate_config_ids: List[str],
        matched_scenario_ids: List[str],
        scenario_weights: Optional[List[float]] = None,
    ) -> List[ScoredConfig]:
        """
        Ranks candidate configurations using normalized empirical results.
        scenario_weights can be used if blending multiple nearest scenarios.
        """
        # Filter to relevant scenario and candidate configs
        subset = domain_df[
            (domain_df["scenario_id"].isin(matched_scenario_ids))
            & (domain_df["config_id"].isin(candidate_config_ids))
        ].copy()

        if subset.empty:
            return []

        # Average across matched scenarios (or primary scenario)
        agg_metrics = subset.groupby("config_id").agg({
            "final_accuracy": "mean",
            "total_wire_bytes": "mean",
            "total_time_seconds": "mean",
            "rounds_completed": "mean",
            "early_stopped": "max",
            "epsilon": "mean",
        }).reset_index()

        # Extract min-max for normalization
        accs = agg_metrics["final_accuracy"].values
        wires = agg_metrics["total_wire_bytes"].values
        times = agg_metrics["total_time_seconds"].values

        min_acc, max_acc = accs.min(), accs.max()
        min_wire, max_wire = wires.min(), wires.max()
        min_time, max_time = times.min(), times.max()

        acc_denom = max(1e-4, max_acc - min_acc)
        wire_denom = max(1e-4, max_wire - min_wire)
        time_denom = max(1e-4, max_time - min_time)

        # Build raw dict for Pareto computation
        pareto_inputs = []
        for _, row in agg_metrics.iterrows():
            pareto_inputs.append({
                "config_id": row["config_id"],
                "accuracy": row["final_accuracy"],
                "wire_bytes": row["total_wire_bytes"],
                "time_seconds": row["total_time_seconds"],
            })
        pareto_optimal_ids = set(compute_pareto_front(pareto_inputs))

        # Weighting factors
        w_acc = self.weights.get("w_acc", 0.5)
        w_comm = self.weights.get("w_comm", 0.3)
        w_lat = self.weights.get("w_lat", 0.2)

        scored_list: List[ScoredConfig] = []
        for _, row in agg_metrics.iterrows():
            cid = row["config_id"]
            acc = float(row["final_accuracy"])
            wire = int(row["total_wire_bytes"])
            t = float(row["total_time_seconds"])

            # Normalization: Accuracy higher is better; Wire & Time lower is better
            norm_a = (acc - min_acc) / acc_denom
            norm_c = 1.0 - ((wire - min_wire) / wire_denom)
            norm_l = 1.0 - ((t - min_time) / time_denom)

            comp_score = w_acc * norm_a + w_comm * norm_c + w_lat * norm_l

            scored_list.append(ScoredConfig(
                config_id=cid,
                name=cid,
                client_algo="FedAvg",  # Will be populated with ConfigSpec details by Engine
                aggregation_rule="FedAvgMean",
                compressor="identity",
                dp_type="none",
                raw_accuracy=acc,
                raw_wire_bytes=wire,
                raw_time_seconds=t,
                rounds_completed=int(round(row["rounds_completed"])),
                early_stopped=bool(row["early_stopped"]),
                epsilon=float(row["epsilon"]),
                norm_accuracy=float(norm_a),
                norm_communication=float(norm_c),
                norm_latency=float(norm_l),
                composite_score=float(comp_score),
                is_pareto_optimal=(cid in pareto_optimal_ids),
                matched_scenario_id=matched_scenario_ids[0],
            ))

        # Sort descending by composite score
        scored_list.sort(key=lambda x: x.composite_score, reverse=True)
        return scored_list
