"""
Leave-One-Scenario-Out (LOSO) Cross-Validation Evaluator.
Evaluates the FLaaS Recommender against 4 competing baselines + 1 simple model baseline:
1. Default FedAvg (cfg_01_baseline)
2. Random Pick (uniform random among domain candidates)
3. Hand-Written Rule (expert decision tree)
4. Domain-Blind Recommender (pools data across all domains without domain constraints)
5. Simple Model Baseline (K-Nearest Neighbors regressor)
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from flaas.recommender.domain_gate import ACTIVE_DOMAINS, validate_domain_active, DATASET_LABEL
from flaas.recommender.engine import RecommenderEngine, RecommendationRequest
from flaas.recommender.scenario_matching import ScenarioQuery, scenario_to_feature_vector
from flaas.recommender.simple_baseline_model import SimpleKNNBaseline
from flaas.grid.scenarios import SCENARIOS, ScenarioSpec, get_domain_configs
from flaas.profiles.kb_v1 import get_profile


def hand_written_rule(scenario: ScenarioSpec, domain_id: str) -> str:
    """
    Expert heuristic decision tree:
    - If adversarial attack present: prioritize Byzantine-robust aggregation (TrimmedMean or MultiKrum).
    - Else if severe non-IID skew (alpha <= 0.1): prioritize proximal drift regularization (FedProx).
    - Else if mobile / partial participation (q <= 0.2): prioritize ternary 2-bit packing.
    - Else: default standard FedAvg baseline.
    """
    if scenario.attack_type is not None and scenario.attack_fraction > 0.0:
        return "cfg_04_trimmed_mean"
    elif scenario.heterogeneity_alpha is not None and scenario.heterogeneity_alpha <= 0.1:
        return "cfg_02_fedprox"
    elif scenario.participation_rate <= 0.2:
        return "cfg_06_ternary_compression"
    else:
        return "cfg_01_baseline"


def domain_blind_recommend(
    all_df: pd.DataFrame,
    query_scenario: ScenarioSpec,
    exclude_scenario_id: Optional[str] = None,
) -> str:
    """
    Domain-blind baseline: Aggregates across ALL domains indiscriminately,
    ignoring domain constraints, modalities, and specific performance profiles.
    """
    train_df = all_df[all_df["scenario_id"] != exclude_scenario_id] if exclude_scenario_id else all_df
    # Average accuracy across all domains per config
    perf = train_df.groupby("config_id")["final_accuracy"].mean()
    return str(perf.idxmax())


class LeaveOneScenarioOutEvaluator:
    """Executes systematic Leave-One-Scenario-Out cross-validation across active domains."""

    def __init__(self, parquet_path: str = "results/stage4_grid.parquet"):
        self.engine = RecommenderEngine(parquet_path=parquet_path)
        self.df = pd.read_parquet(parquet_path)

    def evaluate_domain(
        self,
        domain_id: str,
        metric: str = "final_accuracy",
    ) -> Dict[str, Any]:
        """Runs LOSO across all 16 scenarios for a single domain."""
        validate_domain_active(domain_id)

        domain_df = self.df[self.df["domain"] == domain_id]
        profile = get_profile(domain_id)
        candidate_configs = [c.config_id for c in get_domain_configs(domain_id, profile.typical_num_clients)]

        # Baselines tracked
        methods = [
            "Recommender_Proposed",
            "Baseline_Default_FedAvg",
            "Baseline_Random_Pick",
            "Baseline_Handwritten_Rule",
            "Baseline_Domain_Blind",
            "Baseline_Simple_KNN",
        ]

        accuracies = {m: [] for m in methods}
        regrets = {m: [] for m in methods}
        ranks = {m: [] for m in methods}
        scenario_records = []

        # Train simple model baseline
        simple_model = SimpleKNNBaseline(n_neighbors=3)

        for held_out_scenario in SCENARIOS:
            sid = held_out_scenario.scenario_id
            held_out_df = domain_df[domain_df["scenario_id"] == sid]
            if held_out_df.empty:
                continue

            # Ground truth map: config_id -> accuracy
            perf_map = dict(zip(held_out_df["config_id"], held_out_df[metric]))
            sorted_configs = sorted(perf_map.keys(), key=lambda c: perf_map[c], reverse=True)
            optimal_acc = perf_map[sorted_configs[0]]

            # Fit simple model on remaining scenarios
            train_df = domain_df[domain_df["scenario_id"] != sid]
            simple_model.fit(train_df)

            # 1. Proposed Recommender (holding out sid)
            q = ScenarioQuery(
                heterogeneity_alpha=held_out_scenario.heterogeneity_alpha,
                participation_rate=held_out_scenario.participation_rate,
                attack_type=held_out_scenario.attack_type,
                attack_fraction=held_out_scenario.attack_fraction,
                attack_scale=held_out_scenario.attack_scale,
                privacy_mode=held_out_scenario.privacy_mode,
                scenario_id=sid,
            )
            req = RecommendationRequest(
                domain_id=domain_id,
                scenario_query=q,
                preset="accuracy_first",  # evaluate accuracy objective
            )
            rec_res = self.engine.recommend(req, exclude_scenario_id=sid)
            chosen_rec = rec_res.recommended_config.config_id

            # 2. Default FedAvg
            chosen_fedavg = "cfg_01_baseline" if "cfg_01_baseline" in perf_map else sorted_configs[0]

            # 3. Random Pick (expected value over candidate pool)
            rand_accs = [perf_map[c] for c in candidate_configs if c in perf_map]
            expected_rand_acc = float(np.mean(rand_accs))
            expected_rand_regret = optimal_acc - expected_rand_acc
            expected_rand_rank = float((len(sorted_configs) + 1) / 2.0)

            # 4. Hand-written Rule
            chosen_rule = hand_written_rule(held_out_scenario, domain_id)
            if chosen_rule not in perf_map:
                chosen_rule = "cfg_01_baseline"

            # 5. Domain-Blind Recommender
            chosen_blind = domain_blind_recommend(self.df, held_out_scenario, exclude_scenario_id=sid)
            if chosen_blind not in perf_map:
                chosen_blind = sorted_configs[0]

            # 6. Simple Model Baseline
            q_feat = scenario_to_feature_vector(
                held_out_scenario.heterogeneity_alpha,
                held_out_scenario.participation_rate,
                held_out_scenario.attack_type,
                held_out_scenario.attack_fraction,
                held_out_scenario.attack_scale,
                held_out_scenario.privacy_mode,
            )
            chosen_knn = simple_model.predict_best_config(q_feat, candidate_configs)

            # Helper to compute metric, regret, rank
            def record_pick(m_name: str, c_id: str):
                acc = perf_map.get(c_id, 0.0)
                regret = optimal_acc - acc
                rank = sorted_configs.index(c_id) + 1 if c_id in sorted_configs else len(sorted_configs)
                accuracies[m_name].append(acc)
                regrets[m_name].append(regret)
                ranks[m_name].append(rank)

            record_pick("Recommender_Proposed", chosen_rec)
            record_pick("Baseline_Default_FedAvg", chosen_fedavg)
            record_pick("Baseline_Handwritten_Rule", chosen_rule)
            record_pick("Baseline_Domain_Blind", chosen_blind)
            record_pick("Baseline_Simple_KNN", chosen_knn)

            accuracies["Baseline_Random_Pick"].append(expected_rand_acc)
            regrets["Baseline_Random_Pick"].append(expected_rand_regret)
            ranks["Baseline_Random_Pick"].append(expected_rand_rank)

            scenario_records.append({
                "scenario_id": sid,
                "scenario_name": held_out_scenario.name,
                "optimal_config": sorted_configs[0],
                "optimal_acc": optimal_acc,
                "rec_chosen": chosen_rec,
                "rec_acc": perf_map.get(chosen_rec, 0.0),
                "fedavg_acc": perf_map.get(chosen_fedavg, 0.0),
                "rule_chosen": chosen_rule,
                "rule_acc": perf_map.get(chosen_rule, 0.0),
            })

        # Summary statistics per method
        summary = {}
        for m in methods:
            summary[m] = {
                "mean_accuracy": float(np.mean(accuracies[m])),
                "mean_regret": float(np.mean(regrets[m])),
                "mean_rank": float(np.mean(ranks[m])),
                "std_accuracy": float(np.std(accuracies[m])),
            }

        return {
            "domain_id": domain_id,
            "domain_name": profile.name,
            "num_scenarios_evaluated": len(scenario_records),
            "summary_by_method": summary,
            "per_scenario_records": scenario_records,
            "dataset_label": DATASET_LABEL,
        }

    def evaluate_all_active_domains(self) -> Dict[str, Any]:
        """Runs LOSO across all active domains (mobile_apps, iot_edge)."""
        results = {}
        for d in ACTIVE_DOMAINS:
            results[d] = self.evaluate_domain(d)
        return results
