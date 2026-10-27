"""
Stage 5 Re-Analysis on Existing Stage 4 Grid Data (No Re-run):
1. Compute success rate (reached target accuracy), rounds-to-target, and bytes-to-target.
2. Per-scenario Wins / Ties / Losses against each baseline.
3. Paired bootstrap 95% confidence intervals of the mean difference (Recommender - Baseline).
4. Rigorous tie-aware fractional ranking.
"""

import json
import numpy as np
import pandas as pd
from flaas.recommender.evaluation import LeaveOneScenarioOutEvaluator
from flaas.grid.scenarios import SCENARIOS
from flaas.profiles.kb_v1 import get_profile


def paired_bootstrap_ci(diffs: np.ndarray, n_bootstraps: int = 10000, ci: float = 95.0, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_means = [rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_bootstraps)]
    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha))
    upper = float(np.percentile(boot_means, 100.0 - alpha))
    return lower, upper


def reanalyze():
    evaluator = LeaveOneScenarioOutEvaluator(parquet_path="results/stage4_grid.parquet")
    df = pd.read_parquet("results/stage4_grid.parquet")
    df = df[df["config_id"].str.startswith("cfg_")]

    results = {}
    domains = ["mobile_apps", "iot_edge"]

    for domain_id in domains:
        profile = get_profile(domain_id)
        target_acc = profile.target_accuracy
        dom_df = df[df["domain"] == domain_id].copy()

        # Run LOSO to obtain picks per scenario
        loso_out = evaluator.evaluate_domain(domain_id)
        per_scen = loso_out["per_scenario_records"]

        methods = [
            "Recommender_Proposed",
            "Baseline_Default_FedAvg",
            "Baseline_Random_Pick",
            "Baseline_Handwritten_Rule",
            "Baseline_Domain_Blind",
            "Baseline_Simple_KNN",
        ]

        # Extract per-method metrics per scenario
        method_metrics = {m: {"acc": [], "early_stopped": [], "rounds": [], "bytes": []} for m in methods}

        # We will retrieve row from dom_df for the chosen config in that scenario
        for r in per_scen:
            sid = r["scenario_id"]
            scen_df = dom_df[dom_df["scenario_id"] == sid]

            # Recommender pick
            rec_cid = r["rec_chosen"]
            rec_row = scen_df[scen_df["config_id"] == rec_cid].iloc[0]
            method_metrics["Recommender_Proposed"]["acc"].append(rec_row["final_accuracy"])
            method_metrics["Recommender_Proposed"]["early_stopped"].append(1 if rec_row["early_stopped"] else 0)
            method_metrics["Recommender_Proposed"]["rounds"].append(rec_row["rounds_completed"])
            method_metrics["Recommender_Proposed"]["bytes"].append(rec_row["total_wire_bytes"])

            # Default FedAvg
            base_row = scen_df[scen_df["config_id"] == "cfg_01_baseline"].iloc[0]
            method_metrics["Baseline_Default_FedAvg"]["acc"].append(base_row["final_accuracy"])
            method_metrics["Baseline_Default_FedAvg"]["early_stopped"].append(1 if base_row["early_stopped"] else 0)
            method_metrics["Baseline_Default_FedAvg"]["rounds"].append(base_row["rounds_completed"])
            method_metrics["Baseline_Default_FedAvg"]["bytes"].append(base_row["total_wire_bytes"])

            # Random Pick (mean over all 8 configs)
            method_metrics["Baseline_Random_Pick"]["acc"].append(scen_df["final_accuracy"].mean())
            method_metrics["Baseline_Random_Pick"]["early_stopped"].append(scen_df["early_stopped"].mean())
            method_metrics["Baseline_Random_Pick"]["rounds"].append(scen_df["rounds_completed"].mean())
            method_metrics["Baseline_Random_Pick"]["bytes"].append(scen_df["total_wire_bytes"].mean())

            # Hand-written rule
            rule_cid = r["rule_chosen"]
            rule_row = scen_df[scen_df["config_id"] == rule_cid].iloc[0]
            method_metrics["Baseline_Handwritten_Rule"]["acc"].append(rule_row["final_accuracy"])
            method_metrics["Baseline_Handwritten_Rule"]["early_stopped"].append(1 if rule_row["early_stopped"] else 0)
            method_metrics["Baseline_Handwritten_Rule"]["rounds"].append(rule_row["rounds_completed"])
            method_metrics["Baseline_Handwritten_Rule"]["bytes"].append(rule_row["total_wire_bytes"])

            # Domain blind
            # We find what domain blind picked from loso_out
            blind_cid = None
            from flaas.recommender.evaluation import domain_blind_recommend
            s_obj = [s for s in SCENARIOS if s.scenario_id == sid][0]
            blind_cid = domain_blind_recommend(df, s_obj, exclude_scenario_id=sid)
            blind_row = scen_df[scen_df["config_id"] == blind_cid].iloc[0]
            method_metrics["Baseline_Domain_Blind"]["acc"].append(blind_row["final_accuracy"])
            method_metrics["Baseline_Domain_Blind"]["early_stopped"].append(1 if blind_row["early_stopped"] else 0)
            method_metrics["Baseline_Domain_Blind"]["rounds"].append(blind_row["rounds_completed"])
            method_metrics["Baseline_Domain_Blind"]["bytes"].append(blind_row["total_wire_bytes"])

            # Simple KNN
            from flaas.recommender.simple_baseline_model import SimpleKNNBaseline
            from flaas.recommender.scenario_matching import scenario_to_feature_vector
            knn = SimpleKNNBaseline(n_neighbors=3)
            knn.fit(dom_df[dom_df["scenario_id"] != sid])
            q_feat = scenario_to_feature_vector(
                s_obj.heterogeneity_alpha, s_obj.participation_rate,
                s_obj.attack_type, s_obj.attack_fraction, s_obj.attack_scale, s_obj.privacy_mode
            )
            candidate_cids = scen_df["config_id"].unique().tolist()
            knn_cid = knn.predict_best_config(q_feat, candidate_cids)
            knn_row = scen_df[scen_df["config_id"] == knn_cid].iloc[0]
            method_metrics["Baseline_Simple_KNN"]["acc"].append(knn_row["final_accuracy"])
            method_metrics["Baseline_Simple_KNN"]["early_stopped"].append(1 if knn_row["early_stopped"] else 0)
            method_metrics["Baseline_Simple_KNN"]["rounds"].append(knn_row["rounds_completed"])
            method_metrics["Baseline_Simple_KNN"]["bytes"].append(knn_row["total_wire_bytes"])

        # Compile Summary Table (Accuracy, Success Rate, Rounds to target, MB to target)
        domain_summary = {}
        for m in methods:
            accs = np.array(method_metrics[m]["acc"])
            success = np.array(method_metrics[m]["early_stopped"])
            rounds = np.array(method_metrics[m]["rounds"])
            mb = np.array(method_metrics[m]["bytes"]) / (1024 * 1024)

            domain_summary[m] = {
                "mean_acc": float(np.mean(accs)),
                "success_rate_pct": float(np.mean(success) * 100.0),
                "mean_rounds_to_target": float(np.mean(rounds)),
                "mean_mb_to_target": float(np.mean(mb)),
            }

        # Pairwise Wins / Ties / Losses and Bootstrap 95% CI vs Recommender
        pairwise_comparisons = {}
        rec_accs = np.array(method_metrics["Recommender_Proposed"]["acc"])

        for b in [m for m in methods if m != "Recommender_Proposed"]:
            b_accs = np.array(method_metrics[b]["acc"])
            diffs = rec_accs - b_accs

            wins = int(np.sum(diffs > 0.05))
            ties = int(np.sum(np.abs(diffs) <= 0.05))
            losses = int(np.sum(diffs < -0.05))

            mean_diff = float(np.mean(diffs))
            ci_low, ci_high = paired_bootstrap_ci(diffs, n_bootstraps=10000, ci=95.0, seed=42)

            pairwise_comparisons[b] = {
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "mean_diff": mean_diff,
                "bootstrap_95_ci": [ci_low, ci_high],
            }

        results[domain_id] = {
            "domain_name": profile.name,
            "target_accuracy": target_acc,
            "metrics_summary": domain_summary,
            "pairwise_vs_recommender": pairwise_comparisons,
        }

    return results


if __name__ == "__main__":
    res = reanalyze()
    print(json.dumps(res, indent=2))
