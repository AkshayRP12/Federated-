"""
Comprehensive evaluation script comparing:
1. grid_v1 (canonical 16 scenarios, 8 kb-v1 configs)
2. grid_v2 (expanded domain-specific scenarios, 16 expanded configs)
"""

import json
import numpy as np
import pandas as pd
from flaas.recommender.evaluation import LeaveOneScenarioOutEvaluator
from flaas.grid.scenarios import SCENARIOS
from flaas.grid.expanded_scenarios import IOT_SCENARIOS, MOBILE_SCENARIOS, get_expanded_configs
from flaas.profiles.kb_v1 import get_profile


def paired_bootstrap_ci(diffs: np.ndarray, n_bootstraps: int = 10000, ci: float = 95.0, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_means = [rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_bootstraps)]
    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha))
    upper = float(np.percentile(boot_means, 100.0 - alpha))
    return lower, upper


def evaluate_grid_v1():
    """Evaluates canonical grid_v1 (16 scenarios, 8 configs)."""
    df = pd.read_parquet("results/stage4_grid.parquet")
    df = df[df["config_id"].str.startswith("cfg_")]

    evaluator = LeaveOneScenarioOutEvaluator(parquet_path="results/stage4_grid.parquet")
    results = {}

    for domain_id in ["mobile_apps", "iot_edge"]:
        profile = get_profile(domain_id)
        target = 85.0 if domain_id == "mobile_apps" else 70.0  # Reachable target for IoT
        dom_df = df[df["domain"] == domain_id].copy()

        loso_out = evaluator.evaluate_domain(domain_id)
        per_scen = loso_out["per_scenario_records"]

        methods = ["Recommender_Proposed", "Baseline_Default_FedAvg", "Baseline_Random_Pick", "Baseline_Handwritten_Rule"]
        metrics = {m: {"acc": [], "success": [], "rounds": [], "bytes": []} for m in methods}

        for r in per_scen:
            sid = r["scenario_id"]
            scen_df = dom_df[dom_df["scenario_id"] == sid]

            # Recommender
            rec_row = scen_df[scen_df["config_id"] == r["rec_chosen"]].iloc[0]
            metrics["Recommender_Proposed"]["acc"].append(rec_row["final_accuracy"])
            metrics["Recommender_Proposed"]["success"].append(1 if rec_row["final_accuracy"] >= target else 0)
            metrics["Recommender_Proposed"]["rounds"].append(rec_row["rounds_completed"])
            metrics["Recommender_Proposed"]["bytes"].append(rec_row["total_wire_bytes"])

            # Default FedAvg
            base_row = scen_df[scen_df["config_id"] == "cfg_01_baseline"].iloc[0]
            metrics["Baseline_Default_FedAvg"]["acc"].append(base_row["final_accuracy"])
            metrics["Baseline_Default_FedAvg"]["success"].append(1 if base_row["final_accuracy"] >= target else 0)
            metrics["Baseline_Default_FedAvg"]["rounds"].append(base_row["rounds_completed"])
            metrics["Baseline_Default_FedAvg"]["bytes"].append(base_row["total_wire_bytes"])

            # Random
            metrics["Baseline_Random_Pick"]["acc"].append(scen_df["final_accuracy"].mean())
            metrics["Baseline_Random_Pick"]["success"].append((scen_df["final_accuracy"] >= target).mean())
            metrics["Baseline_Random_Pick"]["rounds"].append(scen_df["rounds_completed"].mean())
            metrics["Baseline_Random_Pick"]["bytes"].append(scen_df["total_wire_bytes"].mean())

            # Handwritten Rule
            rule_row = scen_df[scen_df["config_id"] == r["rule_chosen"]].iloc[0]
            metrics["Baseline_Handwritten_Rule"]["acc"].append(rule_row["final_accuracy"])
            metrics["Baseline_Handwritten_Rule"]["success"].append(1 if rule_row["final_accuracy"] >= target else 0)
            metrics["Baseline_Handwritten_Rule"]["rounds"].append(rule_row["rounds_completed"])
            metrics["Baseline_Handwritten_Rule"]["bytes"].append(rule_row["total_wire_bytes"])

        dom_res = {}
        for m in methods:
            dom_res[m] = {
                "mean_acc": float(np.mean(metrics[m]["acc"])),
                "success_rate_pct": float(np.mean(metrics[m]["success"]) * 100.0),
                "mean_rounds": float(np.mean(metrics[m]["rounds"])),
                "mean_mb": float(np.mean(metrics[m]["bytes"]) / (1024 * 1024)),
            }
        results[domain_id] = {
            "target": target,
            "metrics": dom_res,
        }

    return results


def evaluate_grid_v2():
    """Evaluates expanded grid_v2 (7 scenarios per domain, 16 configs)."""
    df = pd.read_parquet("results/stage4_grid.parquet")
    df = df[df["config_id"].str.startswith("exp_")]

    results = {}
    scenario_map = {
        "iot_edge": IOT_SCENARIOS,
        "mobile_apps": MOBILE_SCENARIOS,
    }

    for domain_id in ["mobile_apps", "iot_edge"]:
        target = 85.0 if domain_id == "mobile_apps" else 70.0
        dom_df = df[df["domain"] == domain_id].copy()
        scenarios = scenario_map[domain_id]

        methods = ["Recommender_Proposed", "Baseline_Default_FedAvg", "Baseline_Random_Pick"]
        metrics = {m: {"acc": [], "success": [], "rounds": [], "bytes": []} for m in methods}

        # Multi-criteria scoring weights (balanced: acc=0.4, comm=0.3, lat=0.3)
        w = {"acc": 0.40, "comm": 0.30, "lat": 0.30}

        for held_out_s in scenarios:
            sid = held_out_s.scenario_id
            held_out_df = dom_df[dom_df["scenario_id"] == sid]
            if held_out_df.empty:
                continue

            # Train set is all OTHER scenarios in this domain
            train_df = dom_df[dom_df["scenario_id"] != sid]

            # Score each config across train scenarios
            agg = train_df.groupby("config_id").agg({
                "final_accuracy": "mean",
                "total_wire_bytes": "mean",
                "total_time_seconds": "mean",
            }).reset_index()

            # Normalization
            accs = agg["final_accuracy"].values
            wires = agg["total_wire_bytes"].values
            times = agg["total_time_seconds"].values

            acc_norm = (accs - accs.min()) / (accs.max() - accs.min() + 1e-9)
            wire_norm = 1.0 - ((wires - wires.min()) / (wires.max() - wires.min() + 1e-9))
            time_norm = 1.0 - ((times - times.min()) / (times.max() - times.min() + 1e-9))

            scores = w["acc"] * acc_norm + w["comm"] * wire_norm + w["lat"] * time_norm
            best_idx = int(np.argmax(scores))
            rec_cid = agg.iloc[best_idx]["config_id"]

            # Evaluate on held-out scenario
            rec_row = held_out_df[held_out_df["config_id"] == rec_cid].iloc[0]
            metrics["Recommender_Proposed"]["acc"].append(rec_row["final_accuracy"])
            metrics["Recommender_Proposed"]["success"].append(1 if rec_row["final_accuracy"] >= target else 0)
            metrics["Recommender_Proposed"]["rounds"].append(rec_row["rounds_completed"])
            metrics["Recommender_Proposed"]["bytes"].append(rec_row["total_wire_bytes"])

            # Default FedAvg (exp_01_fedavg)
            base_row = held_out_df[held_out_df["config_id"] == "exp_01_fedavg"].iloc[0]
            metrics["Baseline_Default_FedAvg"]["acc"].append(base_row["final_accuracy"])
            metrics["Baseline_Default_FedAvg"]["success"].append(1 if base_row["final_accuracy"] >= target else 0)
            metrics["Baseline_Default_FedAvg"]["rounds"].append(base_row["rounds_completed"])
            metrics["Baseline_Default_FedAvg"]["bytes"].append(base_row["total_wire_bytes"])

            # Random Pick (mean over all 16 configs)
            metrics["Baseline_Random_Pick"]["acc"].append(held_out_df["final_accuracy"].mean())
            metrics["Baseline_Random_Pick"]["success"].append((held_out_df["final_accuracy"] >= target).mean())
            metrics["Baseline_Random_Pick"]["rounds"].append(held_out_df["rounds_completed"].mean())
            metrics["Baseline_Random_Pick"]["bytes"].append(held_out_df["total_wire_bytes"].mean())

        dom_res = {}
        for m in methods:
            dom_res[m] = {
                "mean_acc": float(np.mean(metrics[m]["acc"])),
                "success_rate_pct": float(np.mean(metrics[m]["success"]) * 100.0),
                "mean_rounds": float(np.mean(metrics[m]["rounds"])),
                "mean_mb": float(np.mean(metrics[m]["bytes"]) / (1024 * 1024)),
            }
        results[domain_id] = {
            "target": target,
            "metrics": dom_res,
        }

    return results


if __name__ == "__main__":
    res_v1 = evaluate_grid_v1()
    res_v2 = evaluate_grid_v2()

    output = {
        "grid_v1_canonical": res_v1,
        "grid_v2_expanded": res_v2,
    }
    with open("results/grid_v1_v2_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    print("SUCCESS: Saved grid comparison to results/grid_v1_v2_comparison.json")
    print(json.dumps(output, indent=2))
