"""
Comprehensive Systematic Re-Analysis on Expanded Grid (grid_v2):
1. Evaluates all 6 methods under Leave-One-Scenario-Out (LOSO):
   - Recommender (Proposed)
   - Baseline: Default FedAvg
   - Baseline: Random Pick
   - Baseline: Hand-written Rule
   - Baseline: Domain-Blind Recommender
   - Baseline: Simple KNN Regressor
2. Uses ONE justified target per domain (85% of clean IID ceiling):
   - Mobile Apps: 73.3% (0.85 * 86.24%)
   - IoT Edge: 72.2% (0.85 * 84.93%)
3. Computes:
   - Mean Accuracy, Success Rate, Mean Uplink MB, Mean Simulation Time
   - Regret vs Pool-Restricted Oracle
   - Wins / Ties / Losses vs Recommender
   - Paired Bootstrap 95% Confidence Intervals
   - Per-Scenario Detailed Breakdown (showing attack & privacy scenarios)
   - Constraint-Satisfaction Rates (Privacy, Bandwidth, and Byzantine Robustness)
"""

import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from flaas.grid.expanded_scenarios import IOT_SCENARIOS, MOBILE_SCENARIOS, ExpandedScenarioSpec
from flaas.recommender.scenario_matching import scenario_to_feature_vector


def paired_bootstrap_ci(diffs: np.ndarray, n_bootstraps: int = 10000, ci: float = 95.0, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_means = [rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_bootstraps)]
    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha))
    upper = float(np.percentile(boot_means, 100.0 - alpha))
    return lower, upper


def handwritten_rule_v2(scenario: ExpandedScenarioSpec) -> str:
    """Literature-grounded baseline decision rule."""
    if scenario.attack_type is not None and scenario.attack_fraction > 0:
        return "exp_06_trimmed_mean"
    elif scenario.privacy_mode == "client_dp":
        return "exp_12_dp_eps8"
    elif scenario.participation_rate <= (0.1 if scenario.domain_id == "mobile_apps" else 0.2):
        return "exp_10_ternary_compression"
    elif scenario.heterogeneity_alpha is not None and scenario.heterogeneity_alpha <= 0.1:
        return "exp_02_fedprox"
    else:
        return "exp_01_fedavg"


def run_full_grid_v2_evaluation():
    df = pd.read_parquet("results/stage4_grid.parquet")
    df = df[df["config_id"].str.startswith("exp_")].copy()

    scenario_registry = {
        "mobile_apps": MOBILE_SCENARIOS,
        "iot_edge": IOT_SCENARIOS,
    }

    # Fixed, justified targets: 85% of clean IID scenario max accuracy
    targets = {
        "mobile_apps": 73.30,  # 86.24 * 0.85
        "iot_edge": 72.19,     # 84.93 * 0.85
    }

    methods = [
        "Recommender_Proposed",
        "Baseline_Default_FedAvg",
        "Baseline_Random_Pick",
        "Baseline_Handwritten_Rule",
        "Baseline_Domain_Blind",
        "Baseline_Simple_KNN",
    ]

    all_domain_results = {}

    for domain_id in ["mobile_apps", "iot_edge"]:
        dom_df = df[df["domain"] == domain_id].copy()
        scenarios = scenario_registry[domain_id]
        target_acc = targets[domain_id]
        candidate_cids = sorted(dom_df["config_id"].unique().tolist())

        # Track per-method records
        method_metrics = {m: {
            "acc": [], "success": [], "bytes": [], "time": [], "regret": [],
            "satisfied_privacy": [], "satisfied_bandwidth": [], "satisfied_robustness": []
        } for m in methods}

        per_scenario_records = []

        w = {"acc": 0.40, "comm": 0.30, "lat": 0.30}

        for held_out_s in scenarios:
            sid = held_out_s.scenario_id
            held_out_df = dom_df[dom_df["scenario_id"] == sid]
            if held_out_df.empty:
                continue

            # Ground truth map
            perf_map = dict(zip(held_out_df["config_id"], held_out_df["final_accuracy"]))
            wire_map = dict(zip(held_out_df["config_id"], held_out_df["total_wire_bytes"]))
            time_map = dict(zip(held_out_df["config_id"], held_out_df["total_time_seconds"]))

            # Pool-restricted oracle
            oracle_cid = max(perf_map.keys(), key=lambda c: perf_map[c])
            oracle_acc = perf_map[oracle_cid]

            # 1. Proposed Recommender (LOSO: trained on remaining scenarios in domain)
            train_dom_df = dom_df[dom_df["scenario_id"] != sid]
            agg = train_dom_df.groupby("config_id").agg({
                "final_accuracy": "mean",
                "total_wire_bytes": "mean",
                "total_time_seconds": "mean",
            }).reset_index()

            # Filter feasible if scenario requires DP or Byzantine defense
            require_byz = held_out_s.attack_type is not None and held_out_s.attack_fraction > 0
            require_dp = held_out_s.privacy_mode == "client_dp"
            require_comp = held_out_s.participation_rate <= (0.1 if domain_id == "mobile_apps" else 0.2)

            filtered_agg = []
            for _, row in agg.iterrows():
                cid = row["config_id"]
                if require_byz and not any(k in cid for k in ["trimmed", "krum", "bulyan", "median"]):
                    continue
                if require_dp and not any(k in cid for k in ["dp"]):
                    continue
                if require_comp and not any(k in cid for k in ["ternary", "qsgd"]):
                    continue
                filtered_agg.append(row)
            sub_agg = pd.DataFrame(filtered_agg) if filtered_agg else agg

            accs = sub_agg["final_accuracy"].values
            wires = sub_agg["total_wire_bytes"].values
            times = sub_agg["total_time_seconds"].values

            acc_norm = (accs - accs.min()) / (accs.max() - accs.min() + 1e-9)
            wire_norm = 1.0 - ((wires - wires.min()) / (wires.max() - wires.min() + 1e-9))
            time_norm = 1.0 - ((times - times.min()) / (times.max() - times.min() + 1e-9))
            scores = w["acc"] * acc_norm + w["comm"] * wire_norm + w["lat"] * time_norm
            rec_cid = sub_agg.iloc[int(np.argmax(scores))]["config_id"]

            # 2. Default FedAvg
            fedavg_cid = "exp_01_fedavg"

            # 3. Handwritten Rule
            rule_cid = handwritten_rule_v2(held_out_s)

            # 4. Domain-Blind Recommender (trained on ALL other domain scenarios)
            train_all_df = df[(df["scenario_id"] != sid)]
            blind_agg = train_all_df.groupby("config_id")["final_accuracy"].mean()
            blind_cid = blind_agg.idxmax()

            # 5. Simple KNN Model (trained on remaining scenarios in domain)
            knn_X = []
            knn_y = {c: [] for c in candidate_cids}
            for tr_s in scenarios:
                if tr_s.scenario_id == sid:
                    continue
                feat = scenario_to_feature_vector(
                    tr_s.heterogeneity_alpha, tr_s.participation_rate,
                    tr_s.attack_type, tr_s.attack_fraction, tr_s.attack_scale, tr_s.privacy_mode
                )
                knn_X.append(feat)
                tr_df = dom_df[dom_df["scenario_id"] == tr_s.scenario_id]
                for c in candidate_cids:
                    match_row = tr_df[tr_df["config_id"] == c]
                    acc_val = match_row["final_accuracy"].values[0] if not match_row.empty else 0.0
                    knn_y[c].append(acc_val)

            knn_X_arr = np.array(knn_X)
            knn_q = scenario_to_feature_vector(
                held_out_s.heterogeneity_alpha, held_out_s.participation_rate,
                held_out_s.attack_type, held_out_s.attack_fraction, held_out_s.attack_scale, held_out_s.privacy_mode
            ).reshape(1, -1)

            knn_preds = {}
            for c in candidate_cids:
                model = KNeighborsRegressor(n_neighbors=min(3, len(knn_X)), weights="distance")
                model.fit(knn_X_arr, np.array(knn_y[c]))
                knn_preds[c] = float(model.predict(knn_q)[0])
            knn_cid = max(knn_preds.keys(), key=lambda c: knn_preds[c])

            # Pick mapping
            picks = {
                "Recommender_Proposed": rec_cid,
                "Baseline_Default_FedAvg": fedavg_cid,
                "Baseline_Handwritten_Rule": rule_cid,
                "Baseline_Domain_Blind": blind_cid,
                "Baseline_Simple_KNN": knn_cid,
            }

            # Helper for constraint satisfaction
            def check_constraints(cid_chosen):
                # Privacy satisfied: if required, did it pick DP?
                priv_sat = True
                if require_dp and "dp" not in cid_chosen:
                    priv_sat = False

                # Bandwidth satisfied: if required, did it pick compressed?
                bw_sat = True
                if require_comp and not any(k in cid_chosen for k in ["ternary", "qsgd"]):
                    bw_sat = False

                # Robustness satisfied: if attack, did it pick robust aggregator?
                rob_sat = True
                if require_byz and not any(k in cid_chosen for k in ["trimmed", "krum", "bulyan", "median"]):
                    rob_sat = False

                return priv_sat, bw_sat, rob_sat

            scen_record = {
                "scenario_id": sid,
                "scenario_name": held_out_s.name,
                "oracle_config": oracle_cid,
                "oracle_acc": oracle_acc,
                "picks": {}
            }

            for m in [k for k in methods if k != "Baseline_Random_Pick"]:
                c_pick = picks[m]
                acc = perf_map.get(c_pick, 0.0)
                wire = wire_map.get(c_pick, 0)
                t_sec = time_map.get(c_pick, 0.0)
                reg = oracle_acc - acc
                succ = 1 if acc >= target_acc else 0
                ps, bs, rs = check_constraints(c_pick)

                method_metrics[m]["acc"].append(acc)
                method_metrics[m]["success"].append(succ)
                method_metrics[m]["bytes"].append(wire)
                method_metrics[m]["time"].append(t_sec)
                method_metrics[m]["regret"].append(reg)
                method_metrics[m]["satisfied_privacy"].append(1 if ps else 0)
                method_metrics[m]["satisfied_bandwidth"].append(1 if bs else 0)
                method_metrics[m]["satisfied_robustness"].append(1 if rs else 0)

                scen_record["picks"][m] = {
                    "config_id": c_pick,
                    "acc": acc,
                    "wire_mb": wire / (1024 * 1024),
                    "regret": reg,
                    "success": succ,
                }

            # Random pick
            rand_acc = float(np.mean(list(perf_map.values())))
            rand_wire = float(np.mean(list(wire_map.values())))
            rand_time = float(np.mean(list(time_map.values())))
            rand_reg = oracle_acc - rand_acc
            rand_succ = float(np.mean([1 if a >= target_acc else 0 for a in perf_map.values()]))
            # Random constraint satisfaction
            all_ps, all_bs, all_rs = [], [], []
            for c in candidate_cids:
                ps, bs, rs = check_constraints(c)
                all_ps.append(1 if ps else 0)
                all_bs.append(1 if bs else 0)
                all_rs.append(1 if rs else 0)

            method_metrics["Baseline_Random_Pick"]["acc"].append(rand_acc)
            method_metrics["Baseline_Random_Pick"]["success"].append(rand_succ)
            method_metrics["Baseline_Random_Pick"]["bytes"].append(rand_wire)
            method_metrics["Baseline_Random_Pick"]["time"].append(rand_time)
            method_metrics["Baseline_Random_Pick"]["regret"].append(rand_reg)
            method_metrics["Baseline_Random_Pick"]["satisfied_privacy"].append(float(np.mean(all_ps)))
            method_metrics["Baseline_Random_Pick"]["satisfied_bandwidth"].append(float(np.mean(all_bs)))
            method_metrics["Baseline_Random_Pick"]["satisfied_robustness"].append(float(np.mean(all_rs)))

            scen_record["picks"]["Baseline_Random_Pick"] = {
                "config_id": "random_expectation",
                "acc": rand_acc,
                "wire_mb": rand_wire / (1024 * 1024),
                "regret": rand_reg,
                "success": rand_succ,
            }

            per_scenario_records.append(scen_record)

        # Summary Metrics across 7 scenarios
        summary_table = {}
        for m in methods:
            accs = np.array(method_metrics[m]["acc"])
            succs = np.array(method_metrics[m]["success"])
            wires = np.array(method_metrics[m]["bytes"]) / (1024 * 1024)
            times = np.array(method_metrics[m]["time"])
            regrets = np.array(method_metrics[m]["regret"])

            summary_table[m] = {
                "mean_acc": float(np.mean(accs)),
                "success_rate_pct": float(np.mean(succs) * 100.0),
                "mean_uplink_mb": float(np.mean(wires)),
                "mean_sim_time_sec": float(np.mean(times)),
                "mean_regret": float(np.mean(regrets)),
                "privacy_satisfaction_pct": float(np.mean(method_metrics[m]["satisfied_privacy"]) * 100.0),
                "bandwidth_satisfaction_pct": float(np.mean(method_metrics[m]["satisfied_bandwidth"]) * 100.0),
                "robustness_satisfaction_pct": float(np.mean(method_metrics[m]["satisfied_robustness"]) * 100.0),
            }

        # Pairwise wins / ties / losses and bootstrap 95% CI vs Recommender
        pairwise = {}
        rec_accs = np.array(method_metrics["Recommender_Proposed"]["acc"])
        for b in [m for m in methods if m != "Recommender_Proposed"]:
            b_accs = np.array(method_metrics[b]["acc"])
            diffs = rec_accs - b_accs

            wins = int(np.sum(diffs > 0.05))
            ties = int(np.sum(np.abs(diffs) <= 0.05))
            losses = int(np.sum(diffs < -0.05))
            mean_diff = float(np.mean(diffs))
            ci_low, ci_high = paired_bootstrap_ci(diffs, n_bootstraps=10000, ci=95.0, seed=42)

            pairwise[b] = {
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "mean_diff": mean_diff,
                "bootstrap_95_ci": [ci_low, ci_high],
            }

        all_domain_results[domain_id] = {
            "domain_id": domain_id,
            "target_accuracy": target_acc,
            "target_justification": f"85% of clean IID scenario max accuracy ({target_acc:.1f}%)",
            "summary_by_method": summary_table,
            "pairwise_vs_recommender": pairwise,
            "per_scenario_records": per_scenario_records,
        }

    return all_domain_results


if __name__ == "__main__":
    res = run_full_grid_v2_evaluation()
    with open("results/grid_v2_full_reanalysis.json", "w") as f:
        json.dump(res, f, indent=2)
    print("SUCCESS: Full grid_v2 re-analysis complete and saved to results/grid_v2_full_reanalysis.json")
