"""
Stage 5 Evaluation Script:
Executes systematic Leave-One-Scenario-Out (LOSO) cross-validation for Mobile Apps and IoT Edge,
comparing the FLaaS Recommender against:
1. Default FedAvg
2. Random Pick
3. Hand-Written Rule
4. Domain-Blind Recommender
5. Simple KNN Model Baseline

Saves results to results/stage5_loso_evaluation.json and prints summary table.
"""

import os
import json
from flaas.recommender.evaluation import LeaveOneScenarioOutEvaluator
from flaas.recommender.domain_gate import ACTIVE_DOMAINS, PENDING_DOMAINS, DATASET_LABEL


def run_evaluation():
    print("=" * 80)
    print("STAGE 5: SYSTEMATIC LEAVE-ONE-SCENARIO-OUT (LOSO) EVALUATION")
    print(f"Active Domains (Approved): {ACTIVE_DOMAINS}")
    print(f"Pending Domains (Excluded): {list(PENDING_DOMAINS.keys())}")
    print(f"Evidence Label: {DATASET_LABEL}")
    print("=" * 80)

    evaluator = LeaveOneScenarioOutEvaluator(parquet_path="results/stage4_grid.parquet")
    all_results = evaluator.evaluate_all_active_domains()

    # Save to JSON
    os.makedirs("results", exist_ok=True)
    out_path = "results/stage5_loso_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved complete LOSO evaluation results to {out_path}\n")

    # Print summary tables
    for domain_id, res in all_results.items():
        print("-" * 80)
        print(f"DOMAIN: {res['domain_name']} ({domain_id}) &mdash; 16 Scenarios Evaluated")
        print("-" * 80)
        print(f"{'Method / Baseline':<30} | {'Mean Acc (%)':<14} | {'Mean Regret (%)':<16} | {'Mean Rank (1=best)':<18}")
        print("-" * 80)
        summary = res["summary_by_method"]
        for method, metrics in summary.items():
            acc = f"{metrics['mean_accuracy']:.2f}%"
            regret = f"{metrics['mean_regret']:.2f}%"
            rank = f"{metrics['mean_rank']:.2f}"
            print(f"{method:<30} | {acc:<14} | {regret:<16} | {rank:<18}")
        print("-" * 80)


if __name__ == "__main__":
    run_evaluation()
