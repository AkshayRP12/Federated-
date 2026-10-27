"""
Plain-Language Rationale Generator for FLaaS Recommender.
Synthesizes transparent, human-readable explanations citing exact empirical experiment numbers,
trade-offs against baseline FedAvg, and Pareto front optimality.
Labels all results as 1-seed.
"""

from typing import Dict, Any, Optional
from flaas.recommender.multi_criteria import ScoredConfig
from flaas.grid.scenarios import ScenarioSpec, ConfigSpec


class RationaleGenerator:
    """Generates empirical plain-language justifications for recommended configurations."""

    @staticmethod
    def generate_rationale(
        recommended: ScoredConfig,
        baseline: Optional[ScoredConfig],
        scenario: ScenarioSpec,
        config_spec: ConfigSpec,
        domain_name: str,
    ) -> str:
        """
        Builds a structured, human-interpretable rationale citing empirical experiment numbers.
        """
        lines = []

        # 1. Headline recommendation
        lines.append(
            f"**Recommended Configuration**: `{recommended.name}` ({config_spec.name}) for **{domain_name}**."
        )

        # 2. Operational context
        scen_desc = scenario.description
        lines.append(f"- **Operating Environment**: {scenario.name} &mdash; {scen_desc}")

        # 3. Empirical Accuracy Citation
        acc_str = f"{recommended.raw_accuracy:.2f}%"
        if baseline is not None and baseline.config_id != recommended.config_id:
            base_acc_str = f"{baseline.raw_accuracy:.2f}%"
            diff = recommended.raw_accuracy - baseline.raw_accuracy
            sign = "+" if diff >= 0 else ""
            lines.append(
                f"- **Empirical Accuracy**: Reached **{acc_str}** test accuracy in {recommended.rounds_completed} rounds "
                f"({sign}{diff:.2f}% relative to default FedAvg at {base_acc_str})."
            )
        else:
            lines.append(f"- **Empirical Accuracy**: Reached **{acc_str}** test accuracy in {recommended.rounds_completed} rounds.")

        # 4. Communication & Wire Efficiency
        rec_mb = recommended.raw_wire_bytes / (1024 * 1024)
        if baseline is not None and baseline.raw_wire_bytes > 0:
            base_mb = baseline.raw_wire_bytes / (1024 * 1024)
            ratio = base_mb / max(1e-4, rec_mb)
            if ratio >= 1.2:
                lines.append(
                    f"- **Communication Efficiency**: Transmitted **{rec_mb:.2f} MB** total wire bytes compared to "
                    f"**{base_mb:.2f} MB** for uncompressed baseline ({ratio:.1f}x bandwidth reduction via {config_spec.compressor})."
                )
            else:
                lines.append(f"- **Communication Footprint**: Total wire transmission of **{rec_mb:.2f} MB** across all rounds.")
        else:
            lines.append(f"- **Communication Footprint**: Total wire transmission of **{rec_mb:.2f} MB**.")

        # 5. Latency & Early Stopping
        lines.append(
            f"- **Execution Latency**: Completed in **{recommended.raw_time_seconds:.1f} seconds** "
            f"({'Early-stopped at target threshold' if recommended.early_stopped else 'Ran to max communication rounds'})."
        )

        # 6. Defense & Privacy Mechanism
        if scenario.attack_type is not None:
            if recommended.aggregation_rule in ["TrimmedMean", "Krum", "MultiKrum"]:
                lines.append(
                    f"- **Byzantine Defense**: Robust aggregation via `{recommended.aggregation_rule}` actively filtered out "
                    f"perturbed updates from {int(scenario.attack_fraction * 100)}% Byzantine {scenario.attack_type} adversaries."
                )
            else:
                lines.append(
                    f"- **Adversary Note**: Operating under {int(scenario.attack_fraction * 100)}% {scenario.attack_type} attack."
                )

        if recommended.dp_type == "client" or scenario.privacy_mode == "client_dp":
            lines.append(
                f"- **Differential Privacy**: Enforced Client-Level DP with cumulative budget "
                f"epsilon = {recommended.epsilon:.2f} (delta = 1e-5) via continuous Rényi DP accountant."
            )

        # 7. Pareto Optimality
        if recommended.is_pareto_optimal:
            lines.append(
                "- **Pareto Optimality**: Configuration is mathematically non-dominated on the empirical "
                "Accuracy vs. Communication vs. Latency Pareto front."
            )

        # 8. Empirical Evidence Label
        lines.append(
            "- **Empirical Evidence**: All metrics derived from canonical 1-seed grid execution (seed=42) "
            "under Dirichlet class partitioning."
        )

        return "\n".join(lines)
