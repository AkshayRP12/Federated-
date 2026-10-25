"""
Scenarios and FL Configurations for Stage 4 Knowledge Base Generation.
Defines:
- 16 realistic federated learning operational scenarios.
- 8 candidate FL configurations per domain with architectural rationale.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ScenarioSpec:
    scenario_id: str
    name: str
    description: str
    heterogeneity_alpha: Optional[float]  # None for IID
    participation_rate: float
    attack_type: Optional[str]  # None, 'sign_flip', 'gaussian_noise', 'scaled_update'
    attack_fraction: float
    attack_scale: float
    privacy_mode: str  # 'none', 'client_dp', 'example_dp'
    privacy_target_epsilon: Optional[float]


@dataclass
class ConfigSpec:
    config_id: str
    name: str
    client_algo: str  # 'FedAvg', 'FedProx', 'SCAFFOLD', 'FedNova'
    client_algo_kwargs: Dict[str, Any]
    aggregation_rule: str  # 'FedAvgMean', 'TrimmedMean', 'Krum', 'Bulyan'
    aggregation_kwargs: Dict[str, Any]
    server_optimizer: str  # 'ServerSGD', 'ServerAdam'
    server_opt_kwargs: Dict[str, Any]
    compressor: str  # 'identity', 'ternary_2bit', 'qsgd_8'
    dp_type: str  # 'none', 'client', 'example'
    dp_kwargs: Dict[str, Any]
    rationale: str


# ==============================================================================
# 16 Canonical Scenarios
# ==============================================================================
SCENARIOS: List[ScenarioSpec] = [
    # Baseline & Heterogeneity Scenarios
    ScenarioSpec("scen_01", "Standard Clean IID", "Homogeneous data distribution, full participation, no adversary, no privacy.", None, 1.0, None, 0.0, 1.0, "none", None),
    ScenarioSpec("scen_02", "Mild Non-IID", "Realistic mild Dirichlet label skew (alpha=0.5), full participation.", 0.5, 1.0, None, 0.0, 1.0, "none", None),
    ScenarioSpec("scen_03", "Moderate Non-IID", "Moderate Dirichlet label skew (alpha=0.3), full participation.", 0.3, 1.0, None, 0.0, 1.0, "none", None),
    ScenarioSpec("scen_04", "Severe Non-IID", "Severe Dirichlet label skew (alpha=0.1), extreme client local drift.", 0.1, 1.0, None, 0.0, 1.0, "none", None),
    
    # Partial Participation Scenarios
    ScenarioSpec("scen_05", "Partial Participation Moderate (50%)", "50% random client cohort sampling per round under mild skew (alpha=0.5).", 0.5, 0.5, None, 0.0, 1.0, "none", None),
    ScenarioSpec("scen_06", "Partial Participation Mobile (20%)", "20% mobile device availability under moderate skew (alpha=0.3).", 0.3, 0.2, None, 0.0, 1.0, "none", None),
    ScenarioSpec("scen_07", "Extreme Heterogeneity & Mobile Participation", "20% cohort sampling under severe Dirichlet skew (alpha=0.1).", 0.1, 0.2, None, 0.0, 1.0, "none", None),
    
    # Byzantine Adversarial Attack Scenarios (20% & 30% adversaries)
    ScenarioSpec("scen_08", "Sign-Flip Attack Mild Skew", "20% Byzantine sign-flip adversaries under mild skew (alpha=0.5).", 0.5, 1.0, "sign_flip", 0.2, 2.0, "none", None),
    ScenarioSpec("scen_09", "Sign-Flip Attack Severe Skew", "20% Byzantine sign-flip adversaries under severe skew (alpha=0.1).", 0.1, 1.0, "sign_flip", 0.2, 2.0, "none", None),
    ScenarioSpec("scen_10", "Gaussian Noise Attack", "20% Byzantine adversaries injecting zero-mean Gaussian noise (std=1.0).", 0.3, 1.0, "gaussian_noise", 0.2, 1.0, "none", None),
    ScenarioSpec("scen_11", "Scaled Update Attack", "20% Byzantine adversaries sending 10x scaled malicious updates.", 0.3, 1.0, "scaled_update", 0.2, 10.0, "none", None),
    ScenarioSpec("scen_12", "High-Adversary Attack (30%)", "30% Byzantine sign-flip adversaries testing breakdown limits of robust aggregators.", 0.5, 1.0, "sign_flip", 0.3, 2.0, "none", None),

    # Differential Privacy Scenarios
    ScenarioSpec("scen_13", "Client DP Standard (eps=8.0)", "Cross-silo institutional protection under mild skew (alpha=0.5), epsilon=8.0.", 0.5, 1.0, None, 0.0, 1.0, "client_dp", 8.0),
    ScenarioSpec("scen_14", "Client DP Strict (eps=4.0)", "High-privacy regulatory regime (HIPAA/GDPR) under mild skew, epsilon=4.0.", 0.5, 1.0, None, 0.0, 1.0, "client_dp", 4.0),
    ScenarioSpec("scen_15", "Client DP with Partial Participation", "Client-level DP under 50% Poisson client sampling (epsilon=8.0).", 0.5, 0.5, None, 0.0, 1.0, "client_dp", 8.0),
    ScenarioSpec("scen_16", "Compound Challenge: Skew + Attack + Privacy", "Compound stress test: moderate skew (0.3), 20% sign-flip, and DP protection.", 0.3, 1.0, "sign_flip", 0.2, 2.0, "client_dp", 8.0),
]


# ==============================================================================
# 8 Candidate FL Configurations
# ==============================================================================
def get_domain_configs(domain_id: str, num_clients: int) -> List[ConfigSpec]:
    """
    Returns 8 diverse, well-motivated FL configurations tailored to the domain.
    """
    f_trim = 0.2
    f_krum = max(1, int(num_clients * 0.2))
    
    configs = [
        # 1. Baseline
        ConfigSpec(
            config_id="cfg_01_baseline",
            name="Vanilla FedAvg Baseline",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            rationale="Standard canonical baseline without defense, compression, or privacy constraints.",
        ),
        # 2. Heterogeneity-Robust (FedProx)
        ConfigSpec(
            config_id="cfg_02_fedprox",
            name="FedProx Heterogeneity Regularizer",
            client_algo="FedProx",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32, "mu": 0.01},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            rationale="Constrains local client model deviation using proximal regularization (mu=0.01).",
        ),
        # 3. Drift-Robust (SCAFFOLD or FedNova)
        ConfigSpec(
            config_id="cfg_03_scaffold_nova",
            name="Variance-Reduced SCAFFOLD / FedNova",
            client_algo="SCAFFOLD" if domain_id in ["healthcare", "finance"] else "FedNova",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            rationale="Corrects client gradient drift across heterogeneous local epochs using control variates / normalized updates.",
        ),
        # 4. Byzantine-Robust TrimmedMean
        ConfigSpec(
            config_id="cfg_04_trimmed_mean",
            name="TrimmedMean Coordinate Robust Aggregation",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="TrimmedMean",
            aggregation_kwargs={"beta": f_trim},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            rationale="Coordinate-wise trimming of extreme 20% updates protects against arbitrary scaled and sign-flip attacks.",
        ),
        # 5. Byzantine-Robust MultiKrum / Bulyan
        ConfigSpec(
            config_id="cfg_05_krum_bulyan",
            name="MultiKrum Distance-Based Aggregation",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="MultiKrum",
            aggregation_kwargs={"f": f_krum, "m": max(1, num_clients - f_krum - 2)},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            rationale="Geometric distance clustering filters out malicious vectors that deviate in Euclidean space.",
        ),
        # 6. Communication-Efficient (Ternary 2-bit packing)
        ConfigSpec(
            config_id="cfg_06_ternary_compression",
            name="Ternary 2-Bit Packing (16x Compression)",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="ternary_2bit",
            dp_type="none",
            dp_kwargs={},
            rationale="Compresses 32-bit float parameters into 2-bit ternary states {-1, 0, +1}, cutting uplink bandwidth by 16x.",
        ),
        # 7. Privacy-Preserving Client DP
        ConfigSpec(
            config_id="cfg_07_client_dp",
            name="Client-Level DP Gaussian Mechanism",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="client",
            dp_kwargs={"clip_norm": 1.0, "noise_multiplier": 0.5},
            rationale="Protects entire client silos from membership inference and model inversion with provable RDP guarantees.",
        ),
        # 8. Domain-Optimal Composite Profile
        ConfigSpec(
            config_id="cfg_08_domain_optimal",
            name=f"Domain-Optimal Composite ({domain_id})",
            client_algo="FedProx" if domain_id in ["healthcare", "mobile_apps", "autonomous_vehicles"] else "FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32, "mu": 0.01},
            aggregation_rule="TrimmedMean" if domain_id in ["finance", "autonomous_vehicles"] else "FedAvgMean",
            aggregation_kwargs={"beta": 0.15} if domain_id in ["finance", "autonomous_vehicles"] else {},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="ternary_2bit" if domain_id in ["iot_edge", "mobile_apps"] else "identity",
            dp_type="client" if domain_id == "healthcare" else "none",
            dp_kwargs={"clip_norm": 1.0, "noise_multiplier": 0.3} if domain_id == "healthcare" else {},
            rationale="Holistic composite combining heterogeneity defense, robust aggregation, and bandwidth optimization.",
        ),
    ]
    return configs
