"""
Expanded Domain-Specific Scenarios and Configurations for Mobile and IoT Edge.
Adheres strictly to user specifications:
- Domain-specific scenarios with realistic client counts and participation.
- Added SCAFFOLD, FedNova, FedAdam, Bulyan, Median, QSGD.
- DP as an explicit configuration axis: off, eps 8, eps 4, eps 1.
- Leaves frozen kb-v1 profiles untouched.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ExpandedScenarioSpec:
    scenario_id: str
    name: str
    description: str
    domain_id: str
    num_clients: int
    heterogeneity_alpha: Optional[float]  # None for IID
    participation_rate: float
    attack_type: Optional[str]  # None, 'sign_flip', 'gaussian_noise', 'scaled_update'
    attack_fraction: float
    attack_scale: float
    privacy_mode: str  # 'none', 'client_dp'
    privacy_target_epsilon: Optional[float]


@dataclass
class ExpandedConfigSpec:
    config_id: str
    name: str
    client_algo: str  # 'FedAvg', 'FedProx', 'SCAFFOLD', 'FedNova'
    client_algo_kwargs: Dict[str, Any]
    aggregation_rule: str  # 'FedAvgMean', 'TrimmedMean', 'Krum', 'Bulyan', 'CoordinateMedian'
    aggregation_kwargs: Dict[str, Any]
    server_optimizer: str  # 'ServerSGD', 'ServerAdam'
    server_opt_kwargs: Dict[str, Any]
    compressor: str  # 'identity', 'ternary_2bit', 'qsgd_8'
    dp_type: str  # 'none', 'client'
    dp_kwargs: Dict[str, Any]
    dp_target_epsilon: Optional[float]  # None (off), 8.0, 4.0, 1.0
    rationale: str


# ==============================================================================
# Domain-Specific Scenarios for IoT Edge (20 clients)
# ==============================================================================
IOT_SCENARIOS: List[ExpandedScenarioSpec] = [
    ExpandedScenarioSpec(
        "iot_scen_01_iid",
        "IoT Uniform Telemetry (IID)",
        "20 wearable edge devices with uniform sensor activity distributions and 50% participation.",
        domain_id="iot_edge",
        num_clients=20,
        heterogeneity_alpha=None,
        participation_rate=0.5,
        attack_type=None,
        attack_fraction=0.0,
        attack_scale=1.0,
        privacy_mode="none",
        privacy_target_epsilon=None,
    ),
    ExpandedScenarioSpec(
        "iot_scen_02_mild_skew",
        "IoT Realistic User Skew (alpha=0.5)",
        "20 devices with realistic user habit variance (Dirichlet alpha=0.5) and 50% battery availability.",
        domain_id="iot_edge",
        num_clients=20,
        heterogeneity_alpha=0.5,
        participation_rate=0.5,
        attack_type=None,
        attack_fraction=0.0,
        attack_scale=1.0,
        privacy_mode="none",
        privacy_target_epsilon=None,
    ),
    ExpandedScenarioSpec(
        "iot_scen_03_severe_skew",
        "IoT Extreme Specialized Users (alpha=0.1)",
        "20 devices with severe behavioral skew where users engage in 1-2 dominant activities.",
        domain_id="iot_edge",
        num_clients=20,
        heterogeneity_alpha=0.1,
        participation_rate=0.5,
        attack_type=None,
        attack_fraction=0.0,
        attack_scale=1.0,
        privacy_mode="none",
        privacy_target_epsilon=None,
    ),
    ExpandedScenarioSpec(
        "iot_scen_04_flaky_sensor",
        "IoT Intermittent Connectivity (20% participation)",
        "Low availability where only 20% of wearable sensors connect per round under moderate skew.",
        domain_id="iot_edge",
        num_clients=20,
        heterogeneity_alpha=0.5,
        participation_rate=0.2,
        attack_type=None,
        attack_fraction=0.0,
        attack_scale=1.0,
        privacy_mode="none",
        privacy_target_epsilon=None,
    ),
    ExpandedScenarioSpec(
        "iot_scen_05_byzantine_signflip",
        "IoT Byzantine Sign-Flip Attack (20% adversaries)",
        "20% compromised edge nodes injecting sign-flipped sensor gradients (scale=2.0).",
        domain_id="iot_edge",
        num_clients=20,
        heterogeneity_alpha=0.5,
        participation_rate=0.5,
        attack_type="sign_flip",
        attack_fraction=0.2,
        attack_scale=2.0,
        privacy_mode="none",
        privacy_target_epsilon=None,
    ),
    ExpandedScenarioSpec(
        "iot_scen_06_sensor_gaussian_noise",
        "IoT Faulty Sensor Gaussian Noise (20% nodes)",
        "20% failing hardware nodes transmitting zero-mean Gaussian noise updates (std=1.0).",
        domain_id="iot_edge",
        num_clients=20,
        heterogeneity_alpha=0.5,
        participation_rate=0.5,
        attack_type="gaussian_noise",
        attack_fraction=0.2,
        attack_scale=1.0,
        privacy_mode="none",
        privacy_target_epsilon=None,
    ),
    ExpandedScenarioSpec(
        "iot_scen_07_health_telemetry_dp",
        "IoT Biometric Privacy Compliance (Client DP)",
        "Strict biometric privacy requirements enforcing client-level differential privacy.",
        domain_id="iot_edge",
        num_clients=20,
        heterogeneity_alpha=0.5,
        participation_rate=0.5,
        attack_type=None,
        attack_fraction=0.0,
        attack_scale=1.0,
        privacy_mode="client_dp",
        privacy_target_epsilon=8.0,
    ),
]


# ==============================================================================
# Domain-Specific Scenarios for Mobile Apps (50 clients)
# ==============================================================================
MOBILE_SCENARIOS: List[ExpandedScenarioSpec] = [
    ExpandedScenarioSpec(
        "mob_scen_01_iid",
        "Mobile Uniform Device Cohort (IID)",
        "50 consumer smartphones with uniform visual categories and 20% Wi-Fi charging availability.",
        domain_id="mobile_apps",
        num_clients=50,
        heterogeneity_alpha=None,
        participation_rate=0.2,
        attack_type=None,
        attack_fraction=0.0,
        attack_scale=1.0,
        privacy_mode="none",
        privacy_target_epsilon=None,
    ),
    ExpandedScenarioSpec(
        "mob_scen_02_mild_skew",
        "Mobile Realistic User Skew (alpha=0.5)",
        "50 smartphones with realistic non-IID user photography distributions (Dirichlet alpha=0.5).",
        domain_id="mobile_apps",
        num_clients=50,
        heterogeneity_alpha=0.5,
        participation_rate=0.2,
        attack_type=None,
        attack_fraction=0.0,
        attack_scale=1.0,
        privacy_mode="none",
        privacy_target_epsilon=None,
    ),
    ExpandedScenarioSpec(
        "mob_scen_03_severe_skew",
        "Mobile Extreme Topic Skew (alpha=0.1)",
        "50 smartphones with extreme Dirichlet skew (alpha=0.1) under 20% participation.",
        domain_id="mobile_apps",
        num_clients=50,
        heterogeneity_alpha=0.1,
        participation_rate=0.2,
        attack_type=None,
        attack_fraction=0.0,
        attack_scale=1.0,
        privacy_mode="none",
        privacy_target_epsilon=None,
    ),
    ExpandedScenarioSpec(
        "mob_scen_04_flaky_cellular",
        "Mobile Spotty Cellular Uplink (10% participation)",
        "Low availability where only 10% of devices are online per round over cellular data.",
        domain_id="mobile_apps",
        num_clients=50,
        heterogeneity_alpha=0.3,
        participation_rate=0.1,
        attack_type=None,
        attack_fraction=0.0,
        attack_scale=1.0,
        privacy_mode="none",
        privacy_target_epsilon=None,
    ),
    ExpandedScenarioSpec(
        "mob_scen_05_sybil_signflip",
        "Mobile Sybil Sign-Flip Attack (20% botnet)",
        "20% malicious smartphones coordinated in a sign-flip poisoning attack (scale=2.0).",
        domain_id="mobile_apps",
        num_clients=50,
        heterogeneity_alpha=0.3,
        participation_rate=0.2,
        attack_type="sign_flip",
        attack_fraction=0.2,
        attack_scale=2.0,
        privacy_mode="none",
        privacy_target_epsilon=None,
    ),
    ExpandedScenarioSpec(
        "mob_scen_06_scaled_update",
        "Mobile Scaled Update Poisoning (20% adversaries)",
        "20% compromised devices injecting 5x magnified parameter updates to divert global model.",
        domain_id="mobile_apps",
        num_clients=50,
        heterogeneity_alpha=0.3,
        participation_rate=0.2,
        attack_type="scaled_update",
        attack_fraction=0.2,
        attack_scale=5.0,
        privacy_mode="none",
        privacy_target_epsilon=None,
    ),
    ExpandedScenarioSpec(
        "mob_scen_07_consumer_dp",
        "Mobile User Privacy Compliance (Client DP)",
        "Strict consumer data protection enforcing Client-Level DP across mobile rounds.",
        domain_id="mobile_apps",
        num_clients=50,
        heterogeneity_alpha=0.3,
        participation_rate=0.2,
        attack_type=None,
        attack_fraction=0.0,
        attack_scale=1.0,
        privacy_mode="client_dp",
        privacy_target_epsilon=8.0,
    ),
]


def get_expanded_configs(num_clients: int) -> List[ExpandedConfigSpec]:
    """
    Generates 16 expanded configurations covering:
    - SCAFFOLD, FedNova, FedAdam, FedProx, FedAvg
    - Bulyan, CoordinateMedian, Krum, TrimmedMean, FedAvgMean
    - QSGD, ThresholdedTernary (ternary 2-bit packing), Identity
    - DP axis: off, eps 8, eps 4, eps 1
    """
    f_byz = max(1, int(num_clients * 0.2))

    return [
        # 1. Baseline
        ExpandedConfigSpec(
            "exp_01_fedavg",
            "FedAvg (Standard SGD)",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            dp_target_epsilon=None,
            rationale="Standard canonical baseline without defense, compression, or privacy constraints.",
        ),
        # 2. FedProx Regularizer
        ExpandedConfigSpec(
            "exp_02_fedprox",
            "FedProx (Proximal mu=0.01)",
            client_algo="FedProx",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32, "mu": 0.01},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            dp_target_epsilon=None,
            rationale="Restricts client local drift under heterogeneous non-IID data via proximal term.",
        ),
        # 3. SCAFFOLD Control Variates
        ExpandedConfigSpec(
            "exp_03_scaffold",
            "SCAFFOLD (Variance Reduction)",
            client_algo="SCAFFOLD",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            dp_target_epsilon=None,
            rationale="Corrects client drift using control variates across heterogeneous local updates.",
        ),
        # 4. FedNova Normalized Averaging
        ExpandedConfigSpec(
            "exp_04_fednova",
            "FedNova (Normalized Updates)",
            client_algo="FedNova",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            dp_target_epsilon=None,
            rationale="Normalizes client gradient updates by local step counts to eliminate objective inconsistency.",
        ),
        # 5. FedAdam Adaptive Server Optimizer
        ExpandedConfigSpec(
            "exp_05_fedadam",
            "FedAdam (Adaptive Server Moments)",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerAdam",
            server_opt_kwargs={"lr": 1.0, "beta1": 0.9, "beta2": 0.99, "epsilon": 1e-4},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            dp_target_epsilon=None,
            rationale="Applies Adam first and second momentum to global server parameter updates.",
        ),
        # 6. TrimmedMean
        ExpandedConfigSpec(
            "exp_06_trimmed_mean",
            "TrimmedMean (beta=0.2)",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="TrimmedMean",
            aggregation_kwargs={"beta": 0.2},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            dp_target_epsilon=None,
            rationale="Coordinate-wise trimming of extreme 20% updates protects against arbitrary scaling attacks.",
        ),
        # 7. MultiKrum Geometric Filtering
        ExpandedConfigSpec(
            "exp_07_krum",
            "MultiKrum (Distance Clustering)",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="MultiKrum",
            aggregation_kwargs={"f": f_byz, "m": max(1, num_clients - f_byz - 2)},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            dp_target_epsilon=None,
            rationale="Filters out updates that deviate significantly in Euclidean distance space.",
        ),
        # 8. Bulyan Resilient Aggregator
        ExpandedConfigSpec(
            "exp_08_bulyan",
            "Bulyan (Krum + TrimmedMean Combined)",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="Bulyan",
            aggregation_kwargs={"f": f_byz},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            dp_target_epsilon=None,
            rationale="Combines geometric distance pre-selection with coordinate-wise trimmed mean for robust Byzantine defense.",
        ),
        # 9. CoordinateMedian
        ExpandedConfigSpec(
            "exp_09_median",
            "CoordinateMedian (Robust Coordinate Statistics)",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="CoordinateMedian",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="none",
            dp_kwargs={},
            dp_target_epsilon=None,
            rationale="Coordinate-wise median provides high breakdown point against heavy-tailed noise.",
        ),
        # 10. Ternary 2-Bit Packing
        ExpandedConfigSpec(
            "exp_10_ternary_compression",
            "Ternary 2-Bit Packing (16x Compression)",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="ternary_2bit",
            dp_type="none",
            dp_kwargs={},
            dp_target_epsilon=None,
            rationale="Quantizes 32-bit floats into 2-bit ternary states {-1, 0, +1}, cutting uplink bandwidth by 16x.",
        ),
        # 11. QSGD Stochastic Quantization (8-level)
        ExpandedConfigSpec(
            "exp_11_qsgd",
            "QSGD 8-Level Stochastic Quantization",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="qsgd_8",
            dp_type="none",
            dp_kwargs={"s": 8},
            dp_target_epsilon=None,
            rationale="Stochastic 8-level quantization providing unbiased gradient compression with bounded variance.",
        ),
        # 12. DP Axis: eps = 8.0
        ExpandedConfigSpec(
            "exp_12_dp_eps8",
            "Client DP (Target eps = 8.0)",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="client",
            dp_kwargs={"clip_norm": 1.0, "noise_multiplier": 0.4},
            dp_target_epsilon=8.0,
            rationale="Moderate privacy guarantee (eps = 8.0) balancing model utility and data protection.",
        ),
        # 13. DP Axis: eps = 4.0
        ExpandedConfigSpec(
            "exp_13_dp_eps4",
            "Client DP (Target eps = 4.0)",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="client",
            dp_kwargs={"clip_norm": 1.0, "noise_multiplier": 0.8},
            dp_target_epsilon=4.0,
            rationale="Rigorous privacy compliance (eps = 4.0) protecting against membership inference.",
        ),
        # 14. DP Axis: eps = 1.0
        ExpandedConfigSpec(
            "exp_14_dp_eps1",
            "Client DP (Target eps = 1.0)",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="identity",
            dp_type="client",
            dp_kwargs={"clip_norm": 1.0, "noise_multiplier": 1.5},
            dp_target_epsilon=1.0,
            rationale="High-security differential privacy (eps = 1.0) under strong threat environments.",
        ),
        # 15. Robust + Compressed Combined
        ExpandedConfigSpec(
            "exp_15_robust_ternary",
            "TrimmedMean + Ternary 2-Bit Packing",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="TrimmedMean",
            aggregation_kwargs={"beta": 0.2},
            server_optimizer="ServerSGD",
            server_opt_kwargs={"lr": 1.0},
            compressor="ternary_2bit",
            dp_type="none",
            dp_kwargs={},
            dp_target_epsilon=None,
            rationale="Combines 16x uplink compression with coordinate-wise trimming for low-bandwidth adversarial edge networks.",
        ),
        # 16. FedAdam + Compressed Combined
        ExpandedConfigSpec(
            "exp_16_fedadam_ternary",
            "FedAdam + Ternary 2-Bit Packing",
            client_algo="FedAvg",
            client_algo_kwargs={"lr": 0.01, "epochs": 1, "batch_size": 32},
            aggregation_rule="FedAvgMean",
            aggregation_kwargs={},
            server_optimizer="ServerAdam",
            server_opt_kwargs={"lr": 1.0, "beta1": 0.9, "beta2": 0.99, "epsilon": 1e-4},
            compressor="ternary_2bit",
            dp_type="none",
            dp_kwargs={},
            dp_target_epsilon=None,
            rationale="Combines adaptive server momentum with 16x ternary compression for accelerating non-IID edge convergence.",
        ),
    ]
