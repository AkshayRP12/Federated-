"""
Frozen Domain Profiles (kb-v1) for Federated Learning Configuration Recommender.
Encapsulates domain constraints, privacy requirements, communication bandwidth,
heterogeneity levels, and candidate FL configurations.
"""

from typing import Dict, Any, List
import json
import os
from pydantic import BaseModel, Field


class DomainProfile(BaseModel):
    domain_id: str
    name: str
    description: str
    data_modality: str  # vision, tabular, sensor
    num_classes: int
    typical_num_clients: int
    participation_rate: float
    heterogeneity_alpha: float  # Dirichlet alpha
    privacy_budget_epsilon: float  # Target epsilon
    privacy_delta: float
    bandwidth_constraint: str  # unconstrained, moderate, severe
    recommended_compressor: str
    attack_risk: str  # low, medium, high
    primary_metric: str  # accuracy, macro_f1, auprc
    candidate_algorithms: List[str]
    candidate_aggregators: List[str]
    target_accuracy: float


KB_V1_PROFILES: Dict[str, DomainProfile] = {
    "healthcare": DomainProfile(
        domain_id="healthcare",
        name="Healthcare Multi-Hospital Silos (MedMNIST)",
        description="Cross-silo collaborative diagnostics across hospital consortiums. Strict privacy compliance required.",
        data_modality="vision",
        num_classes=8,
        typical_num_clients=10,
        participation_rate=1.0,
        heterogeneity_alpha=0.3,
        privacy_budget_epsilon=4.0,
        privacy_delta=1e-5,
        bandwidth_constraint="moderate",
        recommended_compressor="identity",
        attack_risk="low",
        primary_metric="accuracy",
        candidate_algorithms=["FedAvg", "FedProx", "SCAFFOLD"],
        candidate_aggregators=["FedAvgMean", "TrimmedMean"],
        target_accuracy=85.0,
    ),
    "finance": DomainProfile(
        domain_id="finance",
        name="Financial Fraud Detection (Credit Card)",
        description="Cross-bank collaborative fraud scoring with severe class imbalance and regulatory privacy requirements.",
        data_modality="tabular",
        num_classes=2,
        typical_num_clients=10,
        participation_rate=1.0,
        heterogeneity_alpha=0.5,
        privacy_budget_epsilon=8.0,
        privacy_delta=1e-5,
        bandwidth_constraint="moderate",
        recommended_compressor="identity",
        attack_risk="medium",
        primary_metric="macro_f1",
        candidate_algorithms=["FedAvg", "FedProx"],
        candidate_aggregators=["FedAvgMean", "TrimmedMean"],
        target_accuracy=80.0,
    ),
    "iot_edge": DomainProfile(
        domain_id="iot_edge",
        name="IoT Wearable Sensors (UCI HAR)",
        description="Battery-constrained wearable sensor telemetry. Severe uplink communication limits.",
        data_modality="sensor",
        num_classes=6,
        typical_num_clients=20,
        participation_rate=0.5,
        heterogeneity_alpha=0.5,
        privacy_budget_epsilon=8.0,
        privacy_delta=1e-5,
        bandwidth_constraint="severe",
        recommended_compressor="ternary_2bit",
        attack_risk="low",
        primary_metric="accuracy",
        candidate_algorithms=["FedAvg", "FedNova"],
        candidate_aggregators=["FedAvgMean"],
        target_accuracy=88.0,
    ),
    "mobile_apps": DomainProfile(
        domain_id="mobile_apps",
        name="Mobile Edge Visual Categorization (Fashion/EMNIST)",
        description="Cross-device consumer smartphone federation. Unreliable WiFi/cellular uplink and partial client participation.",
        data_modality="vision",
        num_classes=10,
        typical_num_clients=50,
        participation_rate=0.2,
        heterogeneity_alpha=0.2,
        privacy_budget_epsilon=8.0,
        privacy_delta=1e-5,
        bandwidth_constraint="severe",
        recommended_compressor="ternary_2bit",
        attack_risk="medium",
        primary_metric="accuracy",
        candidate_algorithms=["FedAvg", "FedProx"],
        candidate_aggregators=["FedAvgMean", "TrimmedMean"],
        target_accuracy=85.0,
    ),
    "autonomous_vehicles": DomainProfile(
        domain_id="autonomous_vehicles",
        name="Autonomous Vehicles Traffic Sign Fleet (GTSRB)",
        description="Connected vehicle edge cameras exposed to adversarial sign-flip and sensor noise attacks.",
        data_modality="vision",
        num_classes=43,
        typical_num_clients=10,
        participation_rate=1.0,
        heterogeneity_alpha=0.1,
        privacy_budget_epsilon=8.0,
        privacy_delta=1e-5,
        bandwidth_constraint="moderate",
        recommended_compressor="qsgd_8",
        attack_risk="high",
        primary_metric="accuracy",
        candidate_algorithms=["FedAvg", "FedProx"],
        candidate_aggregators=["TrimmedMean", "Krum", "Bulyan"],
        target_accuracy=82.0,
    ),
}


def get_profile(domain: str) -> DomainProfile:
    """Returns frozen profile for domain."""
    if domain not in KB_V1_PROFILES:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(KB_V1_PROFILES.keys())}")
    return KB_V1_PROFILES[domain]


def export_kb_v1_json(filepath: str = "flaas/profiles/kb_v1.json") -> None:
    """Exports kb-v1 profiles to JSON artifact."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {k: v.model_dump() for k, v in KB_V1_PROFILES.items()}
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    export_kb_v1_json()
    print("Exported kb-v1 profiles to flaas/profiles/kb_v1.json")
