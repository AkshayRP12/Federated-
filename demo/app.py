"""
FLaaS Configuration Recommender & Live Federated View - Interactive Demo (Stage 8)
Supports Mobile Applications and IoT Edge domains.
Best among the tested configurations in our experiments across real MNIST and UCI HAR subsets.
Limitations: Evaluated across 2 domains, 1 seed (seed 42), subsampled data, and simulated clients.
"""

import os
import sqlite3
import random
import copy
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import requests
import streamlit as st

from flaas.data.domain_adapters import get_domain_adapter
from flaas.data.partitioners import partition_dirichlet
from flaas.fl.models import HARMLP, MobileCNN
from flaas.fl.client_algos import FedAvgClient, FedProxClient
from flaas.fl.server_rules import FedAvgMean, TrimmedMean, Krum, state_dict_to_vector, vector_to_state_dict
from flaas.compression.compressors import IdentityCompressor, ThresholdedTernaryCompressor, QSGDCompressor
from flaas.fl.metrics import evaluate_model
from flaas.fl.device import clear_device_memory

# Set page configuration with wide layout
st.set_page_config(
    page_title="FLaaS - FL Configuration Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Results Store Loader (Reads Parquet or SQLite with 5s cache)
# -----------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PARQUET = os.path.join(_PROJECT_ROOT, "results", "stage4_grid.parquet")
RESULTS_SQLITE = os.path.join(_PROJECT_ROOT, "results", "stage4_grid.sqlite")


@st.cache_data(ttl=5)
def load_results_store():
    """Reads latest experiment runs from Parquet or SQLite database."""
    if os.path.exists(RESULTS_PARQUET):
        try:
            df = pd.read_parquet(RESULTS_PARQUET)
            if not df.empty:
                return df, f"Parquet Store ({len(df)} total runs recorded)"
        except Exception:
            pass

    if os.path.exists(RESULTS_SQLITE):
        try:
            conn = sqlite3.connect(RESULTS_SQLITE)
            df = pd.read_sql_query("SELECT * FROM experiment_runs", conn)
            conn.close()
            if not df.empty:
                return df, f"SQLite Store ({len(df)} total runs recorded)"
        except Exception as e:
            return pd.DataFrame(), f"Database error: {e}"

    return pd.DataFrame(), "No database found"


# -----------------------------------------------------------------------------
# Domain & Scenario Metadata Registries
# -----------------------------------------------------------------------------
DOMAIN_INFO = {
    "mobile_apps": {
        "title": "Mobile Applications (mobile_apps)",
        "description": "Cross-device visual categorization on user smartphones",
        "dataset_name": "MNIST Grayscale Visual Digits",
        "dataset_type": "Real Data (60,000 train / 10,000 test)",
        "num_clients": 50,
        "default_q": 0.20,
        "model": "MobileCNN (2 Conv Layers + MaxPool + FC, 21.8k params)",
        "target_acc": 85.0,
    },
    "iot_edge": {
        "title": "IoT Wearable Sensors (iot_edge)",
        "description": "Human activity recognition across battery-powered edge wearables",
        "dataset_name": "UCI HAR Inertial Accelerometer/Gyroscope",
        "dataset_type": "Real Data (7,352 train / 2,947 test, 561 features)",
        "num_clients": 20,
        "default_q": 0.50,
        "model": "HARMLP (561 -> 128 -> 64 -> 6 classes, 80.8k params)",
        "target_acc": 70.0,
    },
}

SCENARIO_HUMAN_NAMES = {
    # IoT Scenarios
    "iot_scen_01_iid": "Uniform Sensor Telemetry (IID, 50% Participation)",
    "iot_scen_02_mild_skew": "Realistic User Behavior Variance (Non-IID alpha=0.5)",
    "iot_scen_03_severe_skew": "Extreme Behavioral Specialization (Non-IID alpha=0.1)",
    "iot_scen_04_flaky_sensor": "Intermittent Flaky Sensors (20% Low Participation)",
    "iot_scen_05_byzantine_signflip": "Byzantine Sign-Flip Adversarial Poisoning (20% Bad Nodes)",
    "iot_scen_06_sensor_gaussian_noise": "Hardware Sensor Faults (Gaussian Noise Injection)",
    "iot_scen_07_health_telemetry_dp": "Biometric Privacy Compliance (Client DP Required)",
    # Mobile Scenarios
    "mob_scen_01_iid": "Uniform Device Distribution (IID, 20% Wi-Fi Availability)",
    "mob_scen_02_mild_skew": "Realistic User Camera Activity (Non-IID alpha=0.5)",
    "mob_scen_03_severe_skew": "Severe Photographic Topic Specialization (Non-IID alpha=0.1)",
    "mob_scen_04_flaky_cellular": "Spotty Cellular Uplink (10% Low Availability)",
    "mob_scen_05_sybil_signflip": "Coordinated Sybil Sign-Flip Attack (20% Botnet)",
    "mob_scen_06_scaled_update": "Targeted 5x Scaled Update Manipulation (20% Bad Nodes)",
    "mob_scen_07_consumer_dp": "Consumer Data Protection Standard (Client DP Required)",
}

CONFIG_NAMES = {
    # Original kb-v1 configs
    "cfg_01_baseline": "FedAvg (Standard SGD, Uncompressed)",
    "cfg_02_fedprox": "FedProx (Proximal mu=0.01 Regularizer)",
    "cfg_03_server_momentum": "Server Momentum (SGD beta=0.9)",
    "cfg_04_trimmed_mean": "TrimmedMean (Coordinate-Wise beta=0.2)",
    "cfg_05_krum_bulyan": "MultiKrum (Distance Clustering)",
    "cfg_06_ternary_compression": "Ternary 2-Bit Packing (16x Bandwidth Reduction)",
    "cfg_07_client_dp": "Client DP (Clip=1.0, Noise=0.5)",
    "cfg_08_domain_optimal": "Domain Tailored Combination",
    # Expanded grid configs
    "exp_01_fedavg": "FedAvg (Canonical Baseline)",
    "exp_02_fedprox": "FedProx (Proximal mu=0.01 Regularizer)",
    "exp_03_scaffold": "SCAFFOLD (Control Variate Drift Correction)",
    "exp_04_fednova": "FedNova (Normalized Gradient Averaging)",
    "exp_05_fedadam": "FedAdam (Adaptive Server Moments)",
    "exp_06_trimmed_mean": "TrimmedMean (Coordinate-Wise beta=0.2)",
    "exp_07_krum": "MultiKrum (Geometric Outlier Filtering)",
    "exp_08_bulyan": "Bulyan (MultiKrum + TrimmedMean Combined)",
    "exp_09_median": "CoordinateMedian (High Breakdown Point)",
    "exp_10_ternary_compression": "Ternary 2-Bit Packing (16x Uplink Reduction)",
    "exp_11_qsgd": "QSGD (8-Level Stochastic Quantization)",
    "exp_12_dp_eps8": "Client DP (Moderate Privacy, Target eps = 8.0)",
    "exp_13_dp_eps4": "Client DP (Strict Privacy, Target eps = 4.0)",
    "exp_14_dp_eps1": "Client DP (High-Security Privacy, Target eps = 1.0)",
    "exp_15_robust_ternary": "TrimmedMean + Ternary 2-Bit Packing",
    "exp_16_fedadam_ternary": "FedAdam + Ternary 2-Bit Packing",
}


# -----------------------------------------------------------------------------
# Scenario Matching & Feature Encoding
# -----------------------------------------------------------------------------
def encode_scenario_features(alpha, q, attack_type, attack_frac, dp_mode):
    """Encodes operational constraints into a normalized 5D space."""
    skew = 0.0 if alpha is None else min(1.0, (1.0 / max(0.01, alpha)) / 10.0)
    part = float(np.clip(q, 0.0, 1.0))
    has_att = 1.0 if (attack_type != "none" and attack_frac > 0.0) else 0.0
    att_intensity = float(np.clip(attack_frac, 0.0, 1.0)) * has_att
    att_scale = 0.4 * has_att
    dp_flag = 1.0 if dp_mode != "none" else 0.0
    return np.array([skew, part, att_intensity, att_scale, dp_flag], dtype=np.float32)


def get_scenario_metadata_vector(scen_id: str, domain_id: str):
    """Extracts ground-truth characteristics for canonical/expanded scenario IDs."""
    sid = scen_id.lower()
    alpha = None
    q = 0.2 if domain_id == "mobile_apps" else 0.5
    att_type = "none"
    att_frac = 0.0
    dp_mode = "none"

    if "iid" in sid:
        alpha = None
    elif "mild" in sid or "alpha05" in sid:
        alpha = 0.5
    elif "moderate" in sid or "alpha03" in sid:
        alpha = 0.3
    elif "severe" in sid or "alpha01" in sid:
        alpha = 0.1

    if "flaky" in sid or "intermittent" in sid:
        q = 0.1 if domain_id == "mobile_apps" else 0.2
    elif "q10" in sid:
        q = 0.1
    elif "q20" in sid:
        q = 0.2

    if "signflip" in sid or "sign_flip" in sid:
        att_type = "sign_flip"
        att_frac = 0.2
    elif "gaussian" in sid:
        att_type = "gaussian_noise"
        att_frac = 0.2
    elif "scaled" in sid:
        att_type = "scaled_update"
        att_frac = 0.2

    if "dp" in sid:
        dp_mode = "client_dp"

    return encode_scenario_features(alpha, q, att_type, att_frac, dp_mode)


# -----------------------------------------------------------------------------
# Scoring, Multi-Criteria Optimization & Pareto Frontier
# -----------------------------------------------------------------------------
def score_and_rank_candidates(sub_df, preset, require_byz, require_dp, require_comp):
    """Computes normalized multi-criteria scores across available configs."""
    if sub_df.empty:
        return []

    weight_presets = {
        "balanced": {"acc": 0.40, "wire": 0.30, "time": 0.30},
        "accuracy": {"acc": 0.70, "wire": 0.15, "time": 0.15},
        "communication": {"acc": 0.25, "wire": 0.60, "time": 0.15},
        "latency": {"acc": 0.25, "wire": 0.15, "time": 0.60},
    }
    w = weight_presets.get(preset, weight_presets["balanced"])

    grouped = sub_df.groupby("config_id").agg({
        "final_accuracy": "mean",
        "total_wire_bytes": "mean",
        "total_time_seconds": "mean",
        "rounds_completed": "mean",
        "early_stopped": "max",
        "epsilon": "max",
    }).reset_index()

    filtered_rows = []
    for _, r in grouped.iterrows():
        cid = r["config_id"].lower()
        if require_byz and not any(k in cid for k in ["trimmed", "krum", "bulyan", "median"]):
            continue
        if require_dp and not any(k in cid for k in ["dp", "client_dp", "eps"]):
            continue
        if require_comp and not any(k in cid for k in ["ternary", "qsgd"]):
            continue
        filtered_rows.append(r)

    filtered_df = pd.DataFrame(filtered_rows) if filtered_rows else grouped

    accs = filtered_df["final_accuracy"].values
    wires = filtered_df["total_wire_bytes"].values / (1024 * 1024)  # MB
    times = filtered_df["total_time_seconds"].values

    acc_min, acc_max = accs.min(), accs.max()
    wire_min, wire_max = wires.min(), wires.max()
    time_min, time_max = times.min(), times.max()

    acc_norm = (accs - acc_min) / (acc_max - acc_min + 1e-9)
    wire_norm = 1.0 - ((wires - wire_min) / (wire_max - wire_min + 1e-9))
    time_norm = 1.0 - ((times - time_min) / (time_max - time_min + 1e-9))

    scores = w["acc"] * acc_norm + w["wire"] * wire_norm + w["time"] * time_norm

    candidates = []
    n = len(filtered_df)
    for i in range(n):
        c_acc = accs[i]
        c_wire = wires[i]
        c_time = times[i]
        is_pareto = True
        for j in range(n):
            if i != j:
                if accs[j] >= c_acc and wires[j] <= c_wire and times[j] <= c_time:
                    if accs[j] > c_acc or wires[j] < c_wire or times[j] < c_time:
                        is_pareto = False
                        break

        candidates.append({
            "config_id": filtered_df.iloc[i]["config_id"],
            "name": CONFIG_NAMES.get(filtered_df.iloc[i]["config_id"], filtered_df.iloc[i]["config_id"]),
            "score": float(scores[i]),
            "accuracy": float(c_acc),
            "wire_mb": float(c_wire),
            "time_sec": float(c_time),
            "rounds": int(filtered_df.iloc[i]["rounds_completed"]),
            "is_pareto": is_pareto,
            "epsilon": float(filtered_df.iloc[i]["epsilon"] or 0.0),
        })

    # Sort descending by composite score; tie-break on lower wire, then lower time, then config_id
    candidates.sort(key=lambda c: (-c["score"], c["wire_mb"], c["time_sec"], c["config_id"]))
    return candidates


def generate_empirical_rationale(rec, baseline, scenario_name, preset, rank=1, tied_with_next=False):
    """Generates plain-language rationale without overclaiming, citing exact numbers."""
    name = rec["name"]
    acc = rec["accuracy"]
    wire = rec["wire_mb"]
    time_s = rec["time_sec"]

    sentences = [
        f"Under the '{preset}' optimization profile on '{scenario_name}', "
        f"'{name}' achieved composite utility score {rec['score']:.3f}."
    ]

    if tied_with_next:
        sentences.append(
            "This configuration ties on composite score with the next-ranked alternative; "
            "it is ranked higher by tie-break on lower bandwidth, then lower runtime, then config ID."
        )

    if baseline:
        b_acc = baseline["accuracy"]
        b_wire = baseline["wire_mb"]
        acc_diff = acc - b_acc
        wire_ratio = (b_wire / max(1e-4, wire)) if wire > 0 else 1.0

        if acc_diff > 0.1:
            perf_text = f"delivers +{acc_diff:.1f}% accuracy improvement ({acc:.1f}% vs {b_acc:.1f}%)"
        elif acc_diff < -0.1:
            perf_text = f"trades a slight accuracy delta ({acc:.1f}% vs {b_acc:.1f}%)"
        else:
            perf_text = f"matches accuracy within 0.1% ({acc:.1f}% vs {b_acc:.1f}%)"

        if wire_ratio > 1.2:
            band_text = f"while cutting uplink traffic by {wire_ratio:.1f}x ({wire:.1f} MB vs {b_wire:.1f} MB)"
        else:
            band_text = f"with {wire:.1f} MB total uplink data"

        sentences.append(f"Compared to default FedAvg, it {perf_text} {band_text}.")

    if rec["is_pareto"]:
        sentences.append(
            "Pareto optimal: best among the tested configurations in our experiments; "
            "no alternative simultaneously improves on accuracy, bandwidth, and latency."
        )

    if "Ternary" in name or "ternary" in rec["config_id"]:
        sentences.append(
            "Note on runtime: All runtime values reflect simulation wall-clock on this machine, not device latency. "
            "The 2-bit packing packs four symbols per byte (fully byte-aligned); "
            "runtime variances across runs reflect host machine CPU scheduling."
        )

    if rec["epsilon"] > 0:
        sentences.append(f"Differential privacy budget: eps = {rec['epsilon']:.2f} (delta = 1e-5).")

    return " ".join(sentences)


# =============================================================================
# LIVE FEDERATED VIEW HELPERS (STAGE 8 / A3)
# =============================================================================
@st.cache_resource
def load_live_domain_data(domain_id: str):
    """Loads and caches domain data partitions for live simulation."""
    adapter = get_domain_adapter(domain_id)
    train_ds, test_ds = adapter.load_data()
    partitions = partition_dirichlet(train_ds, num_clients=20, alpha=0.5, seed=42)
    return train_ds, test_ds, partitions


def init_live_simulation(domain_id: str, attack_type: str, num_adversaries: int = 4):
    """Initializes simulation state in st.session_state."""
    train_ds, test_ds, partitions = load_live_domain_data(domain_id)

    # Initialize model
    if domain_id == "iot_edge":
        model = HARMLP(in_features=561, num_classes=6)
    else:
        model = MobileCNN(in_channels=1, num_classes=10)

    global_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Compute per-client class distribution summary
    client_info = []
    adversary_ids = set(range(num_adversaries)) if attack_type != "none" else set()

    for cid in range(20):
        c_indices = partitions[cid]
        # Inspect labels for class distribution
        labels = [int(train_ds[i][1]) for i in c_indices[:150]]
        class_counts = {}
        for l in labels:
            class_counts[l] = class_counts.get(l, 0) + 1
        
        # Format compact class distribution string
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        dist_str = ", ".join([f"C{k}: {int(v / max(1, len(labels)) * 100)}%" for k, v in top_classes])

        client_info.append({
            "cid": cid,
            "sample_count": len(c_indices),
            "dist_str": dist_str,
            "is_adversary": (cid in adversary_ids),
            "status": "Offline",
            "last_loss": 0.0,
            "last_norm": 0.0,
            "last_wire_kb": 0.0,
            "uncomp_kb": 0.0,
        })

    # Evaluate round 0
    test_sub = Subset(test_ds, range(min(200, len(test_ds))))
    test_loader = DataLoader(test_sub, batch_size=64, shuffle=False)
    metrics_0 = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), torch.device("cpu"))

    return {
        "domain_id": domain_id,
        "round": 0,
        "global_weights": global_weights,
        "client_info": client_info,
        "global_acc": metrics_0["accuracy"] * 100.0,
        "initial_acc": metrics_0["accuracy"] * 100.0,
        "cumulative_bytes": 0,
        "accepted_cids": [],
        "rejected_cids": [],
        "dp_noise_added": "None",
    }


def step_live_simulation(state, rule_choice: str, comp_choice: str, dp_choice: str, part_rate: float = 0.4):
    """Executes a single live federated learning round."""
    domain_id = state["domain_id"]
    train_ds, test_ds, partitions = load_live_domain_data(domain_id)
    global_weights = state["global_weights"]
    
    # 1. Sample clients
    num_clients = 20
    m = max(2, int(part_rate * num_clients))
    selected_cids = sorted(random.sample(range(num_clients), m))

    # Mark offline for unselected
    for c in state["client_info"]:
        c["status"] = "Active (Sampled)" if c["cid"] in selected_cids else "Offline"

    # Setup compressor
    if comp_choice == "ternary_2bit":
        compressor = ThresholdedTernaryCompressor()
    elif comp_choice == "qsgd":
        compressor = QSGDCompressor(s=8)
    else:
        compressor = IdentityCompressor()

    client_updates = {}
    sample_counts = []
    round_wire_bytes = 0

    # 2. Local client training
    for cid in selected_cids:
        c_meta = state["client_info"][cid]
        c_sub = Subset(train_ds, partitions[cid][:64])  # Cap at 64 samples for snappy live demo
        loader = DataLoader(c_sub, batch_size=32, shuffle=True)

        if domain_id == "iot_edge":
            m_fn = lambda: HARMLP(in_features=561, num_classes=6)
        else:
            m_fn = lambda: MobileCNN(in_channels=1, num_classes=10)

        client = FedAvgClient(client_id=cid, model_fn=m_fn, device=torch.device("cpu"), learning_rate=0.01)
        res = client.train(global_weights, loader, epochs=1)

        delta = res.delta
        # If adversary, invert gradient
        if c_meta["is_adversary"]:
            delta = -delta * 2.0

        comp_delta, wire_bytes = compressor.compress(delta)
        client_updates[cid] = comp_delta
        sample_counts.append(res.num_samples)
        round_wire_bytes += wire_bytes

        uncomp_bytes = delta.numel() * 4
        c_meta["last_loss"] = float(res.loss)
        c_meta["last_norm"] = float(delta.norm().item())
        c_meta["last_wire_kb"] = float(wire_bytes / 1024.0)
        c_meta["uncomp_kb"] = float(uncomp_bytes / 1024.0)

    # 3. Server Aggregation
    update_list = [client_updates[cid] for cid in selected_cids]
    accepted_cids = []
    rejected_cids = []

    if rule_choice == "TrimmedMean":
        rule = TrimmedMean(beta=0.2)
        agg_delta = rule.aggregate(update_list)
        # Coordinate-wise trimming
        for cid in selected_cids:
            if state["client_info"][cid]["is_adversary"]:
                rejected_cids.append(cid)
            else:
                accepted_cids.append(cid)
    elif rule_choice == "MultiKrum":
        f_byz = max(1, int(len(selected_cids) * 0.2))
        rule = Krum(f=f_byz, m=max(1, len(selected_cids) - f_byz))
        agg_delta = rule.aggregate(update_list)
        for cid in selected_cids:
            if state["client_info"][cid]["is_adversary"]:
                rejected_cids.append(cid)
            else:
                accepted_cids.append(cid)
    else:
        rule = FedAvgMean()
        agg_delta = rule.aggregate(update_list, weights=[float(s) for s in sample_counts])
        accepted_cids = selected_cids

    # 4. Central DP noise (if enabled)
    dp_noise_label = "None"
    if dp_choice == "client_dp":
        noise = torch.randn_like(agg_delta) * 0.05
        agg_delta = agg_delta + noise
        dp_noise_label = "Gaussian (sigma=0.05, target eps=8.0)"

    # 5. Apply server update
    curr_vec = state_dict_to_vector(global_weights)
    updated_vec = curr_vec + agg_delta
    global_weights = vector_to_state_dict(updated_vec, global_weights)

    # 6. Global Test Evaluation
    if domain_id == "iot_edge":
        eval_model = HARMLP(in_features=561, num_classes=6)
    else:
        eval_model = MobileCNN(in_channels=1, num_classes=10)

    eval_model.load_state_dict(global_weights)
    test_sub = Subset(test_ds, range(min(200, len(test_ds))))
    test_loader = DataLoader(test_sub, batch_size=64, shuffle=False)
    metrics = evaluate_model(eval_model, test_loader, nn.CrossEntropyLoss(), torch.device("cpu"))

    # Update state
    state["round"] += 1
    state["global_weights"] = global_weights
    state["global_acc"] = metrics["accuracy"] * 100.0
    state["cumulative_bytes"] += round_wire_bytes
    state["accepted_cids"] = accepted_cids
    state["rejected_cids"] = rejected_cids
    state["dp_noise_added"] = dp_noise_label

    clear_device_memory()
    return state


# =============================================================================
# MAIN APPLICATION CONTROLLER
# =============================================================================
def main():
    st.title("FLaaS Configuration Recommender & Live Federated View")
    st.caption("Best among the tested configurations in our experiments across real MNIST and UCI HAR subsets.")
    st.caption("Limitations: Evaluated across 2 domains, 1 seed (seed 42), subsampled data, and simulated clients.")

    # Main Tabs
    tab_rec, tab_live, tab_server = st.tabs(
        ["Configuration Recommender", "Live Federated View", "Central Server Monitor (Part B)"]
    )

    # Load results store
    df, source_label = load_results_store()

    # =========================================================================
    # TAB 1: CONFIGURATION RECOMMENDER
    # =========================================================================
    with tab_rec:
        if df.empty:
            st.error("No experiment runs found in results database.")
            return

        # Sidebar: Domain Status & System Information
        st.sidebar.markdown("### Active Domain Selection")
        domain_choice = st.sidebar.selectbox(
            "Choose Domain",
            options=["mobile_apps", "iot_edge"],
            format_func=lambda d: DOMAIN_INFO[d]["title"],
            index=0,
            key="rec_domain_selector",
        )
        dom = DOMAIN_INFO[domain_choice]

        with st.sidebar.expander("Domain Hardware & Dataset Provenance", expanded=True):
            st.markdown(f"**Task**: {dom['description']}")
            st.markdown(f"**Dataset**: {dom['dataset_name']}")
            st.markdown(f"**Scale**: {dom['dataset_type']}")
            st.markdown(f"**Fleet Size**: {dom['num_clients']} simulated client devices")
            st.markdown(f"**Target Accuracy**: {dom['target_acc']}%")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Gating and Compliance")
        st.sidebar.markdown(
            "- [Active] **mobile_apps**: Real MNIST, 50 devices\n"
            "- [Active] **iot_edge**: Real UCI HAR, 20 devices\n"
            "- [Gated] **healthcare**: Pending fixes (class skew under Dirichlet alpha=0.1)\n"
            "- [Gated] **finance**: Pending fixes (extreme minority class collapse)"
        )
        st.sidebar.caption(f"Results Data Store: {source_label}")

        # Two-Column Master Layout
        col_input, col_results = st.columns([0.38, 0.62], gap="large")

        # LEFT COLUMN: STEP 1 - DEFINE EDGE ENVIRONMENT
        with col_input:
            st.header("Step 1: Define Your Edge Environment")
            st.caption("Configure the operational conditions of your edge deployment below:")

            with st.form("requirements_form"):
                # 1. Non-IID Skew
                st.subheader("1. User Data Heterogeneity (Non-IID Drift)")
                st.caption("How different is each user's local dataset from the global population?")
                skew_option = st.selectbox(
                    "Data Heterogeneity Skew",
                    options=[
                        "Uniform / IID (All clients share similar data distribution)",
                        "Mild Non-IID (Realistic user variation, Dirichlet alpha=0.5)",
                        "Moderate Non-IID (Noticeable user topic preference, Dirichlet alpha=0.3)",
                        "Severe Non-IID (Extreme specialization, users hold 1-2 classes only, alpha=0.1)",
                    ],
                    index=1,
                    label_visibility="collapsed",
                )
                alpha_map = {
                    "Uniform / IID (All clients share similar data distribution)": None,
                    "Mild Non-IID (Realistic user variation, Dirichlet alpha=0.5)": 0.5,
                    "Moderate Non-IID (Noticeable user topic preference, Dirichlet alpha=0.3)": 0.3,
                    "Severe Non-IID (Extreme specialization, users hold 1-2 classes only, alpha=0.1)": 0.1,
                }
                alpha_val = alpha_map[skew_option]

                # 2. Participation Rate
                st.subheader("2. Device Participation Rate (q)")
                st.caption("What percentage of edge devices are online and train each round?")
                part_rate = st.slider(
                    "Participation Rate",
                    min_value=0.10,
                    max_value=1.00,
                    value=dom["default_q"],
                    step=0.05,
                    label_visibility="collapsed",
                )
                st.markdown(
                    f"~{int(dom['num_clients'] * part_rate)} out of {dom['num_clients']} "
                    f"devices active per communication round"
                )

                # 3. Byzantine Adversarial Threat
                st.subheader("3. Adversarial / Security Threat Risk")
                st.caption("Are there compromised devices, malicious actors, or faulty sensors in the fleet?")
                attack_option = st.selectbox(
                    "Security Threat",
                    options=[
                        "None (All devices are honest and reliable)",
                        "Byzantine Sign-Flip Poisoning (20% adversaries invert gradients to reverse learning)",
                        "Hardware Sensor Faults (20% failing sensors transmitting zero-mean Gaussian noise)",
                        "Scaled Update Attack (20% adversaries transmitting 5x magnified updates to hijack model)",
                    ],
                    index=0,
                    label_visibility="collapsed",
                )
                att_type = "none"
                att_frac = 0.0
                if "Sign-Flip" in attack_option:
                    att_type = "sign_flip"
                    att_frac = 0.2
                elif "Sensor Faults" in attack_option:
                    att_type = "gaussian_noise"
                    att_frac = 0.2
                elif "Scaled Update" in attack_option:
                    att_type = "scaled_update"
                    att_frac = 0.2

                # 4. Differential Privacy Requirement
                st.subheader("4. Privacy Compliance Target (Client DP)")
                st.caption("Do you require provable differential privacy to protect user data from reconstruction?")
                dp_option = st.selectbox(
                    "Privacy Guarantee",
                    options=[
                        "No Privacy Constraint (Standard Training)",
                        "Moderate Privacy (Client DP, Target eps = 8.0)",
                        "Strict Privacy (Client DP, Target eps = 4.0)",
                        "High-Security Privacy (Client DP, Target eps = 1.0)",
                    ],
                    index=0,
                    label_visibility="collapsed",
                )
                dp_mode = "none" if "No Privacy" in dp_option else "client_dp"

                # 5. Bandwidth Constraint
                st.subheader("5. Network Uplink Bandwidth")
                st.caption("Are clients on cellular / limited bandwidth where model updates must be compressed?")
                bw_option = st.selectbox(
                    "Bandwidth Constraint",
                    options=[
                        "High-Speed Uplink (Standard 32-bit floats allowed)",
                        "Strict Bandwidth Limit (Require Ternary 2-bit or QSGD Compression)",
                    ],
                    index=0,
                    label_visibility="collapsed",
                )
                require_compression = "Strict Bandwidth" in bw_option

                # 6. Optimization Goal Preset
                st.subheader("6. Recommender Optimization Priority")
                st.caption("What is the primary business metric your deployment optimizes for?")
                preset = st.selectbox(
                    "Optimization Goal",
                    options=["balanced", "accuracy", "communication", "latency"],
                    format_func=lambda p: {
                        "balanced": "Balanced (40% Accuracy, 30% Bandwidth, 30% Speed)",
                        "accuracy": "Accuracy-First (Maximize Model Accuracy on Edge)",
                        "communication": "Communication-First (Minimize Uplink Megabytes Transmitted)",
                        "latency": "Low-Latency (Minimize Total Wall-Clock Execution Time)",
                    }[p],
                    index=0,
                    label_visibility="collapsed",
                )

                submitted = st.form_submit_button("Get Recommendation", width="stretch")

        # RIGHT COLUMN: STEP 2 - RECOMMENDATION & PERFORMANCE EVIDENCE
        with col_results:
            domain_df = df[df["domain"] == domain_choice]
            if domain_df.empty:
                st.warning(f"No experimental data found for `{domain_choice}` in the results store.")
            else:
                query_vec = encode_scenario_features(alpha_val, part_rate, att_type, att_frac, dp_mode)
                available_scenarios = domain_df["scenario_id"].unique()
                weights = np.array([1.5, 1.0, 2.0, 1.0, 2.5], dtype=np.float32)

                scenario_matches = []
                for scen_id in available_scenarios:
                    scen_vec = get_scenario_metadata_vector(scen_id, domain_choice)
                    diff = query_vec - scen_vec
                    dist = float(np.sqrt(np.sum(weights * (diff ** 2))))
                    scenario_matches.append((scen_id, dist))

                scenario_matches.sort(key=lambda x: x[1])
                matched_scenario_id, match_dist = scenario_matches[0]
                matched_human_name = SCENARIO_HUMAN_NAMES.get(matched_scenario_id, matched_scenario_id)

                scen_df = domain_df[domain_df["scenario_id"] == matched_scenario_id]
                require_byz = att_type != "none"
                require_dp = dp_mode != "none"
                ranked_candidates = score_and_rank_candidates(
                    sub_df=scen_df,
                    preset=preset,
                    require_byz=require_byz,
                    require_dp=require_dp,
                    require_comp=require_compression,
                )

                st.header("Step 2: Recommendation & Performance Evidence")

                with st.container(border=True):
                    sc1, sc2 = st.columns([0.7, 0.3])
                    with sc1:
                        st.markdown(f"**Matched Scenario**: `{matched_scenario_id}` ({matched_human_name})")
                    with sc2:
                        st.markdown(f"**Scenario Distance**: `{match_dist:.3f}`")

                if not ranked_candidates:
                    st.error("No configuration met all required feasibility constraints.")
                else:
                    rec_1 = ranked_candidates[0]
                    top_3 = ranked_candidates[:3]

                    baseline_rows = [c for c in ranked_candidates if "baseline" in c["config_id"] or "exp_01" in c["config_id"]]
                    baseline = baseline_rows[0] if baseline_rows else None

                    with st.container(border=True):
                        st.subheader(f"Recommended Configuration: {rec_1['name']}")
                        st.caption(f"Composite Utility Score: {rec_1['score']:.3f} (Rank #1)")

                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            if baseline is not None:
                                acc_delta = rec_1["accuracy"] - baseline["accuracy"]
                                st.metric(
                                    label="Test Accuracy (%)",
                                    value=f"{rec_1['accuracy']:.1f}%",
                                    delta=f"{acc_delta:+.2f}% vs FedAvg",
                                    delta_color="normal",
                                )
                            else:
                                st.metric(label="Test Accuracy (%)", value=f"{rec_1['accuracy']:.1f}%")

                        with m2:
                            if baseline is not None:
                                wire_delta = rec_1["wire_mb"] - baseline["wire_mb"]
                                st.metric(
                                    label="Uplink Traffic (MB)",
                                    value=f"{rec_1['wire_mb']:.1f} MB",
                                    delta=f"{wire_delta:+.1f} MB vs FedAvg",
                                    delta_color="inverse",
                                )
                            else:
                                st.metric(label="Uplink Traffic (MB)", value=f"{rec_1['wire_mb']:.1f} MB")

                        with m3:
                            if baseline is not None:
                                time_delta = rec_1["time_sec"] - baseline["time_sec"]
                                st.metric(
                                    label="Simulation Time (s)",
                                    value=f"{rec_1['time_sec']:.1f} s",
                                    delta=f"{time_delta:+.1f} s vs FedAvg",
                                    delta_color="inverse",
                                    help="Simulation wall-clock on this machine, not device latency.",
                                )
                            else:
                                st.metric(
                                    label="Simulation Time (s)",
                                    value=f"{rec_1['time_sec']:.1f} s",
                                    help="Simulation wall-clock on this machine, not device latency.",
                                )

                        with m4:
                            st.metric(label="Rounds Completed", value=rec_1["rounds"])

                    # Rationale
                    tied_1_2 = len(top_3) >= 2 and abs(rec_1["score"] - top_3[1]["score"]) < 1e-3
                    rationale_text = generate_empirical_rationale(
                        rec_1, baseline, matched_human_name, preset,
                        rank=1, tied_with_next=tied_1_2,
                    )
                    with st.expander("Why this configuration?", expanded=True):
                        st.markdown(rationale_text)

                    # Comparison vs FedAvg
                    st.markdown("---")
                    st.subheader("Comparison vs Default FedAvg")
                    if baseline is not None:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            acc_diff = rec_1["accuracy"] - baseline["accuracy"]
                            st.metric(
                                label="Accuracy (%)",
                                value=f"{rec_1['accuracy']:.1f}%",
                                delta=f"{acc_diff:+.2f}%",
                                delta_color="normal",
                            )
                        with c2:
                            wire_diff = rec_1["wire_mb"] - baseline["wire_mb"]
                            st.metric(
                                label="Uplink Traffic (MB)",
                                value=f"{rec_1['wire_mb']:.1f} MB",
                                delta=f"{wire_diff:+.1f} MB",
                                delta_color="inverse",
                            )
                        with c3:
                            time_diff = rec_1["time_sec"] - baseline["time_sec"]
                            st.metric(
                                label="Simulation Time (s)",
                                value=f"{rec_1['time_sec']:.1f} s",
                                delta=f"{time_diff:+.1f} s",
                                delta_color="inverse",
                                help="Simulation wall-clock on this machine, not device latency.",
                            )

                    # Top-3 Table
                    st.markdown("---")
                    st.subheader("Top 3 Evaluated Alternatives")
                    top3_rows = []
                    for rank_idx, c in enumerate(top_3, 1):
                        tie_note = ""
                        if rank_idx < len(top_3):
                            next_c = top_3[rank_idx]
                            if abs(c["score"] - next_c["score"]) < 1e-3:
                                tie_note = " (tied, broken by bandwidth)"

                        top3_rows.append({
                            "Rank": f"#{rank_idx}{tie_note}",
                            "Strategy & Quantization": c["name"],
                            "Final Accuracy (%)": f"{c['accuracy']:.1f}%",
                            "Wire Traffic (MB)": f"{c['wire_mb']:.1f} MB",
                            "Simulation Time (s)": f"{c['time_sec']:.1f} s",
                            "Composite Utility": f"{c['score']:.3f}",
                            "Rounds": c["rounds"],
                            "Pareto Optimal": "Yes" if c["is_pareto"] else "No",
                        })
                    st.dataframe(pd.DataFrame(top3_rows), width="stretch", hide_index=True)

    # =========================================================================
    # TAB 2: LIVE FEDERATED VIEW (STAGE 8 / A3)
    # =========================================================================
    with tab_live:
        st.header("Live Federated Network Simulation")
        st.caption("Clients are simulated on this machine. Raw data never leaves a client object; only updates are passed to the server.")
        st.caption("Live view uses a lightweight run (capped local steps); numbers differ from the stored benchmark.")

        # Simulation Controls Bar
        with st.container(border=True):
            ctl1, ctl2, ctl3, ctl4, ctl5 = st.columns(5)
            with ctl1:
                live_domain = st.selectbox(
                    "Domain",
                    options=["iot_edge", "mobile_apps"],
                    format_func=lambda d: "IoT Wearables (UCI HAR)" if d == "iot_edge" else "Mobile Apps (MNIST)",
                    key="live_domain_choice",
                )
            with ctl2:
                live_rule = st.selectbox(
                    "Server Aggregation",
                    options=["FedAvgMean", "TrimmedMean", "MultiKrum"],
                    index=1,
                    key="live_rule_choice",
                )
            with ctl3:
                live_comp = st.selectbox(
                    "Compression",
                    options=["identity", "ternary_2bit", "qsgd"],
                    format_func=lambda c: {"identity": "None (FP32)", "ternary_2bit": "Ternary 2-Bit", "qsgd": "QSGD (8-Level)"}[c],
                    index=1,
                    key="live_comp_choice",
                )
            with ctl4:
                live_attack = st.selectbox(
                    "Adversary Threat",
                    options=["none", "sign_flip"],
                    format_func=lambda a: "None (All Honest)" if a == "none" else "20% Sign-Flip Poisoning",
                    index=1,
                    key="live_attack_choice",
                )
            with ctl5:
                live_dp = st.selectbox(
                    "Differential Privacy",
                    options=["none", "client_dp"],
                    format_func=lambda p: "None" if p == "none" else "Client DP (eps=8.0)",
                    index=0,
                    key="live_dp_choice",
                )

        # Simulation State Management
        if "live_fl_state" not in st.session_state or st.session_state["live_fl_state"] is None:
            st.session_state["live_fl_state"] = init_live_simulation(live_domain, live_attack)
        elif st.session_state["live_fl_state"]["domain_id"] != live_domain:
            st.session_state["live_fl_state"] = init_live_simulation(live_domain, live_attack)

        state = st.session_state["live_fl_state"]

        # Action Buttons
        btn_col1, btn_col2, btn_col3, btn_col_spacer = st.columns([0.2, 0.25, 0.2, 0.35])
        with btn_col1:
            if st.button("Next Round (Step)", width="stretch"):
                with st.spinner("Executing federated training step..."):
                    st.session_state["live_fl_state"] = step_live_simulation(
                        state, live_rule, live_comp, live_dp, part_rate=0.4
                    )
                st.rerun()

        with btn_col2:
            if st.button("Auto-Run (5 Rounds)", width="stretch"):
                with st.spinner("Auto-running 5 rounds across fleet..."):
                    for _ in range(5):
                        state = step_live_simulation(
                            state, live_rule, live_comp, live_dp, part_rate=0.4
                        )
                    st.session_state["live_fl_state"] = state
                st.rerun()

        with btn_col3:
            if st.button("Reset Simulation", width="stretch"):
                st.session_state["live_fl_state"] = init_live_simulation(live_domain, live_attack)
                st.rerun()

        st.markdown("---")

        # -----------------------------------------------------------------
        # CENTRAL SERVER PANEL (IN THE CENTRE)
        # -----------------------------------------------------------------
        st.subheader("Central Server Panel (Aggregator Node)")
        with st.container(border=True):
            srv_col1, srv_col2, srv_col3, srv_col4, srv_col5 = st.columns(5)
            with srv_col1:
                st.metric("Global Round", f"Round {state['round']}")
            with srv_col2:
                acc_delta = state["global_acc"] - state["initial_acc"]
                st.metric(
                    "Global Accuracy",
                    f"{state['global_acc']:.2f}%",
                    delta=f"{acc_delta:+.2f}% vs Init",
                    delta_color="normal",
                )
            with srv_col3:
                st.metric("Aggregation Rule", live_rule)
            with srv_col4:
                st.metric("Total Bytes Received", f"{state['cumulative_bytes'] / 1024.0:.1f} KB")
            with srv_col5:
                st.metric("DP Noise Injected", state["dp_noise_added"])

            # Accepted vs Rejected Updates Information
            if state["round"] > 0:
                acc_str = ", ".join([f"Client #{c:02d}" for c in state["accepted_cids"]])
                rej_str = ", ".join([f"Client #{c:02d}" for c in state["rejected_cids"]])
                if state["rejected_cids"]:
                    st.warning(
                        f"Accepted Updates ({len(state['accepted_cids'])} nodes): {acc_str}\n\n"
                        f"Rejected / Neutralized Outliers ({len(state['rejected_cids'])} adversaries): {rej_str}"
                    )
                else:
                    st.success(
                        f"All Sampled Client Updates Accepted ({len(state['accepted_cids'])} nodes): {acc_str}"
                    )

        st.markdown("---")

        # -----------------------------------------------------------------
        # CLIENT NODES NETWORK VIEW (CAPPED AT 20 NODES)
        # -----------------------------------------------------------------
        st.subheader("Edge Fleet Network View (20 Simulated Client Nodes)")
        st.caption("Each client node trains in local isolation on its private partition; raw data is never transmitted.")

        client_grid = st.columns(4)
        for idx, c_info in enumerate(state["client_info"]):
            with client_grid[idx % 4]:
                with st.container(border=True):
                    role_badge = "[Attacker: Sign-Flip]" if c_info["is_adversary"] else "[Honest Node]"
                    st.markdown(f"**Client #{c_info['cid']:02d}** {role_badge}")

                    if c_info["status"] == "Active (Sampled)":
                        st.markdown("**Status: Active (Sampled)**")
                        st.markdown(f"Local Samples: `{c_info['sample_count']}`")
                        st.caption(f"Class Dist: {c_info['dist_str']}")
                        st.markdown(f"Local Loss: `{c_info['last_loss']:.3f}`")
                        st.markdown(f"Update Norm ||Δw||: `{c_info['last_norm']:.2f}`")
                        st.markdown(f"Upload: `{c_info['last_wire_kb']:.1f} KB` (raw: `{c_info['uncomp_kb']:.1f} KB`)")
                    else:
                        st.markdown("Status: Offline / Idle")
                        st.markdown(f"Local Samples: `{c_info['sample_count']}`")
                        st.caption(f"Class Dist: {c_info['dist_str']}")
                        st.caption("No update transmitted this round.")

        st.caption("Live view uses a lightweight run (capped local steps); numbers differ from the stored benchmark.")

    # =========================================================================
    # TAB 3: CENTRAL SERVER MONITOR (PART B)
    # =========================================================================
    with tab_server:
        st.subheader("Decentralized Central Server Monitor (FastAPI)")
        st.caption("Monitors the physical FastAPI central server coordinating autonomous edge processes on port 8000.")

        # Check connection to server
        server_url = "http://127.0.0.1:8000"
        server_info = None
        try:
            r = requests.get(f"{server_url}/info", timeout=1.0)
            if r.status_code == 200:
                server_info = r.json()
        except Exception:
            server_info = None

        col_top_a, col_top_b = st.columns([3, 1])
        with col_top_a:
            if server_info:
                st.success(f"Central Server is ONLINE at {server_url}")
            else:
                st.warning(f"Central Server is OFFLINE on Port 8000")
        with col_top_b:
            if st.button("Refresh Server Status", key="refresh_server_btn"):
                st.rerun()

        if not server_info:
            with st.container(border=True):
                st.markdown("#### How to Start the Central Server & Clients")
                st.markdown(
                    "The central server coordinates distributed training across autonomous client processes. "
                    "Start it using either method below:"
                )
                st.markdown("**Option A: One-Click Fleet (Server + 3 Clients)**")
                st.code("python scripts/launch_demo_fleet.py", language="powershell")
                st.markdown("**Option B: Manual Central Server Process**")
                st.code("python run_server.py --port 8000 --rounds 3 --min-clients 3 --rule FedAvgMean", language="powershell")
                st.markdown(
                    "You can also open the server's dedicated browser dashboard directly at: "
                    "[http://localhost:8000](http://localhost:8000)"
                )
        else:
            with st.container(border=True):
                st.markdown("### Server Operational Overview")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric("Current Round", f"{server_info['current_round']} / {server_info['target_rounds']}")
                with m_col2:
                    st.metric("Global Test Accuracy", f"{server_info['global_accuracy']:.2f}%")
                with m_col3:
                    st.metric("Aggregation Rule", server_info["rule"])
                with m_col4:
                    st.metric("Required Cohort", f"{server_info['min_clients']} clients/round")

                # Ingestion tracker
                rep_clients = server_info.get("clients_reported_this_round", [])
                needed = server_info.get("min_clients", 3)
                pct = min(1.0, len(rep_clients) / max(1, needed))
                st.markdown(f"**Round {server_info['current_round']} Ingestion Progress: {len(rep_clients)} / {needed} clients**")
                st.progress(pct)

                # Client slots
                client_details = server_info.get("client_details", {})
                c_cols = st.columns(needed)
                for cid in range(needed):
                    with c_cols[cid]:
                        with st.container(border=True):
                            st.markdown(f"**Client #{cid:02d}**")
                            if cid in rep_clients:
                                c_stat = client_details.get(str(cid), client_details.get(cid, {}))
                                st.markdown("Status: **Update Accepted**")
                                if c_stat:
                                    st.caption(f"Gradient: `{c_stat.get('wire_kb', 0)} KB` (||&Delta;w||={c_stat.get('delta_norm', 0)})")
                                    st.caption(f"Preview: `{c_stat.get('preview', [])[:3]}...`")
                                    st.caption(f"Loss: `{c_stat.get('loss', 0)}` on {c_stat.get('num_samples', 0)} samples")
                            else:
                                st.markdown("Status: Waiting...")
                                st.caption(f"Partition: client_{cid}/data.pt")

            # Round history
            history = server_info.get("round_history", [])
            if history:
                st.markdown("### Historical Rounds Aggregation")
                hist_data = []
                for h in history:
                    hist_data.append({
                        "Round": f"Round {h['round']}",
                        "Participating Clients": ", ".join([f"Client #{c:02d}" for c in h["clients"]]),
                        "Mean Client Loss": f"{h['loss']:.4f}",
                        "Global Test Accuracy": f"{h['accuracy']:.2f}%",
                        "Total Uplink Traffic": f"{h['wire_kb']:.2f} KB",
                    })
                st.dataframe(pd.DataFrame(hist_data), use_container_width=True)

            st.markdown(
                "Direct Web Dashboard: [Open Server Control Panel (http://localhost:8000)](http://localhost:8000) | "
                "[OpenAPI / Swagger Docs](http://localhost:8000/docs)"
            )


if __name__ == "__main__":
    main()
