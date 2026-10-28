"""
FLaaS Configuration Recommender - Interactive Demo (Stage 8)
Supports Mobile Applications and IoT Edge domains.
Recommendations come from experiments on real MNIST and UCI HAR subsets, 1 seed, 2 domains.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st

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
        "target_acc": 88.0,
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
            "early_stopped": bool(filtered_df.iloc[i]["early_stopped"]),
            "epsilon": float(filtered_df.iloc[i]["epsilon"]),
            "is_pareto": is_pareto,
        })

    # Sort by composite score descending, then tie-break by:
    # 1. Lower wire_mb (better bandwidth), 2. Lower time_sec, 3. Config ID alphabetically
    candidates.sort(key=lambda x: (-x["score"], x["wire_mb"], x["time_sec"], x["config_id"]))
    return candidates


def generate_empirical_rationale(rec, baseline, scenario_name, preset, rank=1, tied_with_next=False):
    """Builds a plain-language explanation citing exact experiment numbers."""
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
            "Pareto optimal: no other tested configuration achieves higher accuracy "
            "with lower bandwidth and lower training time on this benchmark."
        )

    if rec["epsilon"] > 0:
        sentences.append(f"Differential privacy budget: eps = {rec['epsilon']:.2f} (delta = 1e-5).")

    return " ".join(sentences)


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main():
    # 1. Title
    st.title("FLaaS Configuration Recommender")
    st.caption(
        "Recommendations come from experiments on real MNIST and UCI HAR subsets, 1 seed, 2 domains."
    )

    # Load results store
    df, source_label = load_results_store()
    if df.empty:
        st.error(
            "No experiment runs found in results database. "
            "Please ensure grid runner has written to results/stage4_grid.sqlite."
        )
        st.stop()

    # Sidebar: Domain Status & System Information
    st.sidebar.markdown("### Active Domain Selection")
    domain_choice = st.sidebar.selectbox(
        "Choose Domain",
        options=["mobile_apps", "iot_edge"],
        format_func=lambda d: DOMAIN_INFO[d]["title"],
    )

    dom = DOMAIN_INFO[domain_choice]
    st.sidebar.markdown(f"**Domain Goal**: {dom['description']}")
    st.sidebar.markdown(f"**Dataset**: `{dom['dataset_name']}`")
    st.sidebar.markdown(f"**Dataset Source**: `{dom['dataset_type']}`")
    st.sidebar.markdown(f"**Cohort Size**: `{dom['num_clients']} devices`")
    st.sidebar.markdown(f"**Neural Architecture**: `{dom['model']}`")

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

    # =========================================================================
    # LEFT COLUMN: STEP 1 - DEFINE EDGE ENVIRONMENT
    # =========================================================================
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
            st.caption("What percentage of edge devices are online, plugged-in, and train each round?")
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

            submitted = st.form_submit_button(
                "Get Recommendation", use_container_width=True
            )

    # =========================================================================
    # RIGHT COLUMN: STEP 2 - RECOMMENDATION & PERFORMANCE EVIDENCE
    # =========================================================================
    with col_results:
        domain_df = df[df["domain"] == domain_choice]
        if domain_df.empty:
            st.warning(f"No experimental data found for `{domain_choice}` in the results store.")
            return

        # Scenario Matching in normalized 5D space
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

        # Slice empirical benchmark results
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

        if not ranked_candidates:
            st.info(
                "Strict filtering yielded 0 candidates; relaxing constraints "
                "to display all empirical runs for this scenario."
            )
            ranked_candidates = score_and_rank_candidates(
                sub_df=scen_df,
                preset=preset,
                require_byz=False,
                require_dp=False,
                require_comp=False,
            )

        top_3 = ranked_candidates[:3]
        rec_1 = top_3[0]

        # Baseline FedAvg for direct head-to-head comparison
        baseline_match = [
            c for c in ranked_candidates
            if "fedavg" in c["config_id"].lower() or "baseline" in c["config_id"].lower()
        ]
        baseline = baseline_match[0] if baseline_match else None

        # -----------------------------------------------------------------
        # Matched Scenario Context
        # -----------------------------------------------------------------
        with st.container(border=True):
            st.markdown("**Matched Benchmark Scenario**")
            st.markdown(f"**{matched_human_name}**")
            st.markdown(
                f"Nearest empirical grid execution. "
                f"Identifier: `{matched_scenario_id}` | "
                f"Feature distance: `{match_dist:.3f}`"
            )

        # -----------------------------------------------------------------
        # RECOMMENDATION CARD
        # -----------------------------------------------------------------
        st.header("Step 2: Recommended Configuration")

        with st.container(border=True):
            st.subheader(rec_1["name"])
            st.markdown(
                f"Configuration ID: `{rec_1['config_id']}` | "
                f"Composite score: **{rec_1['score']:.3f}**"
            )

            tags = []
            if rec_1["is_pareto"]:
                tags.append("Pareto optimal")
            if rec_1["epsilon"] > 0:
                tags.append(f"eps = {rec_1['epsilon']:.2f} DP")
            if rec_1["early_stopped"]:
                tags.append(f"Early stopped at round {rec_1['rounds']}")
            else:
                tags.append(f"{rec_1['rounds']} rounds (max reached)")
            if tags:
                st.markdown(" | ".join(tags))

            # Metrics: 4 columns using st.metric
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                if baseline:
                    acc_delta = rec_1["accuracy"] - baseline["accuracy"]
                    st.metric(
                        label="Model Accuracy",
                        value=f"{rec_1['accuracy']:.1f}%",
                        delta=f"{acc_delta:+.1f}% vs FedAvg",
                        delta_color="normal",
                    )
                else:
                    st.metric(label="Model Accuracy", value=f"{rec_1['accuracy']:.1f}%")

            with m2:
                if baseline:
                    wire_delta = rec_1["wire_mb"] - baseline["wire_mb"]
                    st.metric(
                        label="Uplink Traffic (MB)",
                        value=f"{rec_1['wire_mb']:.1f}",
                        delta=f"{wire_delta:+.1f} MB vs FedAvg",
                        delta_color="inverse",
                    )
                else:
                    st.metric(label="Uplink Traffic (MB)", value=f"{rec_1['wire_mb']:.1f}")

            with m3:
                if baseline:
                    time_delta = rec_1["time_sec"] - baseline["time_sec"]
                    st.metric(
                        label="Wall-Clock Time (s)",
                        value=f"{rec_1['time_sec']:.1f}",
                        delta=f"{time_delta:+.1f} s vs FedAvg",
                        delta_color="inverse",
                    )
                else:
                    st.metric(label="Wall-Clock Time (s)", value=f"{rec_1['time_sec']:.1f}")

            with m4:
                st.metric(label="Rounds Completed", value=rec_1["rounds"])

        # -----------------------------------------------------------------
        # Rationale
        # -----------------------------------------------------------------
        # Detect tie between #1 and #2
        tied_1_2 = len(top_3) >= 2 and abs(rec_1["score"] - top_3[1]["score"]) < 1e-3

        rationale_text = generate_empirical_rationale(
            rec_1, baseline, matched_human_name, preset,
            rank=1, tied_with_next=tied_1_2,
        )
        with st.expander("Why this configuration?", expanded=True):
            st.markdown(rationale_text)

        # -----------------------------------------------------------------
        # Head-to-Head Comparison vs Default FedAvg (st.metric)
        # -----------------------------------------------------------------
        st.markdown("---")
        st.header("Comparison vs Default FedAvg")
        st.caption(
            "Direct empirical performance measured under the same scenario conditions "
            "(1 seed, Seed 42)."
        )

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
                    label="Wall-Clock Time (s)",
                    value=f"{rec_1['time_sec']:.1f} s",
                    delta=f"{time_diff:+.1f} s",
                    delta_color="inverse",
                )
        else:
            st.info("Default FedAvg configuration not present in this scenario slice.")

        # -----------------------------------------------------------------
        # TOP 3 CANDIDATE ALTERNATIVES TABLE
        # -----------------------------------------------------------------
        st.markdown("---")
        st.header("Top 3 Evaluated Alternatives")
        st.caption("Ranked by composite score with tie-break on lower bandwidth, then lower runtime, then config ID:")

        top3_rows = []
        for rank_idx, c in enumerate(top_3, 1):
            # Detect tie with next candidate
            tie_note = ""
            if rank_idx < len(top_3):
                next_c = top_3[rank_idx]  # 0-indexed: rank_idx is already offset by 1
                if abs(c["score"] - next_c["score"]) < 1e-3:
                    tie_note = " (tied, broken by bandwidth)"
            top3_rows.append({
                "Rank": f"#{rank_idx}{tie_note}",
                "Configuration": c["name"],
                "Composite Score": f"{c['score']:.3f}",
                "Accuracy (%)": f"{c['accuracy']:.1f}",
                "Uplink (MB)": f"{c['wire_mb']:.1f}",
                "Runtime (s)": f"{c['time_sec']:.1f}",
                "Rounds": c["rounds"],
                "Pareto Optimal": "Yes" if c["is_pareto"] else "No",
            })
        st.dataframe(pd.DataFrame(top3_rows), width="stretch", hide_index=True)

        # -----------------------------------------------------------------
        # FL GLOSSARY (COLLAPSIBLE)
        # -----------------------------------------------------------------
        with st.expander("Plain-Language Guide: What do these algorithms and terms mean?"):
            st.markdown(
                "- **FedAvg**: The standard baseline. Clients train locally on their data "
                "and the server simply averages their weights. Fails when data is non-IID or under attack.\n"
                "- **FedProx**: Adds a proximal regularizer to local loss to prevent clients from "
                "drifting too far from the global model under heterogeneous data distributions.\n"
                "- **SCAFFOLD**: Uses control variates (drift estimators) on clients and server to "
                "correct for client drift under non-IID data distributions.\n"
                "- **FedNova**: Normalizes gradient contributions by each client's local training steps, "
                "correcting for objective inconsistency.\n"
                "- **FedAdam**: Uses adaptive server-side moments (Adam optimizer on the server) for "
                "more stable convergence.\n"
                "- **TrimmedMean**: A Byzantine defense that sorts coordinate updates and trims the "
                "lowest and highest values, filtering out extreme poisoning attacks.\n"
                "- **Bulyan / CoordinateMedian**: Byzantine aggregators providing resilience even when "
                "up to 20% of participating nodes are malicious adversaries.\n"
                "- **Ternary 2-Bit Packing**: Quantizes continuous 32-bit floating point weights into 3 "
                "states {-1, 0, +1} packed at 2 bits per coordinate, cutting network traffic by 16.0x.\n"
                "- **QSGD**: Quantized Stochastic Gradient Descent with 8 quantization levels.\n"
                "- **Client Differential Privacy (eps)**: Injects calibrated noise into the global "
                "aggregated model using Renyi DP, guaranteeing that an adversary cannot determine "
                "whether any individual client participated in training."
            )

    # -------------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "Recommendations come from experiments on real MNIST and UCI HAR subsets, 1 seed, 2 domains."
    )


if __name__ == "__main__":
    main()
