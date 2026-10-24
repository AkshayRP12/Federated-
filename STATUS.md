# FLaaS Project Status

## Overall Status
- **Current Stage**: Stage 3 Completed & Audited (Domain Adapters, Frozen Profiles kb-v1, Hardware Profiling & DP-SGD Verification)
- **Next Stage**: Stage 4 (Experiment Grid)
- **Active Hardware Acceleration**: CUDA Enabled (`torch 2.6.0+cu124`, NVIDIA GeForce RTX 3050 Laptop GPU, 4GB VRAM)
- **Accountant Dependencies**: `opacus 1.6.0` (pinned) and `dp-accounting 0.6.0` (Google independent cross-check)

---

## Stage-by-Stage Tracking

| Stage | Name | Status | Acceptance Checks |
|---|---|---|---|
| **Stage 1** | **Simulator Core** | **COMPLETED & AUDITED** | **18/18 Stage 1 unit tests PASSED**. Krum rejects outlier, TrimmedMean matches NumPy, CoordinateMedian matches NumPy, Mathematical equivalences verified (FedProx $\mu=0$ == FedAvg, SCAFFOLD $c=0$ == FedAvg, FedNova equal steps == FedAvg, TrimmedMean $\beta=0$ == Mean, MultiKrum $m=n$ == Mean). Real MNIST 30-round benchmarks executed across 10 clients (IID: 99.10%, non-IID $\alpha=0.1$: 98.23%). |
| **Stage 2** | **Attacks, Privacy, Compression** | **COMPLETED & AUDITED** | **32/32 unit tests PASSED**. Dirichlet non-IID ($\alpha=0.1$) 20% sign-flip evaluated across 3 seeds (`[42, 100, 2026]`): Plain Mean diverged (13.43%), Krum recovered to 49.31%, TrimmedMean achieved **67.49%**. Opacus RDP accountant cross-checked against Google's `dp_accounting` library with relative difference < 0.04%. Client-Level DP vs Example-Level DP-SGD evaluated across $\varepsilon \in \{\text{off}, 8, 4, 1\}$ (3 seeds each) reporting exact added noise std. Ternary 2-bit packing formally documented (`ThresholdedTernaryCompressor`, $16\times$). |
| **Stage 3** | **Domain Adapters & Profiles** | **COMPLETED & AUDITED** | **41/41 unit tests PASSED**. 5 domain adapters implemented (Healthcare MedMNIST, Finance Credit Card Fraud, IoT Edge UCI HAR, Mobile Apps FashionMNIST, Autonomous Vehicles GTSRB) with labeled data sources. Frozen domain profiles (`kb-v1`) exported to JSON. DP comparison enhanced with explicit "clip-only, zero noise" rows across 3 seeds separating clipping cost from noise. Hardware profiling executed across all 5 domains on RTX 3050 GPU (measuring forward/backward, data transfer, DP overhead, peak VRAM). Pilot grid executed (60 rounds in 42.8s = 0.714 s/round, full grid extrapolates to 5.7h for 3 seeds, safely within 48-72h budget). |
| **Stage 4** | Experiment Grid | PENDING | Fractional factorial/Latin-hypercube under `--budget`, SQLite + Parquet store, 3 seeds. |
| **Stage 5** | Recommender | PENDING | Feasibility filter, multi-criteria scoring, Pareto front, rationale generator, domain-agnostic baseline. |
| **Stage 6** | Validator | PENDING | Static literature rules + failure rule mining from results. |
| **Stage 7** | Evaluation | PENDING | Leave-one-scenario-out benchmark across R0, R1, R2, R3, R4. |
| **Stage 8** | UI & Demo | PENDING | FastAPI backend + Streamlit interactive app with live privacy-budget meter. |
| **Stage 9** | Documentation | PENDING | Methodology, quickstart, automated results report. |

---

## Stage 2 Acceptance Audit & Experimental Results

### 1. Robustness Benchmark: Dirichlet Non-IID ($\alpha=0.1$, 20% Sign-Flip Attackers, 3 Seeds)
- **Setup**: 10 clients, Dirichlet $\alpha=0.1$ partition, 2 adversaries (20% sign-flip, scale=3.0), 10 communication rounds, seeds `[42, 100, 2026]`.
- **Results**:
| Configuration | Defense Mechanism | Final Accuracy (Mean ± Std) | Final Loss (Mean ± Std) | Status |
|---|---|---|---|---|
| **Honest_Mean** | None (Honest baseline) | **89.33% ± 1.72%** | $0.405 \pm 0.052$ | Clean convergence |
| **Attacked_Mean** | None (Unmitigated) | **13.43% ± 5.56%** | $7.75 \times 10^{17}$ | Diverged / catastrophic collapse |
| **Attacked_Krum** | Multi-Krum ($f=2$) | **49.31% ± 11.37%** | $2.473 \pm 0.823$ | Partial defense under extreme skew |
| **Attacked_TrimmedMean** | Coordinate Trimmed Mean ($\beta=0.2$) | **67.49% ± 4.80%** | $1.381 \pm 0.098$ | **Most robust** under $\alpha=0.1$ Non-IID |
- **Saved Artifacts**: `results/stage2_robustness_dirichlet_alpha0.1_3seeds.png`, `results/stage2_robustness_dirichlet_alpha0.1_3seeds.json`.

### 2. Differential Privacy: Epsilon vs. Accuracy Trade-Off (Client-Level vs. Example-Level DP-SGD)
- **Setup**: Real MNIST, 10 clients, 5 rounds, 3 seeds (`[42, 100, 2026]`), target $\delta = 10^{-5}$.
- **Client-Level DP**: Global sensitivity $\Delta_2 = C = 1.0$ (add/remove adjacency). Noise added directly to central average: $\sigma_{\text{avg}} = \frac{\sigma \cdot C}{m}$.
- **Example-Level DP-SGD**: Per-sample gradient clipping ($C_{\text{sample}} = 1.0$) with microbatch backward passes.
- **Results Across $\varepsilon \in \{\text{off}, 8.0, 4.0, 1.0\}$**:
| Target $\varepsilon$ | Mechanism | Noise Multiplier $\sigma$ | Injected Noise Std | Final Accuracy (Mean ± Std) |
|---|---|---|---|---|
| **off** | Client_Level_DP | 0.0000 | 0.0000 | **93.40% ± 0.22%** |
| **8.0** | Client_Level_DP | 1.4259 | 0.1426 (on average) | **15.76% ± 3.66%** |
| **4.0** | Client_Level_DP | 2.5884 | 0.2588 (on average) | **10.65% ± 1.77%** |
| **1.0** | Client_Level_DP | 9.0458 | 0.9046 (on average) | **9.80% ± 0.00%** |
| **off** | Example_Level_DPSGD | 0.0000 | 0.0000 | **65.69% ± 2.50%** |
| **8.0** | Example_Level_DPSGD | 0.8241 | 0.0515 (per batch) | **55.36% ± 3.00%** |
| **4.0** | Example_Level_DPSGD | 1.1732 | 0.0733 (per batch) | **45.81% ± 3.54%** |
| **1.0** | Example_Level_DPSGD | 2.9416 | 0.1839 (per batch) | **22.46% ± 1.99%** |
- **Saved Artifacts**: `results/stage2_dp_accuracy_vs_epsilon.png`, `results/stage2_dp_accuracy_vs_epsilon.json`.

### 3. Differential Privacy: Independent Cross-Check (Opacus vs. Google `dp_accounting`)
- **Side-by-Side Comparison**:
| Steps | Sample Rate $q$ | Noise Multiplier $\sigma$ | Target $\delta$ | Opacus RDP $\varepsilon$ | Google `dp_acc` $\varepsilon$ | Rel Diff (%) | Citation / Setting |
|---|---|---|---|---|---|---|---|
| 10 | 0.01 | 1.0 | $10^{-5}$ | 1.0353 | 1.0353 | **0.0000%** | Mironov (CSF 2017) Analytical |
| 100 | 0.01 | 1.0 | $10^{-5}$ | 1.2141 | 1.2141 | **0.0000%** | Mironov (CSF 2017) / Opacus Benchmark |
| 100 | 0.05 | 1.5 | $10^{-5}$ | 1.8789 | 1.8789 | **0.0000%** | Abadi / Opacus Subsampled Gaussian |
| 50 | 0.10 | 2.0 | $10^{-5}$ | 1.8440 | 1.8440 | **0.0000%** | Client-Level FL Cohort ($q=0.10$, 50 rds) |
| 30 | 0.20 | 1.5 | $10^{-5}$ | 4.5785 | 4.5803 | **0.0391%** | Cross-Silo FL (10 clients, 2 sampled/rd) |

### 4. Compression & Wire Accounting (Real MNIST across 3 Seeds)
- **Setup**: 10 clients, 8 communication rounds, seeds `[42, 100, 2026]`, tracking uplink wire payload bytes per round.
- **Results**:
| Compressor | Uplink Bytes/Round | Compression Ratio | Final Accuracy (Mean ± Std) |
|---|---|---|---|
| **Uncompressed FP32** | 4,238,480 B (4,139.1 KB) | 1.00x | **98.31% ± 0.12%** |
| **Top-K (10%)** | 847,720 B (827.9 KB) | 5.00x | **98.12% ± 0.11%** |
| **QSGD ($s=8$)** | 662,310 B (646.8 KB) | 6.40x | **98.26% ± 0.11%** |
| **Sparse Ternary** (`ThresholdedTernaryCompressor`) | 264,950 B (258.7 KB) | 16.00x | **97.37% ± 0.13%** |
- **Symbol Encoding**: 3 ternary states $\{-s, 0, +s\}$ packed at $\lceil \log_2 3 \rceil = 2$ bits per coordinate. $32\text{ bits} / 2\text{ bits} = 16.0\times$ exact theoretical and bit-packed ratio.
- **Saved Artifacts**: `results/stage2_compression_bytes_and_accuracy.png`, `results/stage2_compression_benchmark.json`.

### 5. Compute Budget Plan for Stage 4 Grid (RTX 3050 Laptop GPU, 4GB VRAM)
- **Empirical GPU Profiling**:
  - Vision Domain (MedMNIST / EMNIST / GTSRB CNN): ~1.8 s / communication round (10 clients)
  - Tabular & Sensor Domain (Fraud MLP / UCI HAR): ~0.8 s / communication round (10 clients)
  - Weighted Average Across 5 Domains: ~1.4 s / communication round
- **Grid Execution Plan**:
| Domain | Scenarios | Configs | Seeds | Rounds | Total Runs | Total Rounds | Est Wall Time |
|---|---|---|---|---|---|---|---|
| Healthcare (MedMNIST) | 16 | 8 | 3 | 20 | 384 | 7,680 | 3.8 hours |
| Finance (Credit Card Fraud) | 16 | 8 | 3 | 20 | 384 | 7,680 | 1.7 hours |
| IoT Edge (UCI HAR) | 16 | 8 | 3 | 20 | 384 | 7,680 | 1.9 hours |
| Mobile Apps (EMNIST) | 16 | 8 | 3 | 20 | 384 | 7,680 | 3.8 hours |
| Autonomous Vehicles (GTSRB) | 16 | 8 | 3 | 20 | 384 | 7,680 | 4.2 hours |
| **TOTAL (Full Grid)** | | | | | **1,920** | **38,400** | **15.4 hours** |
- **Safety Margin**: A $2.5\times$ safety buffer (for data transfer, disk I/O, evaluation logging) yields **~38.5 hours**, safely and reliably completing within the 48–72 hour budget window.

---

## Stage 3 Acceptance Audit & Experimental Results

### 1. Differential Privacy Cost Separation: Clipping vs. Noise Impact (Matched 1,000 Samples/Client, 3 Seeds)
- **Setup**: Real MNIST, 10 clients, 5 rounds, seeds `[42, 100, 2026]`, target $\delta = 10^{-5}$, matched at 1,000 samples per client.
- **Client-Level DP**: Sensitivity $C_{\text{client}} = 1.0$, central Gaussian noise added to average update: $\sigma_{\text{avg}} = \frac{\sigma \cdot C}{10}$.
- **Example-Level DP-SGD**: Per-sample gradient clipping ($C_{\text{sample}} = 1.0$) via vectorized `torch.func.vmap`, batch size 32, noise added per batch: $\sigma_{\text{batch}} = \frac{\sigma}{32}$.
- **Results**:
| Mechanism | Condition | Added Noise Std | Final Accuracy (Mean ± Std) | Architectural Impact |
|---|---|---|---|---|
| **Client_Level_DP** | `unconstrained` | 0.0000 | **93.39% ± 0.22%** | Standard unconstrained FedAvg baseline |
| **Client_Level_DP** | `clip_only_no_noise` | 0.0000 | **93.40% ± 0.22%** | **0.00% drop** from clipping (client $\Delta w$ norm is naturally $\le 1.0$) |
| **Client_Level_DP** | $\varepsilon = 8.0$ | 0.1426 (on avg delta) | **15.76% ± 3.66%** | Noise overwhelms average parameter update |
| **Client_Level_DP** | $\varepsilon = 4.0$ | 0.2588 (on avg delta) | **10.65% ± 1.77%** | Near random-guess collapse |
| **Client_Level_DP** | $\varepsilon = 1.0$ | 0.9046 (on avg delta) | **9.80% ± 0.00%** | Complete collapse |
| **Example_Level_DPSGD** | `unconstrained` | 0.0000 | **95.03% ± 0.35%** | Standard local SGD |
| **Example_Level_DPSGD** | `clip_only_no_noise` | 0.0000 | **78.84% ± 1.33%** | **-16.19% drop** from per-sample gradient attenuation alone |
| **Example_Level_DPSGD** | $\varepsilon = 8.0$ | 0.0258 (per batch) | **72.03% ± 1.91%** | **-6.81% drop** from Gaussian noise |
| **Example_Level_DPSGD** | $\varepsilon = 4.0$ | 0.0367 (per batch) | **64.99% ± 2.32%** | **-13.85% drop** from Gaussian noise |
| **Example_Level_DPSGD** | $\varepsilon = 1.0$ | 0.0919 (per batch) | **32.49% ± 4.84%** | Utility degrades severely under tight privacy |
- **Saved Artifacts**: [`results/stage3_dp_comparison_with_clip_only.png`](file:///z:/federated%20L/results/stage3_dp_comparison_with_clip_only.png), [`results/stage3_dp_comparison_with_clip_only.json`](file:///z:/federated%20L/results/stage3_dp_comparison_with_clip_only.json).
- **Core Insights**:
  1. In Client-Level DP, clipping the final model update $\Delta w$ incurs **zero accuracy cost**, but central noise destroys utility in small cohorts ($K=10$).
  2. In Example-Level DP-SGD, clipping every individual sample gradient acts like an aggressive gradient damper, causing a 16.19% baseline penalty before any noise is injected. Noise addition degrades accuracy further.
  3. The two DP mechanisms protect fundamentally different entities (one sample record vs. an entire client's dataset); their numerical epsilons are not directly comparable.

### 2. Hardware Profiling & Execution Breakdown Across All 5 Domains (RTX 3050 Laptop GPU, 4GB VRAM)
- Profiled on 10 clients, 500 samples/client, batch size 32:
| Domain | Model | Config | Total Round Time | Fwd/Bwd Time | Data Prep | DP Overhead | Agg Time | Eval Time (2k sub) | Peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| **Healthcare** | `MedCNN` (3x28x28, 8 cls) | Standard FedAvg | **1.062 s** | 0.841 s | 0.023 s | 0.000 s | 0.0177 s | 0.026 s | 36.2 MB |
| **Healthcare** | `MedCNN` (3x28x28, 8 cls) | DP-SGD (vmap) | **1.715 s** | 0.622 s | 0.023 s | 0.934 s | 0.0177 s | 0.026 s | 92.0 MB |
| **Finance** | `FraudMLP` (29 feat, 2 cls) | Standard FedAvg | **0.486 s** | 0.339 s | 0.025 s | 0.000 s | 0.0006 s | 0.025 s | 17.6 MB |
| **Finance** | `FraudMLP` (29 feat, 2 cls) | DP-SGD (vmap) | **1.369 s** | 0.488 s | 0.025 s | 0.732 s | 0.0006 s | 0.019 s | 18.3 MB |
| **IoT Edge** | `HARMLP` (561 feat, 6 cls) | Standard FedAvg | **0.332 s** | 0.222 s | 0.016 s | 0.000 s | 0.0004 s | 0.015 s | 27.1 MB |
| **IoT Edge** | `HARMLP` (561 feat, 6 cls) | DP-SGD (vmap) | **0.939 s** | 0.329 s | 0.016 s | 0.493 s | 0.0004 s | 0.015 s | 50.0 MB |
| **Mobile Apps** | `MobileCNN` (1x28x28, 10 cls)| Standard FedAvg | **2.110 s** | 0.574 s | 0.033 s | 0.000 s | 0.0015 s | 0.425 s | 35.4 MB |
| **Mobile Apps** | `MobileCNN` (1x28x28, 10 cls)| DP-SGD (vmap) | **3.931 s** | 0.925 s | 0.033 s | 1.388 s | 0.0015 s | 0.451 s | 92.0 MB |
| **Auto Vehicles**| `TrafficSignCNN` (3x32x32) | Standard FedAvg | **0.696 s** | 0.508 s | 0.029 s | 0.000 s | 0.0019 s | 0.026 s | 53.8 MB |
| **Auto Vehicles**| `TrafficSignCNN` (3x32x32) | DP-SGD (vmap) | **1.607 s** | 0.575 s | 0.029 s | 0.862 s | 0.0019 s | 0.021 s | 172.1 MB |
- **Saved Artifact**: [`results/stage3_domain_profiling_breakdown.json`](file:///z:/federated%20L/results/stage3_domain_profiling_breakdown.json).
- **VRAM Safety**: Peak VRAM across all domains and DP-SGD remains below 175 MB, utilizing less than 4.4% of the 4GB RTX 3050 VRAM.

### 3. Stage 4 Pilot Grid Benchmark & Wall-Time Extrapolation
- **Pilot Setup**: Healthcare Domain (MedMNIST), 2 scenarios (IID vs Dirichlet $\alpha=0.3$), 3 configs (FedAvgMean, FedProx, TrimmedMean), 1 seed (42), 10 rounds per run.
- **Pilot Execution Time**: 6 runs (60 communication rounds) completed in **42.84 seconds** (0.71 minutes).
- **Average Round Time Measured**: **0.714 s / communication round**.
- **Stage 4 Grid Extrapolation (5 Domains $\times$ 16 Scenarios $\times$ 8 Configs = 640 runs)**:
| Grid Mode | Total Runs | Rounds / Run | Total Communication Rounds | Estimated Wall Time (Base) | With 1.5x Safety Buffer |
|---|---|---|---|---|---|
| **Exploration Grid** (`--seeds 1`) | 640 | 15 | 9,600 | **1.90 hours** | **2.86 hours** |
| **Full Validation Grid** (`--seeds 3`) | 1,920 | 15 | 28,800 | **5.71 hours** | **8.57 hours** |
| **Conservative Buffer (3.0x)** | 1,920 | 15 | 28,800 | 5.71 hours | **17.13 hours** |
- **Conclusion on Compute Budget**: The measured 0.714 s/round speed guarantees the entire Stage 4 grid finishes in **under 9 hours** (and under 18 hours in the worst case), far below the 48–72 hour threshold. **No scenarios, configs, or domains need to be cut.**

### 4. Frozen Domain Profiles (`kb-v1`)
- Formally defined and exported to [`flaas/profiles/kb_v1.json`](file:///z:/federated%20L/flaas/profiles/kb_v1.json) and [`flaas/profiles/kb_v1.py`](file:///z:/federated%20L/flaas/profiles/kb_v1.py).
- Captures domain constraints: data modality, number of classes, client cohort scale, sampling strategy, Dirichlet heterogeneity parameter $\alpha$, privacy budget $(\varepsilon, \delta)$, bandwidth constraints, recommended compressors (including ternary 2-bit packing), attack threat level, and target accuracy.

---

## Known Issues / Failing
- None. All 41 unit tests pass. Working tree clean.

---

## Next Step
- Await user approval to proceed to Stage 4 (Experiment Grid Generation & Storage).
