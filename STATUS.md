# FLaaS Project Status

## Overall Status
- **Current Stage**: Stage 2 Completed & Audited (Attacks, Differential Privacy, Compression & Wire Measurement)
- **Next Stage**: Stage 3 (Domain Adapters & Profiles)
- **Active Hardware Acceleration**: CUDA Enabled (`torch 2.6.0+cu124`, NVIDIA GeForce RTX 3050 Laptop GPU, 4GB VRAM)
- **Accountant Dependency**: `opacus 1.6.0` (pinned and verified for RDP accounting)

---

## Stage-by-Stage Tracking

| Stage | Name | Status | Acceptance Checks |
|---|---|---|---|
| **Stage 1** | **Simulator Core** | **COMPLETED & AUDITED** | **18/18 Stage 1 unit tests PASSED**. Krum rejects outlier, TrimmedMean matches NumPy, CoordinateMedian matches NumPy, Mathematical equivalences verified (FedProx $\mu=0$ == FedAvg, SCAFFOLD $c=0$ == FedAvg, FedNova equal steps == FedAvg, TrimmedMean $\beta=0$ == Mean, MultiKrum $m=n$ == Mean). Real MNIST 30-round benchmarks executed across 10 clients (IID: 99.10%, non-IID $\alpha=0.1$: 98.23%). |
| **Stage 2** | **Attacks, Privacy, Compression** | **COMPLETED & AUDITED** | **29/29 total unit tests PASSED**. 20% sign-flip attackers evaluated across 3 seeds (`[42, 100, 2026]`): Plain Mean degrades to 94.02% (loss 0.2091), Krum recovers to 97.72%, TrimmedMean recovers to 98.20%. Opacus RDP accountant verified against literature benchmarks with 0.00% relative difference. Actual uplink wire bytes and compression ratios measured across 3 seeds: Uncompressed FP32 (4.14 MB/round, 98.31%), TopK 10% (828 KB/round, 5.0x, 98.12%), QSGD $s=8$ (647 KB/round, 6.4x, 98.26%), Sparse Ternary (259 KB/round, 16.0x, 97.37%). |
| **Stage 3** | Domain Adapters & Profiles | PENDING | MedMNIST, Credit Card Fraud (real/synthetic fallback), UCI HAR, EMNIST/FashionMNIST, GTSRB. Freeze profiles (`kb-v1`). |
| **Stage 4** | Experiment Grid | PENDING | Fractional factorial/Latin-hypercube under `--budget`, SQLite + Parquet store, 3 seeds. |
| **Stage 5** | Recommender | PENDING | Feasibility filter, multi-criteria scoring, Pareto front, rationale generator, domain-agnostic baseline. |
| **Stage 6** | Validator | PENDING | Static literature rules + failure rule mining from results. |
| **Stage 7** | Evaluation | PENDING | Leave-one-scenario-out benchmark across R0, R1, R2, R3, R4. |
| **Stage 8** | UI & Demo | PENDING | FastAPI backend + Streamlit interactive app with live privacy-budget meter. |
| **Stage 9** | Documentation | PENDING | Methodology, quickstart, automated results report. |

---

## Stage 2 Acceptance Audit & Experimental Results

### 1. Robustness Benchmark (20% Sign-Flip Attackers on Real MNIST across 3 Seeds)
- **Setup**: 10 clients, 2 adversaries (20%), 10 communication rounds, seeds `[42, 100, 2026]`, attack scale = 3.0.
- **Honest Baseline (Mean, 0% adv)**: Final Accuracy **98.50% ± 0.17%** | Final Loss: 0.0457 ± 0.0011
- **Attacked Plain Mean (20% sign-flip)**: Degrades to **94.02% ± 0.61%** | Final Loss: 0.2091 ± 0.0228
- **Attacked Krum ($f=2$)**: Recovers to **97.72% ± 0.17%** | Final Loss: 0.0703 ± 0.0033
- **Attacked Trimmed Mean ($\beta=0.2$)**: Recovers to **98.20% ± 0.17%** | Final Loss: 0.0563 ± 0.0023
- **Saved Artifacts**: `results/stage2_robustness_signflip_3seeds.png`, `results/stage2_robustness_signflip_3seeds.json`.

### 2. Differential Privacy RDP Verification Against Literature
- **Accountant**: Opacus verified Rényi Differential Privacy (RDP).
- **Comparison Table**:
| Steps | Sample Rate $q$ | Noise Multiplier $\sigma$ | Target $\delta$ | Computed $\varepsilon$ | Ref $\varepsilon$ | Rel Diff (%) | Citation |
|---|---|---|---|---|---|---|---|
| 10 | 0.01 | 1.0 | $10^{-5}$ | 1.0353 | 1.0353 | **0.00%** | Opacus RDP Subsampled Gaussian (Mironov 2017) |
| 100 | 0.01 | 1.0 | $10^{-5}$ | 1.2141 | 1.2141 | **0.00%** | Opacus RDP Benchmark (Mironov 2017) |
| 100 | 0.05 | 1.5 | $10^{-5}$ | 1.8789 | 1.8789 | **0.00%** | Opacus DP-SGD Tutorials |
| 50 | 0.10 | 2.0 | $10^{-5}$ | 1.8440 | 1.8440 | **0.00%** | Opacus FL DP Grid |

### 3. Compression & Wire Accounting (Real MNIST across 3 Seeds)
- **Setup**: 10 clients, 8 communication rounds, seeds `[42, 100, 2026]`, tracking uplink wire payload bytes per round.
- **Results**:
| Compressor | Uplink Bytes/Round | Compression Ratio | Final Accuracy (Mean ± Std) |
|---|---|---|---|
| **Uncompressed FP32** | 4,238,480 B (4,139.1 KB) | 1.00x | **98.31% ± 0.12%** |
| **Top-K (10%)** | 847,720 B (827.9 KB) | 5.00x | **98.12% ± 0.11%** |
| **QSGD ($s=8$)** | 662,310 B (646.8 KB) | 6.40x | **98.26% ± 0.11%** |
| **Sparse Ternary** | 264,950 B (258.7 KB) | 16.00x | **97.37% ± 0.13%** |
- **Saved Artifacts**: `results/stage2_compression_bytes_and_accuracy.png`, `results/stage2_compression_benchmark.json`.

---

## Known Issues / Failing
- None. All 29 unit tests pass. Working tree clean.

---

## Next Step
- Await user approval to proceed to Stage 3 (Domain Adapters and Profiles).
