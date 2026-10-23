# FLaaS Project Status

## Overall Status
- **Current Stage**: Stage 1 Validated on Real MNIST (Simulator Core & Mathematical Equivalences)
- **Next Stage**: Stage 2 (Attacks, Privacy, Compression)
- **Active Hardware Acceleration**: CUDA Enabled (`torch 2.6.0+cu124`, NVIDIA GeForce RTX 3050 Laptop GPU, 4GB VRAM)
- **Accountant Dependency**: `opacus 1.6.0` (pinned and verified for RDP accounting)

---

## Stage-by-Stage Tracking

| Stage | Name | Status | Acceptance Checks |
|---|---|---|---|
| **Stage 1** | **Simulator Core** | **COMPLETED & AUDITED** | **15/15 unit tests PASSED**. Krum rejects outlier, TrimmedMean matches NumPy, CoordinateMedian matches NumPy, Mathematical equivalences verified (FedProx mu=0 == FedAvg, SCAFFOLD c=0 == FedAvg, FedNova equal steps == FedAvg). Real MNIST 30-round benchmarks executed across 10 clients: IID reaches 99.10%, Dirichlet alpha=0.1 reaches 98.23% with slower and noisier trajectory. |
| **Stage 2** | Attacks, Privacy, Compression | PENDING | To implement: label_flip, sign_flip, gaussian_noise, scaled_update; clip_noise, Opacus RDP wrapper, secure_agg; topk, qsgd, sparse_ternary. |
| **Stage 3** | Domain Adapters & Profiles | PENDING | MedMNIST, Credit Card Fraud (real/synthetic fallback), UCI HAR, EMNIST/FashionMNIST, GTSRB. Freeze profiles (`kb-v1`). |
| **Stage 4** | Experiment Grid | PENDING | Fractional factorial/Latin-hypercube under `--budget`, SQLite + Parquet store, 3 seeds. |
| **Stage 5** | Recommender | PENDING | Feasibility filter, multi-criteria scoring, Pareto front, rationale generator, domain-agnostic baseline. |
| **Stage 6** | Validator | PENDING | Static literature rules + failure rule mining from results. |
| **Stage 7** | Evaluation | PENDING | Leave-one-scenario-out benchmark across R0, R1, R2, R3, R4. |
| **Stage 8** | UI & Demo | PENDING | FastAPI backend + Streamlit interactive app with live privacy-budget meter. |
| **Stage 9** | Documentation | PENDING | Methodology, quickstart, automated results report. |

---

## Stage 1 Audit & Equivalence Verification
1. **Dataset Transparency**:
   - Initial demo used synthetic data with template hotspots, causing instant 100% accuracy.
   - Replaced with **Official torchvision MNIST** (60,000 train images, 10,000 disjoint test images).
   - Saved full 30-round run traces with `dataset_source: "real"`.
2. **Mathematical Equivalences** (`tests/test_stage1_equivalences.py`):
   - `test_fedprox_mu_zero_equals_fedavg`: Verified $\mu=0$ matches FedAvg exactly ($\Delta_{\text{diff}} < 10^{-6}$).
   - `test_scaffold_zero_control_variates_equals_fedavg`: Verified round 1 with $c=0$ matches FedAvg exactly ($\Delta_{\text{diff}} < 10^{-6}$).
   - `test_fednova_equal_steps_equals_fedavg`: Verified FedNova with equal $\tau_i$ scales back to FedAvg exactly ($\Delta_{\text{diff}} < 10^{-6}$).
3. **Real MNIST 30-Round Benchmarks** (`experiments/run_real_mnist_benchmark.py`):
   - **IID (10 clients)**: Smooth convergence from 12.29% (untrained init) -> 95.51% (R1) -> 98.87% (R10) -> **99.10%** (R30, loss: 0.0263).
   - **Non-IID Dirichlet $\alpha=0.1$ (10 clients)**: Realistic client drift: 12.29% (untrained init) -> 67.45% (R1) -> 95.47% (R10) -> **98.23%** (R30, loss: 0.0525, with observable inter-round fluctuations).

---

## Known Issues / Failing
- None. All 15 tests pass. Working tree clean.

---

## Next Step
- Await user approval to proceed to Stage 2 (Attacks, Privacy, Compression).
