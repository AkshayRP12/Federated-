# FLaaS Project Status

## Overall Status
- **Current Stage**: Stage 1 Completed (Simulator Core)
- **Next Stage**: Stage 2 (Attacks, Privacy, Compression)
- **Active Hardware Acceleration**: CUDA Enabled (`torch 2.6.0+cu124`, NVIDIA GeForce RTX 3050 Laptop GPU, 4GB VRAM)
- **Accountant Dependency**: `opacus 1.6.0` (pinned and verified for RDP accounting)

---

## Stage-by-Stage Tracking

| Stage | Name | Status | Acceptance Checks |
|---|---|---|---|
| **Stage 1** | **Simulator Core** | **COMPLETED** | **12/12 unit tests PASSED**. Krum rejects outlier, TrimmedMean matches NumPy, CoordinateMedian matches NumPy, FedAvg converges on synthetic MNIST from 7.67% to 100.00% (Loss 2.3043 -> 0.0035). |
| **Stage 2** | Attacks, Privacy, Compression | PENDING | To implement: label_flip, sign_flip, gaussian_noise, scaled_update; clip_noise, Opacus RDP wrapper, secure_agg; topk, qsgd, sparse_ternary. |
| **Stage 3** | Domain Adapters & Profiles | PENDING | MedMNIST, Credit Card Fraud (real/synthetic fallback), UCI HAR, EMNIST/FashionMNIST, GTSRB. Freeze profiles (`kb-v1`). |
| **Stage 4** | Experiment Grid | PENDING | Fractional factorial/Latin-hypercube under `--budget`, SQLite + Parquet store, 3 seeds. |
| **Stage 5** | Recommender | PENDING | Feasibility filter, multi-criteria scoring, Pareto front, rationale generator, domain-agnostic baseline. |
| **Stage 6** | Validator | PENDING | Static literature rules + failure rule mining from results. |
| **Stage 7** | Evaluation | PENDING | Leave-one-scenario-out benchmark across R0, R1, R2, R3, R4. |
| **Stage 8** | UI & Demo | PENDING | FastAPI backend + Streamlit interactive app with live privacy-budget meter. |
| **Stage 9** | Documentation | PENDING | Methodology, quickstart, automated results report. |

---

## Stage 1 Verification Details
- **Client Algorithms**: `FedAvgClient`, `FedProxClient`, `SCAFFOLDClient`, `FedNovaClient` (`flaas/fl/client_algos.py`).
- **Server Rules**: `FedAvgMean`, `TrimmedMean`, `CoordinateMedian`, `Krum` & `MultiKrum`, `Bulyan` (`flaas/fl/server_rules.py`).
- **Server Optimizers**: `ServerSGD`, `ServerAdam` (`flaas/fl/server_optimizers.py`).
- **Device Support**: `flaas/fl/device.py` provides CUDA -> MPS -> CPU fallback with aggressive VRAM clearance (`clear_device_memory()`) to prevent OOM on 4GB GPUs.
- **Test Suite Results**:
  - `tests/test_stage1_aggregators.py`: 5 passed (Krum outlier rejection, TrimmedMean vs NumPy ref, CoordinateMedian vs NumPy ref, FedAvg weighted/unweighted, Bulyan).
  - `tests/test_stage1_clients.py`: 4 passed (FedAvg, FedProx, SCAFFOLD, FedNova step and delta generation).
  - `tests/test_stage1_optimizers.py`: 2 passed (ServerSGD momentum, ServerAdam moment tracking).
  - `tests/test_stage1_convergence.py`: 1 passed (FedAvg 5-round convergence on synthetic MNIST).
- **Benchmark Run Output**: Saved to `results/stage1_fedavg_curve.json`.

---

## Known Issues / Failing
- None. All Stage 1 unit tests and convergence benchmarks pass with zero failures.

---

## Next Immediate Step
- Await user review of Stage 1 acceptance checks before commencing Stage 2 (Attacks, Privacy, Compression).
