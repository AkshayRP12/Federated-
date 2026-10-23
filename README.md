# FLaaS: Domain-Aware Federated Learning Configuration Recommender

FLaaS recommends domain-tailored Federated Learning configurations (client algorithm, server aggregation rule, server optimizer, differential privacy mechanism, compression scheme) based on empirical benchmark evidence and real literature citations.

## Hardware Acceleration
- Primary: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.4 via `torch 2.6.0+cu124`)
- Fallback chain: CUDA -> MPS -> CPU (`flaas/fl/device.py`)

## Running Stage 1 Acceptance Tests
```bash
# Run full unit and integration test suite
python -m pytest -v

# Run the standalone FedAvg convergence benchmark
python experiments/stage1_demo.py
```
