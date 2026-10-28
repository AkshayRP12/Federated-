# FLaaS: Federated Learning as a Service & Configuration Recommender

FLaaS solves the **FL Configuration Problem**: deciding *which* client training algorithm, server aggregation rule, server optimizer, differential privacy mechanism, and communication compression scheme to deploy for a given real-world operational domain (Mobile Apps vs. IoT Edge) subject to privacy, bandwidth, and robustness constraints.

---

## Quickstart for Friends & Collaborators (Zero Setup / No Training Required)

All benchmark results, preprocessed Kaggle datasets, and pre-trained models are already included in this repository. You **do not need to spend hours training models from scratch**.

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/AkshayRP12/Federated-.git
cd Federated-
pip install -r requirements.txt
```

### 2. Run the Decentralized Client-Server Fleet
To start the FastAPI central aggregation server and 3 independent edge clients training on local data partitions:
```powershell
python scripts/launch_demo_fleet.py
```
- Open your browser to **http://localhost:8000** to watch the server dashboard update in real time as clients submit parameter gradients.

### 3. Run the Meta-Learning Recommender & Interactive UI
To launch the Streamlit web dashboard:
```powershell
streamlit run demo/app.py
```
- Open your browser to **http://localhost:8501** to inspect the 736 benchmark runs, change operational constraints, and see Pareto-optimal FL recommendations with research citations.

---

## Pre-Trained Model Checkpoints & Preprocessed Kaggle Datasets

This repository includes pre-trained model weights in `checkpoints/` and preprocessed datasets in `data/`:

| Domain | Model Architecture | Dataset & Source | Accuracy | Checkpoint File |
|---|---|---|---|---|
| **IoT Edge** | `HARMLP` (561 features, 6 classes) | UCI HAR Smartphone Dataset (Kaggle) | **87.14%** | `checkpoints/pretrained_har_model.pt` |
| **Mobile Apps** | `MobileCNN` (105,962 parameters) | Subsampled MNIST Digits | **92.60%** | `checkpoints/pretrained_mobile_cnn.pt` |
| **Finance / Fraud** | `FraudMLP` (29 features, 2 classes) | Credit Card Fraud Dataset (Kaggle) | **99.96%** | `checkpoints/pretrained_fraud_model.pt` |

### How to Load Pre-Trained Checkpoints in Python
```python
import torch
from flaas.fl.models import HARMLP, MobileCNN, FraudMLP

# Load IoT Edge HAR model (UCI HAR from Kaggle)
har_ckpt = torch.load("checkpoints/pretrained_har_model.pt", weights_only=False)
model_har = HARMLP(in_features=561, num_classes=6)
model_har.load_state_dict(har_ckpt["state_dict"])
print(f"Loaded {har_ckpt['model_name']} with test accuracy: {har_ckpt['accuracy']:.2f}%")

# Load Mobile CNN model
mobile_ckpt = torch.load("checkpoints/pretrained_mobile_cnn.pt", weights_only=False)
model_mobile = MobileCNN(in_channels=1, num_classes=10)
model_mobile.load_state_dict(mobile_ckpt["state_dict"])
print(f"Loaded {mobile_ckpt['model_name']} with test accuracy: {mobile_ckpt['accuracy']:.2f}%")
```

---

## System Architecture

The repository delivers two complementary systems:
1. **The FLaaS Meta-Learning Recommender (`demo/app.py` & `flaas/recommender/`)**:
   - Matches operational scenarios against 736 empirical FL runs across 16 diverse scenarios.
   - Recommends Pareto-optimal FL configurations with empirical evidence, confidence intervals, and research paper citations.
   - Includes an interactive live federated engine for testing live training directly from the UI.
2. **Part B: Distributed Client-Server System (`run_server.py`, `run_client.py`)**:
   - A real, decoupled network implementation where a FastAPI central server coordinates autonomous local client processes over HTTP.
   - **True Data Isolation**: Each client process strictly reads only its own local partitioned dataset from `data/partitions/client_{id}/data.pt`. No client has access to global data or other clients' partitions.
   - Real model distribution, local training on local batches, vector compression (e.g., 2-bit ternary packing 4 symbols/byte), and server aggregation with global evaluation per round.

---

## Distributed Client-Server Options

### Option A: One-Click Demo Fleet (Recommended)
Runs the central server and 3 concurrent autonomous clients in coordinated subprocesses with live, synchronized terminal output:
```powershell
python scripts/launch_demo_fleet.py
```

### Option B: Manual Multi-Terminal Execution (For Live Demonstrations)

#### Terminal 1: Central Aggregation Server (FastAPI)
```powershell
# Start the central aggregation server on port 8000
python run_server.py --port 8000 --rounds 3 --min-clients 3 --rule FedAvgMean
```
*Options:*
- `--rule FedAvgMean`: Standard sample-weighted averaging.
- `--rule TrimmedMean`: Byzantine-robust coordinate-wise trimmed mean (cuts top/bottom 20% outlier updates).

#### Terminal 2: Edge Client 00 (Identity Compression / Uncompressed FP32)
```powershell
python run_client.py --id 0 --server http://127.0.0.1:8000 --epochs 1 --compressor identity
```

#### Terminal 3: Edge Client 01 (Identity Compression / Uncompressed FP32)
```powershell
python run_client.py --id 1 --server http://127.0.0.1:8000 --epochs 1 --compressor identity
```

#### Terminal 4: Edge Client 02 (Bandwidth-Constrained / 2-Bit Ternary Compression)
```powershell
python run_client.py --id 2 --server http://127.0.0.1:8000 --epochs 1 --compressor ternary_2bit
```

*(Optional: Launch Client 3 and Client 4 similarly using `--id 3` and `--id 4`).*

---

## One-Minute Demo Script (For Presentations & Viva)

| Time | Action | What to Explain to the Audience |
| :--- | :--- | :--- |
| **0:00 - 0:15** | Open **Terminal 1** and start the server: `python run_server.py --rounds 3 --min-clients 3` | *"Here is our central server running on FastAPI at port 8000. It initializes the global neural network (MobileCNN, 105,962 weights) and waits for a cohort of distributed edge clients to check in."* |
| **0:15 - 0:35** | Open **Terminals 2, 3, and 4** and launch clients 0, 1, and 2. | *"Notice that each client process runs completely independently and loads its private local data partition strictly from its own dedicated folder (`data/partitions/client_k/data.pt`). No raw training images are ever shared or transmitted over the network."* |
| **0:35 - 0:50** | Point out the terminal logs in the client and server windows. | *"Look at the terminal logs: Each client downloads the global weights, trains for 1 epoch on its local samples, and uploads only parameter weight updates (deltas). Client 0 and 1 upload uncompressed FP32 updates (413.9 KB), while Client 2 uses our vectorized 2-bit ternary compression, packing 4 symbols per byte down to just 25.8 KB (a 93.7% bandwidth reduction)."* |
| **0:50 - 1:00** | Point out the server round completion log and web dashboard at `http://localhost:8000`. | *"Once the server receives updates from all 3 clients, it executes the aggregation rule (FedAvg or TrimmedMean), updates the global model weights, evaluates accuracy against an unseen test set, and kicks off the next round."* |

---

## Verification & Testing

To run the complete test suite (unit tests, recommender tests, and FastAPI client-server integration):
```powershell
pytest -v
```
All 61 test modules pass cleanly.
