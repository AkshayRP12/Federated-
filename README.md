# FLaaS: Federated Learning as a Service & Configuration Recommender

FLaaS solves the **FL Configuration Problem**: deciding *which* client training algorithm, server aggregation rule, server optimizer, differential privacy mechanism, and communication compression scheme to deploy for a given real-world operational domain (Mobile Apps vs. IoT Edge) subject to privacy, bandwidth, and robustness constraints.

---

## 1. System Components

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

## 2. Part B: Distributed Client-Server Quickstart

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

## 3. One-Minute Demo Script (For Lecturers / HOD)

Use this step-by-step walkthrough during an evaluation or viva:

| Time | Action | What to Explain to the Lecturer |
| :--- | :--- | :--- |
| **0:00 - 0:15** | Open **Terminal 1** and start the server: `python run_server.py --rounds 3 --min-clients 3` | *"Here is our central server running on FastAPI. It initializes the global neural network (MobileCNN, 105,962 weights) and waits for a cohort of distributed edge clients to check in."* |
| **0:15 - 0:35** | Open **Terminals 2, 3, and 4** and launch clients 0, 1, and 2. | *"Notice that each client process runs completely independently and loads its private local data partition strictly from its own dedicated folder (`data/partitions/client_k/data.pt`). No raw training images are ever shared or transmitted over the network."* |
| **0:35 - 0:50** | Point out the terminal logs in the client and server windows. | *"Look at the terminal logs: Each client downloads the global weights, trains for 1 epoch on its local samples, and uploads only parameter weight updates (deltas). Client 0 and 1 upload uncompressed FP32 updates (413.9 KB), while Client 2 uses our vectorized 2-bit ternary compression, packing 4 symbols per byte down to just 25.8 KB (a 93.7% bandwidth reduction)."* |
| **0:50 - 1:00** | Point out the server round completion log. | *"Once the server receives updates from all 3 clients, it executes the aggregation rule (FedAvg or TrimmedMean), updates the global model weights, evaluates accuracy against an unseen test set, and kicks off the next round."* |

---

## 4. Running the Interactive Streamlit UI
```powershell
streamlit run demo/app.py
```
- **Tab 1 (FL Configuration Recommender)**: Select your domain and operational constraints; inspect recommended algorithm configurations, literature citations, and metric trade-offs.
- **Tab 2 (Live Federated Run)**: Trigger an in-browser lightweight federated learning training run and observe live loss convergence and accuracy updates across edge clients.
- **Tab 3 (Edge Fleet Network View)**: Visual topological layout of the edge client fleet and central server.

---

## 5. Verification & Testing

To run the complete test suite (unit tests, recommender tests, and FastAPI client-server integration):
```powershell
pytest -v
```
All 61 test modules pass cleanly.
