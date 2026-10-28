"""
FLaaS Federated Learning Server (FastAPI).
Coordinates decentralized training rounds across autonomous client processes.
Reuses server aggregation rules (FedAvgMean, TrimmedMean) and MobileCNN model.
"""

import os
import sys
import base64
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from flaas.fl.models import MobileCNN
from flaas.fl.server_rules import (
    FedAvgMean,
    TrimmedMean,
    state_dict_to_vector,
    vector_to_state_dict,
)
from flaas.fl.metrics import evaluate_model

app = FastAPI(title="FLaaS Federated Server", version="1.0.0")

# Global Server Context
class ServerContext:
    def __init__(self, target_rounds: int = 3, min_clients: int = 3, rule_name: str = "FedAvgMean"):
        self.target_rounds = target_rounds
        self.min_clients = min_clients
        self.current_round = 1
        self.is_completed = False
        self.rule_name = rule_name
        self.rule = TrimmedMean(beta=0.2) if rule_name == "TrimmedMean" else FedAvgMean()

        # Initialize global model (MobileCNN: 21.8k params)
        self.model = MobileCNN(in_channels=1, num_classes=10)
        self.global_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # Load global test dataset
        part_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "partitions")
        test_file = os.path.join(part_dir, "test_data.pt")
        if os.path.exists(test_file):
            t_data = torch.load(test_file, weights_only=False)
            ds = TensorDataset(t_data["x"], t_data["y"])
            self.test_loader = DataLoader(ds, batch_size=64, shuffle=False)
        else:
            self.test_loader = None

        # Initial evaluation
        self.round_history = []
        if self.test_loader:
            m0 = evaluate_model(self.model, self.test_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
            self.current_acc = float(m0["accuracy"] * 100.0)
        else:
            self.current_acc = 10.0

        print("=" * 80)
        print(f"[SERVER] FLaaS Federated Server Initialized")
        print(f"[SERVER] Model: MobileCNN (21,800 parameters) | Aggregation: {rule_name}")
        print(f"[SERVER] Target Rounds: {target_rounds} | Cohort Size: {min_clients} clients/round")
        print(f"[SERVER] Initial Global Test Accuracy: {self.current_acc:.2f}%")
        print("=" * 80)

        # Buffer for updates in current round: client_id -> update_dict
        self.round_buffer: Dict[int, Dict[str, Any]] = {}


context: ServerContext = None


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FLaaS Central Server Dashboard</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --primary: #2563eb;
      --success: #16a34a;
      --success-bg: #dcfce7;
      --warn: #d97706;
      --warn-bg: #fef3c7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }
    h1 { font-size: 1.6rem; font-weight: 700; color: #1e293b; }
    .subtitle { font-size: 0.9rem; color: var(--muted); margin-top: 4px; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
    }
    .badge-live { background-color: var(--success-bg); color: var(--success); }
    .badge-done { background-color: #dbeafe; color: var(--primary); }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background-color: currentColor;
    }
    .grid-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
    }
    .card-label {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      font-weight: 600;
      margin-bottom: 6px;
    }
    .card-val {
      font-size: 1.75rem;
      font-weight: 700;
      color: #0f172a;
    }
    .card-sub { font-size: 0.8rem; color: var(--muted); margin-top: 4px; }
    .section-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 12px;
      color: #1e293b;
    }
    .progress-bar-bg {
      background: #e2e8f0;
      height: 10px;
      border-radius: 9999px;
      overflow: hidden;
      margin: 12px 0;
    }
    .progress-bar-fill {
      background: var(--primary);
      height: 100%;
      width: 0%;
      transition: width 0.4s ease;
    }
    .client-list {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .client-card {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      background: #fafafa;
    }
    .client-card.received {
      border-color: #86efac;
      background: #f0fdf4;
    }
    .client-header {
      display: flex;
      justify-content: space-between;
      font-weight: 600;
      font-size: 0.9rem;
      margin-bottom: 4px;
    }
    .client-details { font-size: 0.8rem; color: var(--muted); }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 0.9rem;
    }
    th, td {
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
    }
    th {
      background-color: #f1f5f9;
      color: #475569;
      font-weight: 600;
    }
    tr:hover { background-color: #f8fafc; }
    .footer {
      margin-top: 32px;
      padding-top: 16px;
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.85rem;
      color: var(--muted);
    }
    .links a {
      color: var(--primary);
      text-decoration: none;
      margin-left: 16px;
      font-weight: 500;
    }
    .links a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>FLaaS Central Aggregation Server</h1>
        <div class="subtitle">Decentralized Federated Learning Hub & Model Coordinator (FastAPI)</div>
      </div>
      <div>
        <span id="status-badge" class="badge badge-live">
          <span class="status-dot"></span>
          <span id="status-text">LISTENING FOR CLIENTS</span>
        </span>
      </div>
    </header>

    <div class="grid-metrics">
      <div class="card">
        <div class="card-label">Current Round</div>
        <div class="card-val" id="val-round">- / -</div>
        <div class="card-sub" id="sub-round">Target rounds</div>
      </div>
      <div class="card">
        <div class="card-label">Global Test Accuracy</div>
        <div class="card-val" id="val-accuracy">--%</div>
        <div class="card-sub">Evaluated on held-out test partition</div>
      </div>
      <div class="card">
        <div class="card-label">Aggregation Algorithm</div>
        <div class="card-val" style="font-size: 1.3rem;" id="val-rule">FedAvgMean</div>
        <div class="card-sub">MobileCNN (105,962 weights)</div>
      </div>
      <div class="card">
        <div class="card-label">Cohort Required</div>
        <div class="card-val" id="val-cohort">- clients</div>
        <div class="card-sub">Autonomous edge processes</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 24px;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="section-title" style="margin: 0;">Active Round Ingestion Tracker</div>
        <div style="font-size: 0.85rem; font-weight: 600;" id="cohort-ratio">0 / 3 updates received</div>
      </div>
      <div class="progress-bar-bg">
        <div id="progress-fill" class="progress-bar-fill"></div>
      </div>
      <div id="clients-grid" class="client-list">
        <!-- Rendered client status cards -->
      </div>
    </div>

    <div class="card">
      <div class="section-title">Completed Rounds Progression</div>
      <div style="overflow-x: auto;">
        <table id="history-table">
          <thead>
            <tr>
              <th>Round</th>
              <th>Participating Clients</th>
              <th>Mean Client Loss</th>
              <th>Global Test Accuracy</th>
              <th>Uplink Wire Traffic</th>
            </tr>
          </thead>
          <tbody id="history-body">
            <tr>
              <td colspan="5" style="text-align: center; color: var(--muted); padding: 20px;">
                No rounds completed yet. Waiting for client updates...
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="footer">
      <div>Live status updates every 1.0s via HTTP polling</div>
      <div class="links">
        <a href="/docs" target="_blank">OpenAPI / Swagger Docs</a>
        <a href="/model" target="_blank">Model Weight Vector</a>
        <a href="/info" target="_blank">JSON API (/info)</a>
      </div>
    </div>
  </div>

  <script>
    async function updateDashboard() {
      try {
        const res = await fetch('/info');
        if (!res.ok) return;
        const data = await res.json();

        // Round & Acc
        document.getElementById('val-round').textContent = `${data.current_round} / ${data.target_rounds}`;
        document.getElementById('val-accuracy').textContent = `${data.global_accuracy.toFixed(2)}%`;
        document.getElementById('val-rule').textContent = data.rule;
        document.getElementById('val-cohort').textContent = `${data.min_clients} clients`;

        // Status badge
        const badge = document.getElementById('status-badge');
        const badgeText = document.getElementById('status-text');
        if (data.status === 'completed') {
          badge.className = 'badge badge-done';
          badgeText.textContent = 'ALL ROUNDS COMPLETED';
        } else {
          badge.className = 'badge badge-live';
          badgeText.textContent = 'LISTENING FOR CLIENTS';
        }

        // Cohort progress
        const reportedCount = data.clients_reported_this_round.length;
        const totalNeeded = data.min_clients;
        const pct = Math.min(100, Math.round((reportedCount / totalNeeded) * 100));
        document.getElementById('cohort-ratio').textContent = `${reportedCount} / ${totalNeeded} updates received (${pct}%)`;
        document.getElementById('progress-fill').style.width = `${pct}%`;

        // Client cards
        const grid = document.getElementById('clients-grid');
        let html = '';
        for (let i = 0; i < totalNeeded; i++) {
          const isReceived = data.clients_reported_this_round.includes(i);
          const details = data.client_details && data.client_details[i];
          if (isReceived && details) {
            html += `
              <div class="client-card received">
                <div class="client-header">
                  <span>Client #${String(i).padStart(2, '0')}</span>
                  <span style="color: #16a34a; font-size: 0.8rem;">Update Accepted</span>
                </div>
                <div class="client-details">
                  Wire Size: <strong>${details.wire_kb} KB</strong> | Samples: ${details.num_samples}<br>
                  Local Loss: <strong>${details.loss}</strong>
                </div>
              </div>`;
          } else {
            html += `
              <div class="client-card">
                <div class="client-header">
                  <span>Client #${String(i).padStart(2, '0')}</span>
                  <span style="color: var(--muted); font-size: 0.8rem;">Waiting for update...</span>
                </div>
                <div class="client-details">Local partition: data/partitions/client_${i}/data.pt</div>
              </div>`;
          }
        }
        grid.innerHTML = html;

        // History table
        const tbody = document.getElementById('history-body');
        if (data.round_history && data.round_history.length > 0) {
          let rows = '';
          data.round_history.forEach(r => {
            const clientList = r.clients.map(c => `Client #${String(c).padStart(2, '0')}`).join(', ');
            rows += `
              <tr>
                <td><strong>Round ${r.round}</strong></td>
                <td>${clientList}</td>
                <td>${r.loss.toFixed(4)}</td>
                <td><strong style="color: #16a34a;">${r.accuracy.toFixed(2)}%</strong></td>
                <td>${r.wire_kb.toFixed(2)} KB</td>
              </tr>`;
          });
          tbody.innerHTML = rows;
        }
      } catch (e) {
        console.error("Dashboard poll failed", e);
      }
    }
    updateDashboard();
    setInterval(updateDashboard, 1000);
  </script>
</body>
</html>
"""


class UpdatePayload(BaseModel):
    client_id: int
    round_idx: int
    num_samples: int
    delta_b64: str
    wire_bytes: int
    loss: float


@app.get("/info")
def get_info():
    """Returns server configuration and progress."""
    reported = {}
    for cid, u in context.round_buffer.items():
        reported[cid] = {
            "num_samples": u["num_samples"],
            "wire_bytes": u["wire_bytes"],
            "wire_kb": round(u["wire_bytes"] / 1024.0, 2),
            "loss": round(float(u["loss"]), 4),
        }
    return {
        "status": "completed" if context.is_completed else "training",
        "current_round": context.current_round,
        "target_rounds": context.target_rounds,
        "min_clients": context.min_clients,
        "global_accuracy": round(context.current_acc, 2),
        "rule": context.rule_name,
        "clients_reported_this_round": list(context.round_buffer.keys()),
        "client_details": reported,
        "round_history": context.round_history,
        "num_params": 105962,
    }


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Serves the interactive Central Aggregation Server Web Dashboard."""
    return HTMLResponse(content=HTML_DASHBOARD)


@app.get("/model")
def get_model():
    """Provides current global model weights to clients."""
    flat_weights = state_dict_to_vector(context.global_state)
    weight_bytes = flat_weights.numpy().tobytes()
    b64_weights = base64.b64encode(weight_bytes).decode("utf-8")

    return {
        "round_idx": context.current_round,
        "is_completed": context.is_completed,
        "weights_b64": b64_weights,
        "num_params": flat_weights.numel(),
    }


@app.post("/upload_update")
def upload_update(payload: UpdatePayload):
    """Receives local client updates, checks cohort completion, and aggregates."""
    if context.is_completed:
        return {"status": "finished", "message": "Server has completed all rounds."}

    if payload.round_idx != context.current_round:
        raise HTTPException(
            status_code=400,
            detail=f"Mismatched round. Server is at round {context.current_round}, got {payload.round_idx}",
        )

    # Decode update vector
    raw_bytes = base64.b64decode(payload.delta_b64.encode("utf-8"))
    np_arr = np.frombuffer(raw_bytes, dtype=np.float32).copy()
    flat_arr = torch.from_numpy(np_arr)

    size_kb = payload.wire_bytes / 1024.0
    print(
        f"[SERVER] Received update from Client #{payload.client_id:02d} "
        f"(Round {payload.round_idx}): {size_kb:.2f} KB | {payload.num_samples} samples | Loss: {payload.loss:.4f}",
        flush=True,
    )

    context.round_buffer[payload.client_id] = {
        "delta": flat_arr,
        "num_samples": payload.num_samples,
        "wire_bytes": payload.wire_bytes,
        "loss": payload.loss,
    }

    # If cohort is full, aggregate
    if len(context.round_buffer) >= context.min_clients:
        aggregate_round()

    return {
        "status": "accepted",
        "round_idx": context.current_round,
        "clients_received": len(context.round_buffer),
        "clients_needed": context.min_clients,
    }


def aggregate_round():
    """Executes server aggregation, updates global weights, and logs progress."""
    r_idx = context.current_round
    updates = [u["delta"] for u in context.round_buffer.values()]
    samples = [u["num_samples"] for u in context.round_buffer.values()]
    total_wire = sum(u["wire_bytes"] for u in context.round_buffer.values())
    avg_loss = float(sum(u["loss"] for u in context.round_buffer.values()) / len(updates))

    # Aggregation
    agg_delta = context.rule.aggregate(updates, weights=[float(s) for s in samples])

    # Update global weights
    curr_vec = state_dict_to_vector(context.global_state)
    updated_vec = curr_vec + agg_delta
    context.global_state = vector_to_state_dict(updated_vec, context.global_state)
    context.model.load_state_dict(context.global_state)

    # Evaluate accuracy
    if context.test_loader:
        m = evaluate_model(context.model, context.test_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        context.current_acc = float(m["accuracy"] * 100.0)

    total_wire_kb = total_wire / 1024.0
    cids_str = ", ".join([f"Client #{cid:02d}" for cid in sorted(context.round_buffer.keys())])

    print("\n" + "=" * 80, flush=True)
    print(f"[SERVER] ROUND {r_idx:02d} COMPLETE: Aggregated updates from [{cids_str}]", flush=True)
    print(
        f"[SERVER] Global Test Accuracy: {context.current_acc:.2f}% | "
        f"Mean Client Loss: {avg_loss:.4f} | Total Wire Traffic: {total_wire_kb:.2f} KB",
        flush=True,
    )
    print("=" * 80 + "\n", flush=True)

    context.round_history.append({
        "round": r_idx,
        "accuracy": context.current_acc,
        "loss": avg_loss,
        "wire_kb": total_wire_kb,
        "clients": list(context.round_buffer.keys()),
    })

    # Clear round buffer
    context.round_buffer.clear()
    context.current_round += 1

    if context.current_round > context.target_rounds:
        context.is_completed = True
        print(f"[SERVER] All {context.target_rounds} communication rounds finished successfully!\n")


def create_app(target_rounds: int = 3, min_clients: int = 3, rule_name: str = "FedAvgMean"):
    global context
    context = ServerContext(target_rounds=target_rounds, min_clients=min_clients, rule_name=rule_name)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLaaS Federated Learning Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds (default: 3)")
    parser.add_argument("--min-clients", type=int, default=3, help="Cohort size per round (default: 3)")
    parser.add_argument("--rule", type=str, default="FedAvgMean", choices=["FedAvgMean", "TrimmedMean"], help="Aggregation rule")
    args = parser.parse_args()

    create_app(target_rounds=args.rounds, min_clients=args.min_clients, rule_name=args.rule)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
