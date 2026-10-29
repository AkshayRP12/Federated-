"""
FLaaS Edge Client Device Simulator (Port 8001).
Provides a dedicated visual web frontend for edge client devices.
Allows users to trigger local training, view private partitions, and inspect gradient compression.
"""

import os
import sys
import argparse
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import requests

from run_client import train_single_client_round

app = FastAPI(title="FLaaS Edge Client Fleet UI", version="1.0.0")

HTML_CLIENT_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FLaaS Edge Client Fleet Simulator</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --success: #16a34a;
      --success-bg: #dcfce7;
      --purple: #7c3aed;
      --purple-bg: #f3e8ff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1150px; margin: 0 auto; }
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
    .top-controls {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    .btn {
      padding: 9px 18px;
      border-radius: 6px;
      border: none;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      transition: background-color 0.2s, transform 0.1s;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .btn:active { transform: translateY(1px); }
    .btn-primary { background: var(--primary); color: #fff; }
    .btn-primary:hover { background: var(--primary-hover); }
    .btn-success { background: var(--success); color: #fff; }
    .btn-success:hover { background: #15803d; }
    .btn-secondary { background: #e2e8f0; color: #334155; }
    .btn-secondary:hover { background: #cbd5e1; }
    .btn:disabled { opacity: 0.6; cursor: not-allowed; }

    .server-status-card {
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      padding: 14px 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
    }

    .device-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
      gap: 20px;
      margin-bottom: 28px;
    }
    .device-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      position: relative;
    }
    .device-badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .badge-fp32 { background: #e0f2fe; color: #0284c7; }
    .badge-compressed { background: var(--purple-bg); color: var(--purple); }

    .device-title {
      font-size: 1.15rem;
      font-weight: 700;
      margin-top: 10px;
      margin-bottom: 4px;
    }
    .device-model {
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 14px;
    }
    .info-row {
      display: flex;
      justify-content: space-between;
      padding: 6px 0;
      border-bottom: 1px solid #f1f5f9;
      font-size: 0.85rem;
    }
    .info-label { color: var(--muted); }
    .info-val { font-weight: 600; color: var(--text); }

    .log-box {
      margin-top: 14px;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 12px;
      font-size: 0.8rem;
      min-height: 95px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .log-box.success {
      background: #f0fdf4;
      border-color: #86efac;
    }
    .btn-train-client {
      width: 100%;
      margin-top: 14px;
      justify-content: center;
    }

    .footer {
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      color: var(--muted);
    }
    .footer a { color: var(--primary); text-decoration: none; margin-left: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>FLaaS Edge Client Device Fleet</h1>
        <div class="subtitle">Simulated Autonomous Edge Devices Training Locally on Isolated Private Partitions</div>
      </div>
      <div class="top-controls">
        <button id="btn-sync-fleet" class="btn btn-success" onclick="trainAllDevices()">
          ⚡ Train Entire Fleet (One-Click Sync)
        </button>
      </div>
    </header>

    <div class="server-status-card">
      <div>
        <strong style="color: #1e40af;">Connected Central Server:</strong> <span id="srv-url">http://127.0.0.1:8000</span>
        <div style="font-size: 0.85rem; color: #3b82f6; margin-top: 2px;" id="srv-info">Checking connection...</div>
      </div>
      <div>
        <a href="http://localhost:8000" target="_blank" class="btn btn-secondary" style="font-size: 0.85rem; padding: 6px 12px;">
          Open Server Dashboard (Port 8000) ↗
        </a>
      </div>
    </div>

    <div class="device-grid">
      <!-- Device 00 -->
      <div class="device-card" id="card-0">
        <span class="device-badge badge-fp32">FP32 Uncompressed</span>
        <div class="device-title">Client #00: Smartphone</div>
        <div class="device-model">Hardware: Google Pixel 7 (Simulated Edge Node)</div>
        <div class="info-row">
          <span class="info-label">Local Data Partition:</span>
          <span class="info-val">client_0/data.pt</span>
        </div>
        <div class="info-row">
          <span class="info-label">Private Training Samples:</span>
          <span class="info-val">200 samples (MNIST)</span>
        </div>
        <div class="info-row">
          <span class="info-label">Local SGD Epochs:</span>
          <span class="info-val">1 epoch (Batch size: 32)</span>
        </div>
        <div class="info-row">
          <span class="info-label">Compression Scheme:</span>
          <span class="info-val">Identity (413.91 KB)</span>
        </div>
        <div class="log-box" id="log-0">
          Device status: <strong>IDLE / READY</strong><br>
          Waiting for training command.
        </div>
        <button class="btn btn-primary btn-train-client" id="btn-0" onclick="trainDevice(0, 'identity')">
          🚀 Train Client #00 & Upload
        </button>
      </div>

      <!-- Device 01 -->
      <div class="device-card" id="card-1">
        <span class="device-badge badge-fp32">FP32 Uncompressed</span>
        <div class="device-title">Client #01: Tablet</div>
        <div class="device-model">Hardware: Samsung Galaxy Tab (Simulated Edge Node)</div>
        <div class="info-row">
          <span class="info-label">Local Data Partition:</span>
          <span class="info-val">client_1/data.pt</span>
        </div>
        <div class="info-row">
          <span class="info-label">Private Training Samples:</span>
          <span class="info-val">200 samples (MNIST)</span>
        </div>
        <div class="info-row">
          <span class="info-label">Local SGD Epochs:</span>
          <span class="info-val">1 epoch (Batch size: 32)</span>
        </div>
        <div class="info-row">
          <span class="info-label">Compression Scheme:</span>
          <span class="info-val">Identity (413.91 KB)</span>
        </div>
        <div class="log-box" id="log-1">
          Device status: <strong>IDLE / READY</strong><br>
          Waiting for training command.
        </div>
        <button class="btn btn-primary btn-train-client" id="btn-1" onclick="trainDevice(1, 'identity')">
          🚀 Train Client #01 & Upload
        </button>
      </div>

      <!-- Device 02 -->
      <div class="device-card" id="card-2">
        <span class="device-badge badge-compressed">2-Bit Ternary (93.7% Smaller)</span>
        <div class="device-title">Client #02: Smartwatch / IoT</div>
        <div class="device-model">Hardware: Low-Power Sensor (Bandwidth Constrained)</div>
        <div class="info-row">
          <span class="info-label">Local Data Partition:</span>
          <span class="info-val">client_2/data.pt</span>
        </div>
        <div class="info-row">
          <span class="info-label">Private Training Samples:</span>
          <span class="info-val">200 samples (MNIST)</span>
        </div>
        <div class="info-row">
          <span class="info-label">Local SGD Epochs:</span>
          <span class="info-val">1 epoch (Batch size: 32)</span>
        </div>
        <div class="info-row">
          <span class="info-label">Compression Scheme:</span>
          <span class="info-val" style="color: var(--purple); font-weight: 700;">2-Bit Packing (25.87 KB)</span>
        </div>
        <div class="log-box" id="log-2">
          Device status: <strong>IDLE / READY</strong><br>
          Waiting for training command.
        </div>
        <button class="btn btn-primary btn-train-client" id="btn-2" style="background: var(--purple);" onclick="trainDevice(2, 'ternary_2bit')">
          ⚡ Train Client #02 (2-Bit Compressed)
        </button>
      </div>
    </div>

    <div class="footer">
      <div>Data Isolation Guarantee: Raw data never leaves client memory. Only gradients (Δw) are transmitted.</div>
      <div>
        <a href="http://localhost:8000" target="_blank">Server Dashboard (Port 8000)</a>
        <a href="http://localhost:8501" target="_blank">Recommender UI (Port 8501)</a>
      </div>
    </div>
  </div>

  <script>
    async function checkServer() {
      try {
        const res = await fetch('/api/server_info');
        const data = await res.json();
        if (data.status === 'online') {
          document.getElementById('srv-info').innerHTML = 
            `🟢 Server is <strong>ONLINE</strong> (Round ${data.current_round} / ${data.target_rounds} | Global Acc: <strong>${data.global_accuracy.toFixed(2)}%</strong> | Rule: ${data.rule})`;
        } else {
          document.getElementById('srv-info').innerHTML = `🔴 Server is <strong>OFFLINE</strong> on port 8000. Start it with 'python run_server.py'`;
        }
      } catch (e) {
        document.getElementById('srv-info').textContent = 'Could not contact server on port 8000.';
      }
    }
    checkServer();
    setInterval(checkServer, 2000);

    async function trainDevice(cid, compressor) {
      const btn = document.getElementById(`btn-${cid}`);
      const log = document.getElementById(`log-${cid}`);
      btn.disabled = true;
      btn.textContent = '⏳ Training locally (PyTorch SGD)...';
      log.className = 'log-box';
      log.innerHTML = `Downloading model weights...<br>Executing local forward & backward pass on 200 samples...`;

      try {
        const res = await fetch(`/api/train_client?id=${cid}&compressor=${compressor}`, { method: 'POST' });
        const data = await res.json();

        if (data.status === 'success') {
          log.className = 'log-box success';
          log.innerHTML = `
            ✅ <strong>Update Accepted by Server (Round ${data.round_idx})</strong><br>
            • Local Loss: <strong>${data.loss}</strong> (1 epoch in ${data.duration_sec}s)<br>
            • Wire Size: <strong>${data.wire_kb} KB</strong> (Ratio: ${data.compression_ratio})<br>
            • Gradient Norm ||Δw||: <code>${data.delta_norm}</code><br>
            • Preview: <code>[${data.preview.join(', ')}]</code>
          `;
          btn.textContent = `✓ Round ${data.round_idx} Completed`;
        } else {
          log.innerHTML = `❌ Error: ${data.message}`;
          btn.textContent = 'Retry Training';
        }
      } catch (e) {
        log.innerHTML = `❌ Network Error: ${e}`;
        btn.textContent = 'Retry Training';
      } finally {
        setTimeout(() => { btn.disabled = false; }, 2000);
      }
    }

    async function trainAllDevices() {
      const mainBtn = document.getElementById('btn-sync-fleet');
      mainBtn.disabled = true;
      mainBtn.textContent = '⏳ Syncing Entire Fleet to Server...';

      await trainDevice(0, 'identity');
      await trainDevice(1, 'identity');
      await trainDevice(2, 'ternary_2bit');

      mainBtn.disabled = false;
      mainBtn.textContent = '⚡ Train Entire Fleet (One-Click Sync)';
      checkServer();
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def get_client_dashboard():
    """Serves the Client Fleet Device Simulator Dashboard."""
    return HTMLResponse(content=HTML_CLIENT_DASHBOARD)


@app.get("/api/server_info")
def get_server_info():
    """Proxies status check to FastAPI central server on port 8000."""
    try:
        r = requests.get("http://127.0.0.1:8000/info", timeout=1.5)
        if r.status_code == 200:
            data = r.json()
            data["status"] = "online"
            return data
    except Exception:
        pass
    return {"status": "offline"}


@app.post("/api/train_client")
def api_train_client(
    id: int = Query(..., description="Client ID (0, 1, 2)"),
    compressor: str = Query("identity", description="Compressor type (identity or ternary_2bit)"),
):
    """Triggers one local training round on client device and uploads to server."""
    res = train_single_client_round(
        client_id=id,
        server_url="http://127.0.0.1:8000",
        compressor_type=compressor,
        epochs=1,
    )
    return JSONResponse(content=res)


def main():
    parser = argparse.ArgumentParser(description="FLaaS Edge Client Device Dashboard")
    parser.add_argument("--port", type=int, default=8001, help="Port to run client dashboard (default: 8001)")
    args = parser.parse_args()

    print("=" * 80)
    print(f"[CLIENT UI] FLaaS Edge Client Device Dashboard starting on http://localhost:{args.port}")
    print(f"[CLIENT UI] Communicating with Central Server on http://localhost:8000")
    print("=" * 80)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

