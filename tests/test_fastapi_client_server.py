"""
Unit test for Part B: FastAPI client-server endpoints and aggregation logic.
Uses fastapi.testclient.TestClient for headless verification.
"""

import os
import base64
import torch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from run_server import create_app
from flaas.fl.server_rules import state_dict_to_vector


def test_fastapi_server_endpoints_and_aggregation():
    # Initialize server app for 2 rounds with cohort size of 2
    app = create_app(target_rounds=2, min_clients=2, rule_name="FedAvgMean")
    client = TestClient(app)

    # 1. Test / and /dashboard HTML endpoints
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "FLaaS Central Aggregation Server" in root_resp.text

    dash_resp = client.get("/dashboard")
    assert dash_resp.status_code == 200
    assert "Active Round Ingestion Tracker" in dash_resp.text

    # 2. Test /info endpoint
    info_resp = client.get("/info")
    assert info_resp.status_code == 200
    info = info_resp.json()
    assert info["current_round"] == 1
    assert info["min_clients"] == 2
    assert info["status"] == "training"

    # 3. Test /model endpoint
    model_resp = client.get("/model")
    assert model_resp.status_code == 200
    model_data = model_resp.json()
    assert model_data["round_idx"] == 1
    assert model_data["num_params"] == 105962
    assert "weights_b64" in model_data

    # Decode weights
    raw_b64 = model_data["weights_b64"]
    raw_bytes = base64.b64decode(raw_b64.encode("utf-8"))
    weights = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.float32).copy())
    assert weights.numel() == 105962

    # 3. Simulate Client 0 upload
    delta_0 = torch.ones(105962, dtype=torch.float32) * 0.05
    d0_bytes = delta_0.numpy().tobytes()
    payload_0 = {
        "client_id": 0,
        "round_idx": 1,
        "num_samples": 100,
        "delta_b64": base64.b64encode(d0_bytes).decode("utf-8"),
        "wire_bytes": len(d0_bytes),
        "loss": 0.45,
    }
    up0_resp = client.post("/upload_update", json=payload_0)
    assert up0_resp.status_code == 200
    assert up0_resp.json()["clients_received"] == 1

    # 4. Simulate Client 1 upload (triggers aggregation!)
    delta_1 = torch.ones(105962, dtype=torch.float32) * 0.03
    d1_bytes = delta_1.numpy().tobytes()
    payload_1 = {
        "client_id": 1,
        "round_idx": 1,
        "num_samples": 100,
        "delta_b64": base64.b64encode(d1_bytes).decode("utf-8"),
        "wire_bytes": len(d1_bytes),
        "loss": 0.38,
    }
    up1_resp = client.post("/upload_update", json=payload_1)
    assert up1_resp.status_code == 200

    # 5. Verify server moved to Round 2
    info_resp_r2 = client.get("/info")
    assert info_resp_r2.status_code == 200
    assert info_resp_r2.json()["current_round"] == 2

    # 6. Verify updated weights changed by exactly average delta
    model_r2_resp = client.get("/model")
    weights_r2_bytes = base64.b64decode(model_r2_resp.json()["weights_b64"].encode("utf-8"))
    weights_r2 = torch.from_numpy(np.frombuffer(weights_r2_bytes, dtype=np.float32).copy())
    expected_diff = (0.05 + 0.03) / 2.0
    actual_diff = (weights_r2 - weights).mean().item()
    assert abs(actual_diff - expected_diff) < 1e-5
