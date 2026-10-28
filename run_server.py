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
    return {
        "status": "completed" if context.is_completed else "training",
        "current_round": context.current_round,
        "target_rounds": context.target_rounds,
        "min_clients": context.min_clients,
        "global_accuracy": context.current_acc,
        "rule": context.rule_name,
        "clients_reported_this_round": list(context.round_buffer.keys()),
    }


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
