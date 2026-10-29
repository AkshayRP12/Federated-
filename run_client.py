"""
FLaaS Federated Learning Client.
Loads isolated local data partition strictly from its own dedicated folder:
data/partitions/client_{id}/data.pt

Reuses FedAvgClient from flaas.fl.client_algos and ThresholdedTernaryCompressor.
"""

import os
import sys
import time
import base64
import argparse
from typing import Dict, Any
import requests
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from flaas.fl.models import MobileCNN
from flaas.fl.client_algos import FedAvgClient
from flaas.fl.server_rules import vector_to_state_dict
from flaas.compression.compressors import IdentityCompressor, ThresholdedTernaryCompressor


def run_client(
    client_id: int,
    server_url: str = "http://127.0.0.1:8000",
    batch_size: int = 32,
    epochs: int = 1,
    lr: float = 0.01,
    compressor_type: str = "identity",
    max_wait_seconds: int = 60,
):
    # 1. Load local partition strictly from client's dedicated folder
    part_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "partitions", f"client_{client_id}"
    )
    data_path = os.path.join(part_folder, "data.pt")

    if not os.path.exists(data_path):
        print(f"[CLIENT #{client_id:02d}] ERROR: Local partition file not found at {data_path}!")
        print(f"[CLIENT #{client_id:02d}] Run 'python scripts/prepare_client_partitions.py' first.")
        sys.exit(1)

    c_data = torch.load(data_path, weights_only=False)
    x = c_data["x"]
    y = c_data["y"]
    num_samples = len(y)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("=" * 70, flush=True)
    print(f"[CLIENT #{client_id:02d}] INITIALIZED", flush=True)
    print(f"[CLIENT #{client_id:02d}] Local Data Path: {data_path}", flush=True)
    print(f"[CLIENT #{client_id:02d}] Local Sample Count: {num_samples} samples", flush=True)
    print(f"[CLIENT #{client_id:02d}] Connected Server: {server_url}", flush=True)
    print(f"[CLIENT #{client_id:02d}] Compression: {compressor_type}", flush=True)
    print("=" * 70, flush=True)

    # Setup compressor
    compressor = ThresholdedTernaryCompressor() if compressor_type == "ternary_2bit" else IdentityCompressor()

    # Template model for state_dict unflattening
    template_model = MobileCNN(in_channels=1, num_classes=10)
    template_state = {k: v.cpu().clone() for k, v in template_model.state_dict().items()}

    # Initialize client algorithm
    client_algo = FedAvgClient(
        client_id=client_id,
        model_fn=lambda: MobileCNN(in_channels=1, num_classes=10),
        device=torch.device("cpu"),
        learning_rate=lr,
    )

    last_trained_round = 0
    start_time = time.time()

    while True:
        if time.time() - start_time > max_wait_seconds:
            print(f"[CLIENT #{client_id:02d}] Timeout: No response from server in {max_wait_seconds}s. Exiting.", flush=True)
            break

        try:
            resp = requests.get(f"{server_url}/model", timeout=5)
            if resp.status_code != 200:
                time.sleep(1.0)
                continue
            model_info = resp.json()
        except Exception:
            time.sleep(1.0)
            continue

        if model_info.get("is_completed", False):
            print(f"[CLIENT #{client_id:02d}] Server signaled completion of all rounds. Exiting cleanly.", flush=True)
            break

        server_round = model_info["round_idx"]

        # If a new round is ready and we haven't trained on it yet
        if server_round > last_trained_round:
            print(f"\n>>> [CLIENT #{client_id:02d}] ROUND {server_round:02d}: Received global model from server.", flush=True)
            print(f"    [CLIENT #{client_id:02d}] Starting local SGD training on {num_samples} samples ({epochs} epoch)...", flush=True)

            # Decode global weights
            raw_b64 = model_info["weights_b64"]
            weight_bytes = base64.b64decode(raw_b64.encode("utf-8"))
            np_arr = np.frombuffer(weight_bytes, dtype=np.float32).copy()
            flat_weights = torch.from_numpy(np_arr)
            global_state_dict = vector_to_state_dict(flat_weights, template_state)

            # Local training
            t_train_start = time.time()
            res = client_algo.train(
                global_state_dict=global_state_dict,
                dataloader=dataloader,
                epochs=epochs,
            )
            t_train_duration = time.time() - t_train_start

            # Compress delta
            comp_delta, wire_bytes = compressor.compress(res.delta)
            wire_kb = wire_bytes / 1024.0

            # Encode delta for upload
            delta_bytes = comp_delta.numpy().tobytes()
            delta_b64 = base64.b64encode(delta_bytes).decode("utf-8")

            # Compute gradient norm & sample slice for demonstration
            delta_norm = float(torch.norm(res.delta).item())
            sample_vals = [round(float(v), 5) for v in res.delta[:4].tolist()]

            print(
                f"    [CLIENT #{client_id:02d}] Training finished in {t_train_duration:.2f}s | "
                f"Local Loss: {res.loss:.4f} | Gradient ||Delta_w||: {delta_norm:.4f}",
                flush=True,
            )
            print(
                f"    [CLIENT #{client_id:02d}] Model Gradient Vector Preview (first 4 of 105,962 weights): {sample_vals}",
                flush=True,
            )
            print(
                f"    [CLIENT #{client_id:02d}] Uploading parameter gradient update ({wire_kb:.2f} KB) to server (NO raw images)...",
                flush=True,
            )

            # Upload to server
            upload_payload = {
                "client_id": client_id,
                "round_idx": server_round,
                "num_samples": res.num_samples,
                "delta_b64": delta_b64,
                "wire_bytes": wire_bytes,
                "loss": float(res.loss),
            }

            try:
                up_resp = requests.post(
                    f"{server_url}/upload_update",
                    json=upload_payload,
                    timeout=10,
                )
                if up_resp.status_code == 200:
                    print(f"    [CLIENT #{client_id:02d}] Update successfully accepted by server.", flush=True)
                    last_trained_round = server_round
                else:
                    print(f"    [CLIENT #{client_id:02d}] Upload error: {up_resp.text}", flush=True)
            except Exception as e:
                print(f"    [CLIENT #{client_id:02d}] Connection error uploading update: {e}", flush=True)

        # Sleep briefly between polls
        time.sleep(0.5)


def train_single_client_round(
    client_id: int,
    server_url: str = "http://127.0.0.1:8000",
    batch_size: int = 32,
    epochs: int = 1,
    lr: float = 0.01,
    compressor_type: str = "identity",
) -> Dict[str, Any]:
    """Executes exactly one local training round for client_id and uploads the update to server_url."""
    try:
        resp = requests.get(f"{server_url}/model", timeout=3)
        if resp.status_code != 200:
            return {"status": "error", "message": f"Server returned status {resp.status_code}"}
        model_info = resp.json()
    except Exception as e:
        return {"status": "error", "message": f"Could not connect to server at {server_url}: {e}"}

    if model_info.get("is_completed", False):
        return {"status": "completed", "message": "Server has already completed all rounds."}

    server_round = model_info["round_idx"]

    # Load local data partition
    part_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "partitions", f"client_{client_id}"
    )
    data_path = os.path.join(part_folder, "data.pt")
    if not os.path.exists(data_path):
        return {"status": "error", "message": f"Local partition not found at {data_path}"}

    c_data = torch.load(data_path, weights_only=False)
    num_samples = len(c_data["y"])
    dataset = TensorDataset(c_data["x"], c_data["y"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Decode global weights
    raw_b64 = model_info["weights_b64"]
    weight_bytes = base64.b64decode(raw_b64.encode("utf-8"))
    flat_weights = torch.from_numpy(np.frombuffer(weight_bytes, dtype=np.float32).copy())

    template_model = MobileCNN(in_channels=1, num_classes=10)
    template_state = {k: v.cpu().clone() for k, v in template_model.state_dict().items()}
    global_state_dict = vector_to_state_dict(flat_weights, template_state)

    # Setup compressor & client algorithm
    compressor = ThresholdedTernaryCompressor() if compressor_type == "ternary_2bit" else IdentityCompressor()
    client_algo = FedAvgClient(
        client_id=client_id,
        model_fn=lambda: MobileCNN(in_channels=1, num_classes=10),
        device=torch.device("cpu"),
        learning_rate=lr,
    )

    t_start = time.time()
    res = client_algo.train(global_state_dict=global_state_dict, dataloader=dataloader, epochs=epochs)
    duration = time.time() - t_start

    # Compress delta
    comp_delta, wire_bytes = compressor.compress(res.delta)
    wire_kb = wire_bytes / 1024.0
    uncomp_kb = (res.delta.numel() * 4) / 1024.0

    delta_norm = float(torch.norm(res.delta).item())
    sample_vals = [round(float(v), 5) for v in res.delta[:4].tolist()]

    delta_bytes = comp_delta.numpy().tobytes()
    delta_b64 = base64.b64encode(delta_bytes).decode("utf-8")

    upload_payload = {
        "client_id": client_id,
        "round_idx": server_round,
        "num_samples": res.num_samples,
        "delta_b64": delta_b64,
        "wire_bytes": wire_bytes,
        "loss": float(res.loss),
    }

    try:
        up_resp = requests.post(f"{server_url}/upload_update", json=upload_payload, timeout=10)
        if up_resp.status_code == 200:
            server_feedback = up_resp.json()
            return {
                "status": "success",
                "round_idx": server_round,
                "client_id": client_id,
                "num_samples": num_samples,
                "loss": round(float(res.loss), 4),
                "wire_kb": round(wire_kb, 2),
                "uncompressed_kb": round(uncomp_kb, 2),
                "compression_ratio": f"{uncomp_kb / max(0.1, wire_kb):.1f}x",
                "delta_norm": round(delta_norm, 4),
                "preview": sample_vals,
                "duration_sec": round(duration, 2),
                "server_feedback": server_feedback,
            }
        else:
            return {"status": "error", "message": f"Server rejected update: {up_resp.text}"}
    except Exception as e:
        return {"status": "error", "message": f"Upload network error: {e}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLaaS Federated Learning Client")
    parser.add_argument("--id", type=int, required=True, help="Client ID (e.g. 0, 1, 2, 3)")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", help="Server URL")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32, help="Local batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.01, help="Local learning rate (default: 0.01)")
    parser.add_argument(
        "--compressor",
        type=str,
        default="identity",
        choices=["identity", "ternary_2bit"],
        help="Compression type (identity or ternary_2bit)",
    )
    args = parser.parse_args()

    run_client(
        client_id=args.id,
        server_url=args.server,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        compressor_type=args.compressor,
    )
