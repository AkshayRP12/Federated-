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

            print(
                f"    [CLIENT #{client_id:02d}] Training finished in {t_train_duration:.2f}s | "
                f"Local Loss: {res.loss:.4f}",
                flush=True,
            )
            print(
                f"    [CLIENT #{client_id:02d}] Uploading parameter update ({wire_kb:.2f} KB) to server...",
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
