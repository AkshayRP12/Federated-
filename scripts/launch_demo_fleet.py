"""
One-Click Demo Launcher for Part B.
Spawns the FastAPI central server on localhost:8000 and 3 autonomous clients.
Runs 3 rounds of real decentralized federated learning with live terminal logging.
"""

import os
import sys
import time
import subprocess
import requests

SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_server.py")
CLIENT_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_client.py")
DATA_PREP_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "prepare_client_partitions.py")


def launch():
    print("=" * 80)
    print("FLaaS PART B: DISTRIBUTED CLIENT-SERVER DEMO LAUNCHER")
    print("Scenario: 3 autonomous clients on localhost training MobileCNN on isolated MNIST")
    print("=" * 80)

    # 1. Ensure client partition folders exist
    part_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "partitions", "client_0")
    if not os.path.exists(part_dir):
        print("[LAUNCHER] Partitioning data for clients...")
        subprocess.run([sys.executable, DATA_PREP_SCRIPT], check=True)

    # 2. Spawn Server
    port = 8000
    server_env = os.environ.copy()
    server_env["PYTHONUNBUFFERED"] = "1"
    server_cmd = [sys.executable, SERVER_SCRIPT, "--port", str(port), "--rounds", "3", "--min-clients", "3", "--rule", "FedAvgMean"]
    print(f"\n[LAUNCHER] Starting central server process on port {port}...")
    server_proc = subprocess.Popen(server_cmd, env=server_env)

    # Wait for server to come online
    server_ready = False
    for _ in range(15):
        time.sleep(0.5)
        try:
            r = requests.get(f"http://127.0.0.1:{port}/info", timeout=1)
            if r.status_code == 200:
                server_ready = True
                break
        except Exception:
            pass

    if not server_ready:
        print("[LAUNCHER] ERROR: Server failed to start.")
        server_proc.terminate()
        sys.exit(1)

    print("[LAUNCHER] Central server is ONLINE.\n")

    # 3. Spawn 3 concurrent clients
    client_procs = []
    num_clients = 3
    for cid in range(num_clients):
        # Let client 2 test 2-bit ternary compression
        comp = "ternary_2bit" if cid == 2 else "identity"
        c_cmd = [
            sys.executable, CLIENT_SCRIPT,
            "--id", str(cid),
            "--server", f"http://127.0.0.1:{port}",
            "--compressor", comp,
            "--epochs", "1",
        ]
        print(f"[LAUNCHER] Launching Client #{cid:02d} process (compression={comp})...")
        proc = subprocess.Popen(c_cmd, env=server_env)
        client_procs.append(proc)
        time.sleep(0.3)

    print("\n[LAUNCHER] All client processes running. Monitoring round progression...\n")

    # Wait for all client processes to complete
    for p in client_procs:
        p.wait()

    print("\n[LAUNCHER] All clients finished training.")
    time.sleep(1.0)
    server_proc.terminate()
    print("[LAUNCHER] Server process terminated. Distributed FL demonstration COMPLETE!\n")


if __name__ == "__main__":
    launch()
