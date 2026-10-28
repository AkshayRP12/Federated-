"""
Prepares isolated local client data partitions for Part B distributed deployment.
Each client's partition is saved strictly into its own isolated directory:
data/partitions/client_0/data.pt
data/partitions/client_1/data.pt
...
data/partitions/test_data.pt
"""

import os
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from flaas.data.partitioners import partition_dirichlet

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "partitions")


def prepare_partitions(num_clients: int = 5, samples_per_client: int = 200, alpha: float = 0.5):
    os.makedirs(DATA_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    raw_train = datasets.MNIST(root="data_cache", train=True, download=True, transform=transform)
    raw_test = datasets.MNIST(root="data_cache", train=False, download=True, transform=transform)

    # Subsample 2000 train samples for lightweight fast demo
    total_samples = min(len(raw_train), num_clients * samples_per_client * 2)
    sub_train = Subset(raw_train, range(total_samples))

    # Partition with Dirichlet non-IID skew
    client_indices = partition_dirichlet(sub_train, num_clients=num_clients, alpha=alpha, seed=42)

    for cid in range(num_clients):
        c_folder = os.path.join(DATA_DIR, f"client_{cid}")
        os.makedirs(c_folder, exist_ok=True)
        c_sub = Subset(sub_train, client_indices[cid][:samples_per_client])

        # Extract tensors to save in self-contained format
        x_list = []
        y_list = []
        for i in range(len(c_sub)):
            x, y = c_sub[i]
            x_list.append(x)
            y_list.append(y)
        
        c_data = {
            "client_id": cid,
            "x": torch.stack(x_list),
            "y": torch.tensor(y_list, dtype=torch.long),
        }
        out_file = os.path.join(c_folder, "data.pt")
        torch.save(c_data, out_file)
        print(f"Saved Client {cid} data: {len(y_list)} samples -> {out_file}")

    # Save test partition
    test_sub = Subset(raw_test, range(500))
    tx_list, ty_list = [], []
    for i in range(len(test_sub)):
        x, y = test_sub[i]
        tx_list.append(x)
        ty_list.append(y)
    test_data = {
        "x": torch.stack(tx_list),
        "y": torch.tensor(ty_list, dtype=torch.long),
    }
    test_file = os.path.join(DATA_DIR, "test_data.pt")
    torch.save(test_data, test_file)
    print(f"Saved Global Test data: {len(ty_list)} samples -> {test_file}")


if __name__ == "__main__":
    prepare_partitions()
