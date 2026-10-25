"""
Dataset partitioners for Federated Learning:
- IID Partitioning
- Dirichlet Non-IID Partitioning over class distributions
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader


def partition_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Partitions dataset uniformly and independently at random across num_clients.
    """
    np.random.seed(seed)
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    client_indices: Dict[int, List[int]] = {}
    samples_per_client = num_samples // num_clients

    for cid in range(num_clients):
        start = cid * samples_per_client
        end = (cid + 1) * samples_per_client if cid < num_clients - 1 else num_samples
        client_indices[cid] = indices[start:end].tolist()

    return client_indices


def partition_dirichlet(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.1,
    num_classes: int = 10,
    seed: int = 42,
    min_samples_per_client: int = 10,
) -> Dict[int, List[int]]:
    """
    Partitions dataset according to a Dirichlet distribution Dir(alpha)
    over class allocations, inducing realistic non-IID label skew.
    """
    np.random.seed(seed)

    # Extract all targets from dataset
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        targets = np.array(dataset.labels)
    else:
        # Fallback iteration
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        c_indices = np.where(targets == c)[0]
        np.random.shuffle(c_indices)

        # Draw proportions from Dirichlet(alpha)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Scale proportions to number of class samples
        counts = (proportions * len(c_indices)).astype(int)

        # Fix rounding discrepancies
        remainder = len(c_indices) - counts.sum()
        if remainder > 0:
            top_clients = np.argsort(proportions)[-remainder:]
            for cid in top_clients:
                counts[cid] += 1

        # Split and assign
        offset = 0
        for cid in range(num_clients):
            cnt = counts[cid]
            if cnt > 0:
                client_indices[cid].extend(c_indices[offset : offset + cnt].tolist())
                offset += cnt

    # Enforce minimum samples per client to prevent empty client subsets
    for cid in range(num_clients):
        if len(client_indices[cid]) < min_samples_per_client:
            needed = min_samples_per_client - len(client_indices[cid])
            donor_cid = max(client_indices.keys(), key=lambda c: len(client_indices[c]))
            if len(client_indices[donor_cid]) > min_samples_per_client + needed:
                donated = client_indices[donor_cid][-needed:]
                client_indices[donor_cid] = client_indices[donor_cid][:-needed]
                client_indices[cid].extend(donated)

    # Shuffle local datasets
    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])

    return client_indices


def get_client_dataloaders(
    dataset: Dataset,
    client_indices: Dict[int, List[int]],
    batch_size: int = 32,
    shuffle: bool = True,
) -> Dict[int, DataLoader]:
    """Wraps client index subsets into PyTorch DataLoaders."""
    loaders = {}
    for cid, idxs in client_indices.items():
        subset = Subset(dataset, idxs)
        loaders[cid] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
