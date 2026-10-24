"""
Simulated Secure Aggregation (SecAgg).
Implements additive zero-sum pairwise secret sharing (Bonawitz et al., CCS 2017).
Guarantees that individual updates are blinded while the sum of masks over all participating
clients is identically zero: sum(m_i) = 0.
"""

from typing import List, Dict, Tuple
import torch


class SimulatedSecureAggregation:
    """
    Simulates pairwise masking in secure aggregation.
    For each pair of clients (i, j) with i < j, generates a pseudo-random mask r_{i,j}.
    Client i adds +r_{i,j}; client j adds -r_{i,j}.
    The sum of all masks across clients is identically zero:
    sum_i (x_i + m_i) = sum_i x_i.
    """

    def __init__(self, seed: int = 42):
        self.base_seed = seed

    def mask_updates(
        self,
        client_updates: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Blinds each client update with zero-sum pairwise masks.
        """
        client_ids = sorted(list(client_updates.keys()))
        num_clients = len(client_ids)
        if num_clients <= 1:
            return client_updates

        template = next(iter(client_updates.values()))
        device = template.device
        dtype = template.dtype
        shape = template.shape

        # Initialize zero masks for each client
        masks = {cid: torch.zeros(shape, dtype=dtype, device=device) for cid in client_ids}

        # Generate pairwise masks
        for i_idx in range(num_clients):
            cid_i = client_ids[i_idx]
            for j_idx in range(i_idx + 1, num_clients):
                cid_j = client_ids[j_idx]

                # Deterministic seed for pair (cid_i, cid_j)
                pair_seed = (self.base_seed * 10007 + min(cid_i, cid_j) * 31 + max(cid_i, cid_j)) % (2**31 - 1)
                gen = torch.Generator(device=device)
                gen.manual_seed(pair_seed)

                # Pairwise random mask
                r_ij = torch.randn(shape, dtype=dtype, device=device, generator=gen)

                # Additive cancellation
                masks[cid_i].add_(r_ij)
                masks[cid_j].sub_(r_ij)

        # Apply masks to updates
        masked_updates = {}
        for cid in client_ids:
            masked_updates[cid] = client_updates[cid] + masks[cid]

        return masked_updates
