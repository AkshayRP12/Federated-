"""
Core FL Server Aggregation Rules implemented in PyTorch.
Includes:
- FedAvgMean (weighted/unweighted)
- Krum & Multi-Krum (Blanchard et al., NeurIPS 2017)
- TrimmedMean (Yin et al., ICML 2018)
- CoordinateMedian (Yin et al., ICML 2018)
- Bulyan (Guerraoui et al., ICML 2018)
"""

from typing import List, Optional, Tuple, Dict
import torch


def parameters_to_vector(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of tensors into a single 1D tensor."""
    return torch.cat([t.detach().reshape(-1) for t in tensors])


def vector_to_parameters(vec: torch.Tensor, parameter_shapes: List[torch.Size]) -> List[torch.Tensor]:
    """Unflatten a 1D tensor into a list of tensors matching target shapes."""
    tensors = []
    offset = 0
    for shape in parameter_shapes:
        numel = shape.numel()
        tensors.append(vec[offset : offset + numel].reshape(shape))
        offset += numel
    return tensors


def state_dict_to_vector(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten model state_dict float parameters/buffers into a 1D tensor."""
    flat_parts = []
    for k, v in state_dict.items():
        if torch.is_floating_point(v):
            flat_parts.append(v.detach().reshape(-1))
    return torch.cat(flat_parts)


def vector_to_state_dict(
    vec: torch.Tensor,
    template_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Unflatten a 1D tensor into state_dict format using template structure."""
    out_dict = {}
    offset = 0
    for k, v in template_state_dict.items():
        if torch.is_floating_point(v):
            numel = v.numel()
            out_dict[k] = vec[offset : offset + numel].reshape(v.shape).to(v.device, dtype=v.dtype)
            offset += numel
        else:
            out_dict[k] = v.clone()
    return out_dict


class BaseServerRule:
    """Abstract base class for server aggregation rules."""

    def aggregate(
        self,
        updates: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Aggregates a list of 1D update tensors [Delta_1, ..., Delta_n]
        into a single aggregated 1D update tensor.
        """
        raise NotImplementedError


class FedAvgMean(BaseServerRule):
    """Standard FedAvg coordinate-wise weighted or unweighted mean."""

    def aggregate(
        self,
        updates: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        if not updates:
            raise ValueError("No updates provided to aggregate.")

        n = len(updates)
        if weights is None:
            # Unweighted mean
            stacked = torch.stack(updates, dim=0)
            return torch.mean(stacked, dim=0)

        if len(weights) != n:
            raise ValueError(f"Length of weights ({len(weights)}) does not match updates ({n}).")

        w_tensor = torch.tensor(weights, dtype=updates[0].dtype, device=updates[0].device)
        w_sum = torch.sum(w_tensor)
        if w_sum <= 0:
            raise ValueError("Sum of weights must be strictly positive.")
        w_norm = w_tensor / w_sum

        stacked = torch.stack(updates, dim=0)  # [n, d]
        # (w_norm[:, None] * stacked).sum(dim=0)
        return torch.sum(w_norm.unsqueeze(1) * stacked, dim=0)


class TrimmedMean(BaseServerRule):
    """
    Coordinate-wise Trimmed Mean (Yin et al., 2018).
    Discards the smallest and largest beta fraction of values for each coordinate.
    """

    def __init__(self, beta: float = 0.1):
        if not 0.0 <= beta < 0.5:
            raise ValueError(f"beta must be in [0, 0.5), got {beta}")
        self.beta = beta

    def aggregate(
        self,
        updates: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        if not updates:
            raise ValueError("No updates provided to aggregate.")

        stacked = torch.stack(updates, dim=0)  # [n, d]
        n, d = stacked.shape
        k = int(self.beta * n)

        if k == 0:
            # No trimming
            return torch.mean(stacked, dim=0)

        if 2 * k >= n:
            # Fallback if trimming removes everything
            k = max(0, (n - 1) // 2)

        sorted_vals, _ = torch.sort(stacked, dim=0)
        # Keep values from index k to n - k
        trimmed = sorted_vals[k : n - k, :]
        return torch.mean(trimmed, dim=0)


class CoordinateMedian(BaseServerRule):
    """
    Coordinate-wise Median (Yin et al., 2018).
    Computes coordinate-wise median across client updates.
    """

    def aggregate(
        self,
        updates: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        if not updates:
            raise ValueError("No updates provided to aggregate.")

        stacked = torch.stack(updates, dim=0)  # [n, d]
        n = stacked.shape[0]

        sorted_vals, _ = torch.sort(stacked, dim=0)
        if n % 2 == 1:
            return sorted_vals[n // 2, :]
        else:
            mid = n // 2
            return 0.5 * (sorted_vals[mid - 1, :] + sorted_vals[mid, :])


class Krum(BaseServerRule):
    """
    Krum Byzantine-robust aggregation (Blanchard et al., NeurIPS 2017).
    Selects update vector that minimizes sum of Euclidean distances to nearest n - f - 2 neighbors.
    Also supports Multi-Krum (averaging top-m candidates).
    """

    def __init__(self, f: int = 1, multi_krum: bool = False, m: int = 1):
        if f < 0:
            raise ValueError(f"f must be non-negative, got {f}")
        self.f = f
        self.multi_krum = multi_krum
        self.m = m

    def aggregate(
        self,
        updates: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        if not updates:
            raise ValueError("No updates provided to aggregate.")

        n = len(updates)
        if n <= 2:
            # Edge case with very few clients
            stacked = torch.stack(updates, dim=0)
            return torch.mean(stacked, dim=0)

        # Number of neighbors: max(1, min(n - 2, n - self.f - 2))
        num_neighbors = n - self.f - 2
        if num_neighbors < 1:
            num_neighbors = 1

        stacked = torch.stack(updates, dim=0)  # [n, d]

        # Compute pairwise squared Euclidean distances: ||x_i - x_j||^2
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>
        dots = torch.matmul(stacked, stacked.T)
        sq_norms = torch.diag(dots)
        dist_matrix = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * dots
        dist_matrix = torch.clamp(dist_matrix, min=0.0)

        # Score for client i: sum of smallest distances to other clients
        scores = []
        for i in range(n):
            dists = dist_matrix[i].clone()
            dists[i] = float("inf")  # Exclude self
            sorted_dists, _ = torch.sort(dists)
            score = torch.sum(sorted_dists[:num_neighbors])
            scores.append(score)

        scores_tensor = torch.stack(scores)

        if not self.multi_krum:
            best_idx = torch.argmin(scores_tensor).item()
            return updates[best_idx]
        else:
            m = min(self.m, n - self.f)
            m = max(1, m)
            top_indices = torch.topk(scores_tensor, k=m, largest=False).indices.tolist()
            selected = [updates[idx] for idx in top_indices]
            return torch.mean(torch.stack(selected, dim=0), dim=0)


class Bulyan(BaseServerRule):
    """
    Bulyan Byzantine-robust aggregation (Guerraoui et al., ICML 2018).
    Combines Multi-Krum candidate selection with Coordinate-wise Trimmed Mean.
    Standard guarantee requires n >= 4f + 3.
    """

    def __init__(self, f: int = 1):
        if f < 0:
            raise ValueError(f"f must be non-negative, got {f}")
        self.f = f

    def aggregate(
        self,
        updates: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        if not updates:
            raise ValueError("No updates provided to aggregate.")

        n = len(updates)
        if n <= 2:
            return torch.stack(updates, dim=0).mean(dim=0)

        # Effective f if n is small
        eff_f = self.f
        if n < 4 * eff_f + 3:
            eff_f = max(1, (n - 3) // 4) if n >= 7 else 0

        if eff_f == 0:
            return torch.stack(updates, dim=0).mean(dim=0)

        theta = n - 2 * eff_f
        if theta < 1:
            theta = 1

        # Step 1: Iteratively run Krum to select theta candidates
        remaining_indices = list(range(n))
        selected_indices = []

        stacked = torch.stack(updates, dim=0)
        dots = torch.matmul(stacked, stacked.T)
        sq_norms = torch.diag(dots)
        dist_matrix = torch.clamp(sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * dots, min=0.0)

        for _ in range(min(theta, n)):
            best_score = float("inf")
            best_cand = None
            num_neighbors = max(1, len(remaining_indices) - eff_f - 2)

            for idx in remaining_indices:
                other_indices = [j for j in remaining_indices if j != idx]
                if not other_indices:
                    cand_score = torch.tensor(0.0, device=stacked.device)
                else:
                    dists = dist_matrix[idx, other_indices]
                    sorted_dists, _ = torch.sort(dists)
                    cand_score = torch.sum(sorted_dists[:num_neighbors])

                if cand_score < best_score:
                    best_score = cand_score
                    best_cand = idx

            if best_cand is not None:
                selected_indices.append(best_cand)
                remaining_indices.remove(best_cand)

        selected_updates = [updates[i] for i in selected_indices]
        selected_stacked = torch.stack(selected_updates, dim=0)  # [theta, d]

        # Step 2: Trimmed mean on selected candidates
        # Trim gamma = eff_f from top and bottom
        theta_curr = selected_stacked.shape[0]
        gamma = min(eff_f, max(0, (theta_curr - 1) // 2))

        if gamma == 0:
            return torch.mean(selected_stacked, dim=0)

        sorted_cand, _ = torch.sort(selected_stacked, dim=0)
        trimmed = sorted_cand[gamma : theta_curr - gamma, :]
        return torch.mean(trimmed, dim=0)
