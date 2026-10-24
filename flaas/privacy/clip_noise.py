"""
Differential Privacy primitives for Federated Learning:
- Client-Level L2 Update Clipping
- Calibrated Central / Local Gaussian Perturbation

Formal Sensitivity & Noise Calibration:
- Let C be the client-level L2 clipping threshold: ||Delta_i||_2 <= C.
- When m clients participate in a round, the sum of clipped updates has L2 global sensitivity:
    Delta_2 (Sum) = C
- Calibrated Gaussian noise added to the sum has variance: sigma^2 * C^2 * I.
- For the averaged model update Delta_bar = (1/m) * Sum(Delta_i), the added noise is:
    N(0, (sigma^2 * C^2 / m^2) * I) with standard deviation: (sigma * C) / m.
- This formally realizes Central Client-Level (User-Level) Differential Privacy (McMahan et al., 2018).
"""

from typing import Tuple, Optional
import torch


class DPClipperNoise:
    """
    Client-level Differential Privacy engine.
    Supports both central aggregation noise addition and local client perturbation.
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.0,
        mode: str = "central",  # "central" or "local"
        seed: Optional[int] = None,
    ):
        if clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {clip_norm}")
        if noise_multiplier < 0:
            raise ValueError(f"noise_multiplier cannot be negative, got {noise_multiplier}")
        if mode not in ("central", "local"):
            raise ValueError(f"mode must be 'central' or 'local', got {mode}")

        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.mode = mode
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def clip_only(self, delta: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Enforces client-level L2 sensitivity threshold:
        delta_clipped = delta * min(1, C / ||delta||_2).
        Returns: (clipped_delta, raw_l2_norm)
        """
        l2_norm = torch.norm(delta, p=2).item()
        scaling_factor = min(1.0, self.clip_norm / max(l2_norm, 1e-12))
        clipped_delta = delta * scaling_factor
        return clipped_delta, l2_norm

    def add_central_noise_to_average(self, aggregated_avg_delta: torch.Tensor, num_sampled_clients: int) -> torch.Tensor:
        """
        Adds calibrated Central Client-Level Gaussian noise to the averaged update.
        Sensitivity of sum is C; sensitivity of average is C / m.
        std = (noise_multiplier * C) / m.
        """
        if self.noise_multiplier <= 0.0 or num_sampled_clients <= 0:
            return aggregated_avg_delta

        std = (self.noise_multiplier * self.clip_norm) / float(num_sampled_clients)
        noise = torch.randn(
            aggregated_avg_delta.shape,
            dtype=aggregated_avg_delta.dtype,
            device=aggregated_avg_delta.device,
            generator=self.rng if self.rng is not None else None,
        ) * std
        return aggregated_avg_delta + noise

    def add_noise(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """Adds zero-mean Gaussian noise with specified standard deviation sigma."""
        if sigma <= 0.0:
            return tensor
        noise = torch.randn(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=self.rng if self.rng is not None else None,
        ) * sigma
        return tensor + noise

    def apply(self, delta: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Legacy/Local apply: clips delta to C and adds noise with std = noise_multiplier * C.
        Used for local DP or standalone vector tests.
        """
        clipped, raw_norm = self.clip_only(delta)
        if self.noise_multiplier > 0.0:
            noisy = self.add_noise(clipped, sigma=self.noise_multiplier * self.clip_norm)
        else:
            noisy = clipped
        return noisy, raw_norm
