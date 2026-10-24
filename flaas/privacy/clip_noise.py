"""
Differential Privacy primitives for Federated Learning:
- L2 Gradient/Delta Clipping
- Calibrated Gaussian Noise Addition
"""

from typing import Tuple, Optional
import torch


class DPClipperNoise:
    """
    Applies L2 norm clipping and calibrated Gaussian perturbation to updates.
    Delta_clipped = Delta * min(1, clip_norm / ||Delta||_2)
    Delta_perturbed = Delta_clipped + N(0, (noise_multiplier * clip_norm)^2 * I)
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.0,
        seed: Optional[int] = None,
    ):
        if clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {clip_norm}")
        if noise_multiplier < 0:
            raise ValueError(f"noise_multiplier cannot be negative, got {noise_multiplier}")

        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def apply(self, delta: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Clips and optionally adds Gaussian noise.
        Returns: (perturbed_delta, unclipped_l2_norm)
        """
        l2_norm = torch.norm(delta, p=2).item()
        scaling_factor = min(1.0, self.clip_norm / max(l2_norm, 1e-12))
        clipped_delta = delta * scaling_factor

        if self.noise_multiplier > 0.0:
            sigma = self.noise_multiplier * self.clip_norm
            noise = torch.randn(
                clipped_delta.shape,
                dtype=clipped_delta.dtype,
                device=clipped_delta.device,
                generator=self.rng if self.rng is not None else None,
            ) * sigma
            perturbed_delta = clipped_delta + noise
        else:
            perturbed_delta = clipped_delta

        return perturbed_delta, l2_norm
