"""
Communication Compression Schemes for Federated Learning:
- Top-k Sparsification
- QSGD (Alistarh et al., NeurIPS 2017)
- Sparse Ternary Quantization
Each compressor provides encode/decode methods and computes exact wire byte footprints.
"""

from typing import Tuple, Dict, Any, Optional
import math
import torch


class BaseCompressor:
    """Base class for communication compressors."""

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Compresses tensor.
        Returns: (decompressed_tensor, wire_bytes)
        """
        raise NotImplementedError


class IdentityCompressor(BaseCompressor):
    """Pass-through uncompressed baseline (standard 32-bit float)."""

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        wire_bytes = tensor.numel() * 4  # 4 bytes per float32
        return tensor.clone(), wire_bytes


class TopKCompressor(BaseCompressor):
    """
    Top-k Sparsification:
    Retains the k elements with largest absolute magnitude; sets rest to zero.
    Wire format: k float32 values (4 bytes) + k int32 indices (4 bytes) + 4 bytes header.
    Total wire bytes = 4 + 8 * k.
    """

    def __init__(self, k_ratio: float = 0.1):
        if not 0.0 < k_ratio <= 1.0:
            raise ValueError(f"k_ratio must be in (0, 1], got {k_ratio}")
        self.k_ratio = k_ratio

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        d = tensor.numel()
        k = max(1, int(self.k_ratio * d))

        if k >= d:
            wire_bytes = d * 4
            return tensor.clone(), wire_bytes

        flat = tensor.reshape(-1)
        _, indices = torch.topk(torch.abs(flat), k=k, largest=True)

        sparse = torch.zeros_like(flat)
        sparse[indices] = flat[indices]

        # Wire bytes: 4 bytes (header for count k) + k * (4 bytes float32 + 4 bytes int32)
        wire_bytes = 4 + k * 8
        return sparse.reshape(tensor.shape), wire_bytes


class QSGDCompressor(BaseCompressor):
    """
    Quantized SGD (QSGD) (Alistarh et al., NeurIPS 2017).
    Stochastically quantizes coordinates into s uniformly spaced levels in [-1, 1],
    scaled by the L2 norm of the vector.
    Wire format: L2 norm (float32, 4 bytes) + d * bits_per_coord.
    """

    def __init__(self, s: int = 8, seed: Optional[int] = None):
        if s < 1:
            raise ValueError(f"s must be >= 1, got {s}")
        self.s = s
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        flat = tensor.reshape(-1)
        d = flat.numel()

        norm = torch.norm(flat, p=2)
        if norm.item() == 0.0:
            wire_bytes = 4 + math.ceil(d * 1 / 8)
            return torch.zeros_like(tensor), wire_bytes

        # Normalized coordinates in [0, s]
        abs_flat = torch.abs(flat)
        scaled = (abs_flat / norm) * self.s
        floored = torch.floor(scaled)
        prob = scaled - floored

        # Stochastic rounding
        rand = torch.rand(
            flat.shape,
            dtype=flat.dtype,
            device=flat.device,
            generator=self.rng if self.rng is not None else None,
        )
        rounded = floored + (rand < prob).float()

        # Dequantize: sign(v) * (norm / s) * rounded
        dequantized = torch.sign(flat) * (norm / self.s) * rounded

        # Wire bits: 1 sign bit + ceil(log2(s + 1)) level bits per coordinate
        bits_per_coord = 1 + math.ceil(math.log2(self.s + 1))
        # Total wire bytes: 4 bytes norm + ceil(d * bits_per_coord / 8)
        wire_bytes = 4 + math.ceil((d * bits_per_coord) / 8)

        return dequantized.reshape(tensor.shape), wire_bytes


class SparseTernaryCompressor(BaseCompressor):
    """
    Sparse Ternary Quantization:
    Coordinates with magnitude above threshold are quantized to {-1, +1} * scale,
    while smaller coordinates are zeroed out.
    Wire format: scale (float32, 4 bytes) + ternary symbols (2 bits per coordinate).
    """

    def __init__(self, threshold_factor: float = 1.0):
        self.threshold_factor = threshold_factor

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        flat = tensor.reshape(-1)
        d = flat.numel()

        l1_mean = torch.mean(torch.abs(flat)).item()
        threshold = self.threshold_factor * l1_mean

        mask = torch.abs(flat) >= threshold
        selected_elements = flat[mask]

        if selected_elements.numel() == 0:
            wire_bytes = 4 + math.ceil((d * 2) / 8)
            return torch.zeros_like(tensor), wire_bytes

        # Scale is average magnitude of active elements
        scale = torch.mean(torch.abs(selected_elements)).item()

        ternary = torch.zeros_like(flat)
        ternary[mask] = torch.sign(selected_elements) * scale

        # Wire bytes: 4 bytes scale + 2 bits per coordinate for ternary {-1, 0, +1}
        wire_bytes = 4 + math.ceil((d * 2) / 8)
        return ternary.reshape(tensor.shape), wire_bytes
