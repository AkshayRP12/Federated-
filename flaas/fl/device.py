"""
Hardware device helper with automatic fallback and VRAM protection.
Order of preference: CUDA -> MPS -> CPU.
"""

from typing import Optional
import torch


def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """
    Selects the best available compute device or respects preferred_device if valid.
    Fallback chain: CUDA -> MPS -> CPU.
    """
    if preferred_device is not None:
        if preferred_device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preferred_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif preferred_device == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_device_memory() -> None:
    """
    Safely releases unreferenced device memory (especially critical for 4GB VRAM).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
