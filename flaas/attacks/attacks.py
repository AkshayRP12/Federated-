"""
Adversarial attacks for Federated Learning robustness evaluation.
Implemented in PyTorch:
- SignFlipAttack
- GaussianNoiseAttack
- ScaledUpdateAttack
- LabelFlipWrapper (Dataset/Dataloader label flipping)
"""

from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset


class BaseAttack:
    """Base class for update-manipulation attacks."""

    def perturb(self, honest_delta: torch.Tensor) -> torch.Tensor:
        """Perturbs an honest client update vector."""
        raise NotImplementedError


class SignFlipAttack(BaseAttack):
    """
    Sign-Flip Attack: Negates and optionally scales the honest update direction.
    delta_adv = -scale * delta_honest
    """

    def __init__(self, scale: float = 1.0):
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.scale = scale

    def perturb(self, honest_delta: torch.Tensor) -> torch.Tensor:
        return -self.scale * honest_delta.clone()


class GaussianNoiseAttack(BaseAttack):
    """
    Gaussian Noise Attack: Injects additive or pure Gaussian noise to disrupt convergence.
    """

    def __init__(self, std: float = 1.0, additive: bool = False):
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        self.std = std
        self.additive = additive

    def perturb(self, honest_delta: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(honest_delta) * self.std
        if self.additive:
            return honest_delta + noise
        return noise


class ScaledUpdateAttack(BaseAttack):
    """
    Scaled Update Attack: Drastically scales the update to dominate mean aggregation.
    delta_adv = scale * delta_honest
    """

    def __init__(self, scale: float = 100.0):
        self.scale = scale

    def perturb(self, honest_delta: torch.Tensor) -> torch.Tensor:
        return self.scale * honest_delta.clone()


class LabelFlippedDataset(Dataset):
    """
    Wraps an existing PyTorch Dataset and flips classification labels:
    y_adv = (num_classes - 1 - y) % num_classes
    """

    def __init__(self, base_dataset: Dataset, num_classes: int = 10):
        self.base_dataset = base_dataset
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx]
        y_flipped = (self.num_classes - 1 - int(y)) % self.num_classes
        return x, y_flipped
