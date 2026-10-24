"""
Model architectures for 5 FL Domains, optimized for 4GB VRAM
and fast FL experimentation with GroupNorm/LayerNorm for non-IID stability.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


class MNIST_CNN(nn.Module):
    """
    Standard CNN for 28x28 image classification (MNIST, FashionMNIST, EMNIST).
    Uses GroupNorm instead of BatchNorm to ensure theoretical stability
    under non-IID class skew (Dirichlet distributions).
    Parameter count: ~30,000 parameters.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


class CompactCNN(MNIST_CNN):
    """Alias for backwards compatibility with earlier tests."""
    pass


class MobileCNN(MNIST_CNN):
    """Mobile domain model (EMNIST/FashionMNIST)."""
    pass


class MedCNN(nn.Module):
    """
    Healthcare domain model (MedMNIST BloodMNIST/PathMNIST: 3x28x28 or 1x28x28, 8 classes).
    Uses GroupNorm for stable federated gradient aggregation.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


class TrafficSignCNN(nn.Module):
    """
    Autonomous Vehicles domain model (GTSRB: 3x32x32, 43 classes).
    Uses GroupNorm for non-IID stability across edge road cameras.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


class FraudMLP(nn.Module):
    """
    Finance domain model (Credit Card Fraud: 29 features, binary classification).
    Uses LayerNorm for stable cross-bank tabular feature distributions.
    """

    def __init__(self, in_features: int = 29, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HARMLP(nn.Module):
    """
    IoT Edge domain model (UCI HAR: 561 sensor features, 6 human activities).
    Uses LayerNorm for non-IID sensor drift resistance.
    """

    def __init__(self, in_features: int = 561, hidden_dim: int = 128, num_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompactMLP(FraudMLP):
    """Alias for backwards compatibility."""
    pass
