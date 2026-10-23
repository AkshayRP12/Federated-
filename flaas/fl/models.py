"""
Model architectures and lightweight benchmark dataset generator
designed for 4GB VRAM and fast FL experimentation.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


class MNIST_CNN(nn.Module):
    """
    Standard CNN for 28x28 image classification (MNIST, FashionMNIST).
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


class CompactMLP(nn.Module):
    """
    Compact MLP for tabular and vector-based domains (e.g. Credit Card Fraud, UCI HAR).
    """

    def __init__(self, in_features: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
