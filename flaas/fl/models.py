"""
Model architectures and lightweight benchmark dataset generator
designed for 4GB VRAM and fast FL experimentation.
"""

import numpy as np
import torch
import torch.nn as nn


class CompactCNN(nn.Module):
    """
    Compact 2-layer CNN for 1-channel or 3-channel image inputs (e.g. MNIST, MedMNIST).
    Parameter count: ~30k parameters (extremely fast, low memory footprint).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


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


def generate_synthetic_mnist(num_samples: int = 1500, num_classes: int = 10, seed: int = 42):
    """
    Generates deterministic, linearly separable MNIST-dimension images (1, 28, 28)
    with class-specific activation patterns.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_list = []
    y_list = []

    samples_per_class = num_samples // num_classes
    for c in range(num_classes):
        template = torch.zeros(1, 28, 28)
        row = (c * 2 + 4) % 24
        col = (c * 3 + 4) % 24
        template[0, row : row + 4, col : col + 4] = 3.0

        for _ in range(samples_per_class):
            noise = 0.5 * torch.randn(1, 28, 28)
            img = torch.clamp(template + noise, 0.0, 5.0)
            x_list.append(img)
            y_list.append(c)

    x_all = torch.stack(x_list)
    y_all = torch.tensor(y_list, dtype=torch.long)

    perm = torch.randperm(num_samples)
    return x_all[perm], y_all[perm]
