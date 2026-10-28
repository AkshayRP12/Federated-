"""
Domain Adapters for 5 Federated Learning Application Domains.
Each adapter provides:
- Real dataset loader with graceful, labelled synthetic fallback.
- Explicit 'dataset_source' metadata ('real' or 'synthetic_fallback').
- Fast GPU tensor preloading (GPUDataset) to eliminate CPU->GPU PCIe transfer bottlenecks.
"""

from typing import Dict, Any, Tuple, Optional, List
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader


class GPUDataset(Dataset):
    """
    In-memory GPU-resident tensor dataset.
    Eliminates CPU->GPU memory copy overhead during training mini-batches.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, device: Optional[torch.device] = None):
        if device is not None:
            self.x = x.to(device)
            self.y = y.to(device)
        else:
            self.x = x
            self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

    def get_batch(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[indices], self.y[indices]


class BaseDomainAdapter:
    """Base interface for domain adapters."""

    domain_name: str = "base"
    dataset_source: str = "unknown"
    in_shape: Tuple[int, ...] = ()
    num_classes: int = 10

    def load_data(
        self,
        data_dir: Optional[str] = None,
        seed: int = 42,
    ) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "dataset_source": self.dataset_source,
            "in_shape": self.in_shape,
            "num_classes": self.num_classes,
        }


# ==============================================================================
# 1. Healthcare Domain (MedMNIST BloodMNIST / PathMNIST)
# ==============================================================================
class HealthcareAdapter(BaseDomainAdapter):
    """
    Healthcare Domain: MedMNIST PathMNIST (9 classes of colorectal histology)
    or BloodMNIST (8 classes of blood cells).
    Images: 3x28x28, multi-class classification.
    """

    domain_name = "healthcare"
    in_shape = (3, 28, 28)
    num_classes = 9

    def load_data(
        self,
        data_dir: str = "data/medmnist",
        seed: int = 42,
    ) -> Tuple[Dataset, Dataset]:
        os.makedirs(data_dir, exist_ok=True)
        path_npz = os.path.join(data_dir, "pathmnist.npz")
        blood_npz = os.path.join(data_dir, "bloodmnist.npz")

        target_npz = None
        if os.path.exists(path_npz) and os.path.getsize(path_npz) > 1000:
            target_npz = path_npz
            self.num_classes = 9
        elif os.path.exists(blood_npz) and os.path.getsize(blood_npz) > 1000:
            target_npz = blood_npz
            self.num_classes = 8

        if target_npz is not None:
            try:
                data = np.load(target_npz)
                x_train = torch.from_numpy(data["train_images"]).permute(0, 3, 1, 2).float() / 255.0
                y_train = torch.from_numpy(data["train_labels"].squeeze()).long()
                x_test = torch.from_numpy(data["test_images"]).permute(0, 3, 1, 2).float() / 255.0
                y_test = torch.from_numpy(data["test_labels"].squeeze()).long()
                self.dataset_source = "real"
                return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
            except Exception:
                pass

        # Deterministic, labelled synthetic fallback (avoids stalling on network timeouts)
        rng = np.random.RandomState(seed)
        n_train, n_test = 3000, 800
        x_train = torch.from_numpy(rng.randn(n_train, 3, 28, 28).astype(np.float32) * 0.2 + 0.5).clamp(0, 1)
        y_train = torch.from_numpy(rng.randint(0, self.num_classes, size=(n_train,))).long()

        x_test = torch.from_numpy(rng.randn(n_test, 3, 28, 28).astype(np.float32) * 0.2 + 0.5).clamp(0, 1)
        y_test = torch.from_numpy(rng.randint(0, self.num_classes, size=(n_test,))).long()

        self.dataset_source = "synthetic_fallback"
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)


# ==============================================================================
# 2. Finance Domain (Credit Card Fraud Detection)
# ==============================================================================
class FinanceAdapter(BaseDomainAdapter):
    """
    Finance Domain: Credit Card Fraud Detection (29 PCA features + Amount, binary class).
    Class 0: Legitimate (99.8%), Class 1: Fraud (0.2%).
    """

    domain_name = "finance"
    in_shape = (29,)
    num_classes = 2

    def load_data(
        self,
        data_dir: str = "data/fraud",
        seed: int = 42,
    ) -> Tuple[Dataset, Dataset]:
        os.makedirs(data_dir, exist_ok=True)
        cached_pt = os.path.join(data_dir, "creditcard_preprocessed.pt")
        if os.path.exists(cached_pt):
            try:
                cached = torch.load(cached_pt, weights_only=True)
                self.dataset_source = "real"
                return TensorDataset(cached["x_train"], cached["y_train"]), TensorDataset(cached["x_test"], cached["y_test"])
            except Exception:
                pass

        csv_path = os.path.join(data_dir, "creditcard.csv")

        if os.path.exists(csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                feature_cols = [c for c in df.columns if c not in ["Time", "Class"]]
                x = df[feature_cols].values.astype(np.float32)
                y = df["Class"].values.astype(np.int64)

                # Standardize features
                x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-7)

                n_train = int(len(x) * 0.8)
                x_train, y_train = torch.from_numpy(x[:n_train]), torch.from_numpy(y[:n_train])
                x_test, y_test = torch.from_numpy(x[n_train:]), torch.from_numpy(y[n_train:])

                self.dataset_source = "real"
                return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
            except Exception:
                pass

        # Deterministic, labelled synthetic fallback with realistic fraud imbalance (2% fraud)
        rng = np.random.RandomState(seed)
        n_train, n_test = 5000, 1500
        x_train_np = rng.randn(n_train, 29).astype(np.float32)
        y_train_np = (rng.rand(n_train) < 0.02).astype(np.int64)
        x_train_np[y_train_np == 1] += rng.randn(len(x_train_np[y_train_np == 1]), 29) * 0.8 + 1.2

        x_test_np = rng.randn(n_test, 29).astype(np.float32)
        y_test_np = (rng.rand(n_test) < 0.02).astype(np.int64)
        x_test_np[y_test_np == 1] += rng.randn(len(x_test_np[y_test_np == 1]), 29) * 0.8 + 1.2

        x_train, y_train = torch.from_numpy(x_train_np).float(), torch.from_numpy(y_train_np).long()
        x_test, y_test = torch.from_numpy(x_test_np).float(), torch.from_numpy(y_test_np).long()

        self.dataset_source = "synthetic_fallback"
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)


# ==============================================================================
# 3. IoT Edge Domain (UCI HAR Human Activity Recognition)
# ==============================================================================
class IoTEdgeAdapter(BaseDomainAdapter):
    """
    IoT Edge Domain: UCI HAR (Human Activity Recognition)
    561 feature vectors of wearable triaxial accelerometer/gyroscope signals across 6 activities.
    """

    domain_name = "iot_edge"
    in_shape = (561,)
    num_classes = 6

    def load_data(
        self,
        data_dir: str = "data/uci_har",
        seed: int = 42,
    ) -> Tuple[Dataset, Dataset]:
        os.makedirs(data_dir, exist_ok=True)
        cached_pt = os.path.join(data_dir, "uci_har_preprocessed.pt")
        if os.path.exists(cached_pt):
            try:
                cached = torch.load(cached_pt, weights_only=True)
                self.dataset_source = "real"
                return TensorDataset(cached["x_tr"], cached["y_tr"]), TensorDataset(cached["x_te"], cached["y_te"])
            except Exception:
                pass

        train_x_path = os.path.join(data_dir, "X_train.txt")
        train_y_path = os.path.join(data_dir, "y_train.txt")
        test_x_path = os.path.join(data_dir, "X_test.txt")
        test_y_path = os.path.join(data_dir, "y_test.txt")

        # Also support standard UCI HAR extracted directory layout
        if not (os.path.exists(train_x_path) and os.path.exists(train_y_path)):
            nested_train_x = os.path.join(data_dir, "UCI HAR Dataset", "train", "X_train.txt")
            nested_train_y = os.path.join(data_dir, "UCI HAR Dataset", "train", "y_train.txt")
            nested_test_x = os.path.join(data_dir, "UCI HAR Dataset", "test", "X_test.txt")
            nested_test_y = os.path.join(data_dir, "UCI HAR Dataset", "test", "y_test.txt")
            if os.path.exists(nested_train_x) and os.path.exists(nested_train_y):
                train_x_path, train_y_path = nested_train_x, nested_train_y
                test_x_path, test_y_path = nested_test_x, nested_test_y

        if os.path.exists(train_x_path) and os.path.exists(train_y_path):
            try:
                x_train_np = np.loadtxt(train_x_path, dtype=np.float32)
                y_train_np = np.loadtxt(train_y_path, dtype=np.int64) - 1  # 1-indexed to 0-indexed
                x_test_np = np.loadtxt(test_x_path, dtype=np.float32)
                y_test_np = np.loadtxt(test_y_path, dtype=np.int64) - 1

                x_train, y_train = torch.from_numpy(x_train_np), torch.from_numpy(y_train_np)
                x_test, y_test = torch.from_numpy(x_test_np), torch.from_numpy(y_test_np)

                self.dataset_source = "real"
                return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
            except Exception:
                pass

        # Deterministic, labelled synthetic fallback (561 sensor features, 6 activities)
        rng = np.random.RandomState(seed)
        n_train, n_test = 3000, 1000
        x_train_np = rng.randn(n_train, 561).astype(np.float32)
        y_train_np = rng.randint(0, self.num_classes, size=(n_train,))
        for c in range(self.num_classes):
            mask = y_train_np == c
            x_train_np[mask, c * 50 : (c + 1) * 50] += 1.5

        x_test_np = rng.randn(n_test, 561).astype(np.float32)
        y_test_np = rng.randint(0, self.num_classes, size=(n_test,))
        for c in range(self.num_classes):
            mask = y_test_np == c
            x_test_np[mask, c * 50 : (c + 1) * 50] += 1.5

        x_train, y_train = torch.from_numpy(x_train_np).float(), torch.from_numpy(y_train_np).long()
        x_test, y_test = torch.from_numpy(x_test_np).float(), torch.from_numpy(y_test_np).long()

        self.dataset_source = "synthetic_fallback"
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)


# ==============================================================================
# 4. Mobile Domain (FashionMNIST / EMNIST)
# ==============================================================================
class MobileAdapter(BaseDomainAdapter):
    """
    Mobile Apps Domain: FashionMNIST / EMNIST (10 classes of apparel / digits, 1x28x28).
    """

    domain_name = "mobile_apps"
    in_shape = (1, 28, 28)
    num_classes = 10

    def load_data(
        self,
        data_dir: str = "data/fashion_mnist",
        seed: int = 42,
    ) -> Tuple[Dataset, Dataset]:
        os.makedirs(data_dir, exist_ok=True)
        # Check if local fashion mnist exists
        if os.path.exists(os.path.join(data_dir, "FashionMNIST", "raw")):
            try:
                from torchvision.datasets import FashionMNIST
                from torchvision import transforms
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
                train_ds = FashionMNIST(root=data_dir, train=True, download=False, transform=transform)
                test_ds = FashionMNIST(root=data_dir, train=False, download=False, transform=transform)
                self.dataset_source = "real"
                return train_ds, test_ds
            except Exception:
                pass

        # Check if local MNIST exists (which is already downloaded!)
        if os.path.exists("data/mnist/MNIST/raw"):
            try:
                from torchvision.datasets import MNIST
                raw_train = MNIST(root="data/mnist", train=True, download=False)
                raw_test = MNIST(root="data/mnist", train=False, download=False)

                train_x = ((raw_train.data.float() / 255.0).unsqueeze(1) - 0.1307) / 0.3081
                train_y = raw_train.targets.long()

                test_x = ((raw_test.data.float() / 255.0).unsqueeze(1) - 0.1307) / 0.3081
                test_y = raw_test.targets.long()

                self.dataset_source = "real"
                return TensorDataset(train_x, train_y), TensorDataset(test_x, test_y)
            except Exception:
                pass

        # Deterministic, labelled synthetic fallback
        rng = np.random.RandomState(seed)
        n_train, n_test = 4000, 1000
        x_train = torch.from_numpy(rng.randn(n_train, 1, 28, 28).astype(np.float32) * 0.3).clamp(-1, 1)
        y_train = torch.from_numpy(rng.randint(0, 10, size=(n_train,))).long()

        x_test = torch.from_numpy(rng.randn(n_test, 1, 28, 28).astype(np.float32) * 0.3).clamp(-1, 1)
        y_test = torch.from_numpy(rng.randint(0, 10, size=(n_test,))).long()

        self.dataset_source = "synthetic_fallback"
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)


# ==============================================================================
# 5. Autonomous Vehicles Domain (GTSRB German Traffic Signs)
# ==============================================================================
class AutonomousVehiclesAdapter(BaseDomainAdapter):
    """
    Autonomous Vehicles Domain: GTSRB (German Traffic Sign Recognition Benchmark).
    Images: 3x32x32, 43 classes of traffic signs.
    """

    domain_name = "autonomous_vehicles"
    in_shape = (3, 32, 32)
    num_classes = 43

    def load_data(
        self,
        data_dir: str = "data/gtsrb",
        seed: int = 42,
    ) -> Tuple[Dataset, Dataset]:
        os.makedirs(data_dir, exist_ok=True)
        cached_pt = os.path.join(data_dir, "gtsrb_preprocessed.pt")
        if os.path.exists(cached_pt):
            try:
                cached = torch.load(cached_pt, weights_only=True)
                x, y = cached["x"], cached["y"]
                gen = torch.Generator().manual_seed(seed)
                perm = torch.randperm(len(x), generator=gen)
                n_train = int(len(x) * 0.8)
                train_idx, test_idx = perm[:n_train], perm[n_train:]
                train_ds = TensorDataset(x[train_idx], y[train_idx])
                test_ds = TensorDataset(x[test_idx], y[test_idx])
                self.dataset_source = "real"
                return train_ds, test_ds
            except Exception:
                pass

        train_path = os.path.join(data_dir, "gtsrb", "Train")
        if not os.path.exists(train_path):
            train_path = os.path.join(data_dir, "Train")

        if os.path.exists(train_path):
            try:
                from torchvision.datasets import ImageFolder
                from torchvision import transforms
                from torch.utils.data import random_split
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
                ])
                full_ds = ImageFolder(train_path, transform=transform)
                n_train = int(len(full_ds) * 0.8)
                n_test = len(full_ds) - n_train
                generator = torch.Generator().manual_seed(seed)
                train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=generator)
                self.dataset_source = "real"
                return train_ds, test_ds
            except Exception:
                pass

        if os.path.exists(os.path.join(data_dir, "gtsrb")):
            try:
                from torchvision.datasets import GTSRB
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
                ])
                train_ds = GTSRB(root=data_dir, split="train", download=False, transform=transform)
                test_ds = GTSRB(root=data_dir, split="test", download=False, transform=transform)
                self.dataset_source = "real"
                return train_ds, test_ds
            except Exception:
                pass

        # Deterministic, labelled synthetic fallback (avoids massive GTSRB download)
        rng = np.random.RandomState(seed)
        n_train, n_test = 3000, 800
        x_train = torch.from_numpy(rng.randn(n_train, 3, 32, 32).astype(np.float32) * 0.3).clamp(-1, 1)
        y_train = torch.from_numpy(rng.randint(0, self.num_classes, size=(n_train,))).long()

        x_test = torch.from_numpy(rng.randn(n_test, 3, 32, 32).astype(np.float32) * 0.3).clamp(-1, 1)
        y_test = torch.from_numpy(rng.randint(0, self.num_classes, size=(n_test,))).long()

        self.dataset_source = "synthetic_fallback"
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)


# Registry of domain adapters
DOMAIN_ADAPTERS: Dict[str, BaseDomainAdapter] = {
    "healthcare": HealthcareAdapter(),
    "finance": FinanceAdapter(),
    "iot_edge": IoTEdgeAdapter(),
    "mobile_apps": MobileAdapter(),
    "autonomous_vehicles": AutonomousVehiclesAdapter(),
}


def get_domain_adapter(domain: str) -> BaseDomainAdapter:
    """Retrieves domain adapter by name."""
    if domain not in DOMAIN_ADAPTERS:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_ADAPTERS.keys())}")
    return DOMAIN_ADAPTERS[domain]
