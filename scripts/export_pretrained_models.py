"""
Export Pre-trained Models for GitHub Release.
Trains/saves converged model checkpoints for:
1. IoT Edge (HARMLP on UCI HAR Smartphone activity dataset - Kaggle)
2. Mobile Applications (MobileCNN on MNIST)
3. Financial Fraud Detection (FraudMLP on Credit Card Fraud - Kaggle)

Saves checkpoints into checkpoints/ directory.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from flaas.fl.models import HARMLP, MobileCNN, FraudMLP

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_save_har():
    print("[EXPORT] Training & saving IoT Edge HARMLP (UCI HAR)...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uci_har", "uci_har_preprocessed.pt")
    if not os.path.exists(data_path):
        print("  UCI HAR data not found at", data_path)
        return

    data = torch.load(data_path, weights_only=True)
    train_ds = TensorDataset(data["x_tr"], data["y_tr"])
    test_ds = TensorDataset(data["x_te"], data["y_te"])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = HARMLP(in_features=561, num_classes=6).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = (correct / total) * 100.0
    print(f"  Trained HARMLP Final Test Accuracy: {acc:.2f}%")

    out_path = os.path.join(CHECKPOINT_DIR, "pretrained_har_model.pt")
    torch.save({
        "model_name": "HARMLP",
        "domain": "iot_edge",
        "dataset": "UCI HAR (Kaggle)",
        "in_features": 561,
        "num_classes": 6,
        "accuracy": acc,
        "state_dict": model.cpu().state_dict(),
    }, out_path)
    print(f"  Saved checkpoint to {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


def train_and_save_mobile():
    print("[EXPORT] Training & saving MobileCNN (Mobile Apps / MNIST)...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "partitions", "test_data.pt")
    c0_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "partitions", "client_0", "data.pt")
    if not os.path.exists(data_path) or not os.path.exists(c0_path):
        print("  Data partitions not found.")
        return

    # Combine client partitions for pretraining
    xs, ys = [], []
    for cid in range(5):
        cp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "partitions", f"client_{cid}", "data.pt")
        if os.path.exists(cp):
            cd = torch.load(cp, weights_only=False)
            xs.append(cd["x"])
            ys.append(cd["y"])

    train_ds = TensorDataset(torch.cat(xs, dim=0), torch.cat(ys, dim=0))
    test_data = torch.load(data_path, weights_only=False)
    test_ds = TensorDataset(test_data["x"], test_data["y"])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = MobileCNN(in_channels=1, num_classes=10).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = (correct / total) * 100.0
    print(f"  Trained MobileCNN Final Test Accuracy: {acc:.2f}%")

    out_path = os.path.join(CHECKPOINT_DIR, "pretrained_mobile_cnn.pt")
    torch.save({
        "model_name": "MobileCNN",
        "domain": "mobile_apps",
        "dataset": "MNIST Subsampled",
        "num_params": 105962,
        "accuracy": acc,
        "state_dict": model.cpu().state_dict(),
    }, out_path)
    print(f"  Saved checkpoint to {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


def train_and_save_fraud():
    print("[EXPORT] Training & saving FraudMLP (Credit Card Fraud - Kaggle)...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fraud", "creditcard_preprocessed.pt")
    if not os.path.exists(data_path):
        print("  Credit card fraud data not found at", data_path)
        return

    data = torch.load(data_path, weights_only=True)
    train_ds = TensorDataset(data["x_train"][:20000], data["y_train"][:20000])
    test_ds = TensorDataset(data["x_test"][:5000], data["y_test"][:5000])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = FraudMLP(in_features=29, num_classes=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = (correct / total) * 100.0
    print(f"  Trained FraudMLP Final Test Accuracy: {acc:.2f}%")

    out_path = os.path.join(CHECKPOINT_DIR, "pretrained_fraud_model.pt")
    torch.save({
        "model_name": "FraudMLP",
        "domain": "finance",
        "dataset": "Credit Card Fraud (Kaggle)",
        "in_features": 29,
        "num_classes": 2,
        "accuracy": acc,
        "state_dict": model.cpu().state_dict(),
    }, out_path)
    print(f"  Saved checkpoint to {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    train_and_save_har()
    train_and_save_mobile()
    train_and_save_fraud()
    print("[EXPORT] Pre-trained models exported successfully to checkpoints/")
