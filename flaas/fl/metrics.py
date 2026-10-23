"""
Evaluation metrics for Federated Learning tasks.
Supports:
- Accuracy
- Macro-F1 score
- AUPRC (Area Under Precision-Recall Curve)
- Loss computation
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_curve, auc


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    compute_auprc: bool = False,
) -> Dict[str, float]:
    """
    Evaluates model on dataloader.
    Returns dictionary with accuracy, macro_f1, loss, and optionally auprc.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            if out.shape[1] > 1:
                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)
                # If binary/auprc needed, extract positive class probability
                if out.shape[1] == 2:
                    pos_probs = probs[:, 1]
                else:
                    pos_probs = probs.max(dim=1).values
            else:
                probs = torch.sigmoid(out.squeeze())
                preds = (probs >= 0.5).long()
                pos_probs = probs

            correct += (preds == y).sum().item()

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
            all_probs.extend(pos_probs.cpu().tolist())

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

    avg_loss = total_loss / total_samples
    acc = correct / total_samples

    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)
    macro_f1 = float(f1_score(targets_np, preds_np, average="macro", zero_division=0))

    results = {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }

    if compute_auprc and len(np.unique(targets_np)) > 1:
        probs_np = np.array(all_probs)
        precision, recall, _ = precision_recall_curve(targets_np, probs_np)
        auprc_val = float(auc(recall, precision))
        results["auprc"] = auprc_val
    elif compute_auprc:
        results["auprc"] = 0.0

    return results
