"""
Client-side FL training algorithms implemented in PyTorch.
Includes:
- FedAvg (McMahan et al., AISTATS 2017)
- FedProx (Li et al., MLSys 2020)
- SCAFFOLD (Karimireddy et al., ICML 2020)
- FedNova (Wang et al., NeurIPS 2020)
"""

from typing import Dict, List, Optional, Tuple, Any
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flaas.fl.server_rules import state_dict_to_vector, vector_to_state_dict


class ClientResult:
    """Container for the output of local client training."""

    def __init__(
        self,
        client_id: int,
        delta: torch.Tensor,
        num_samples: int,
        loss: float,
        extra_metrics: Optional[Dict[str, Any]] = None,
        control_variate_delta: Optional[torch.Tensor] = None,
        local_steps: int = 0,
    ):
        self.client_id = client_id
        self.delta = delta  # w_local - w_global (1D tensor)
        self.num_samples = num_samples
        self.loss = loss
        self.extra_metrics = extra_metrics or {}
        self.control_variate_delta = control_variate_delta
        self.local_steps = local_steps


class BaseClient:
    """Base class for FL client algorithms."""

    def __init__(
        self,
        client_id: int,
        model_fn: Any,
        device: torch.device,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
    ):
        self.client_id = client_id
        self.model_fn = model_fn
        self.device = device
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def train(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        dataloader: DataLoader,
        epochs: int = 1,
        **kwargs,
    ) -> ClientResult:
        raise NotImplementedError


class FedAvgClient(BaseClient):
    """
    Standard Federated Averaging client (McMahan et al., 2017).
    Performs E epochs of local SGD.
    """

    def train(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        dataloader: DataLoader,
        epochs: int = 1,
        **kwargs,
    ) -> ClientResult:
        model: nn.Module = self.model_fn().to(self.device)
        model.load_state_dict(global_state_dict)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        total_loss = 0.0
        total_batches = 0
        total_samples = 0
        local_steps = 0

        for _ in range(epochs):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                total_batches += 1
                local_steps += 1

        avg_loss = total_loss / max(1, total_samples)

        # Compute delta = w_local - w_global
        w_global_vec = state_dict_to_vector(global_state_dict).to(self.device)
        w_local_vec = state_dict_to_vector(model.state_dict()).to(self.device)
        delta = (w_local_vec - w_global_vec).cpu()

        # Clean up model
        del model
        return ClientResult(
            client_id=self.client_id,
            delta=delta,
            num_samples=total_samples,
            loss=avg_loss,
            local_steps=local_steps,
        )


class FedProxClient(BaseClient):
    """
    FedProx client (Li et al., 2020) with proximal regularization:
    Loss = Loss_task + (mu / 2) * ||w - w_global||^2
    """

    def __init__(
        self,
        client_id: int,
        model_fn: Any,
        device: torch.device,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        mu: float = 0.01,
    ):
        super().__init__(client_id, model_fn, device, learning_rate, weight_decay)
        self.mu = mu

    def train(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        dataloader: DataLoader,
        epochs: int = 1,
        **kwargs,
    ) -> ClientResult:
        mu = kwargs.get("mu", self.mu)

        model: nn.Module = self.model_fn().to(self.device)
        model.load_state_dict(global_state_dict)
        model.train()

        # Cache global parameter copies on device for proximal penalty calculation
        global_params = [
            param.detach().clone().to(self.device)
            for param in model.parameters()
        ]

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        total_loss = 0.0
        total_samples = 0
        local_steps = 0

        for _ in range(epochs):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                loss_task = criterion(out, y)

                # Proximal regularization term: (mu / 2) * sum(||w - w_global||^2)
                prox_term = 0.0
                for p, g_p in zip(model.parameters(), global_params):
                    prox_term = prox_term + (p - g_p).norm(2).pow(2)

                loss = loss_task + (0.5 * mu) * prox_term
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                local_steps += 1

        avg_loss = total_loss / max(1, total_samples)

        w_global_vec = state_dict_to_vector(global_state_dict).to(self.device)
        w_local_vec = state_dict_to_vector(model.state_dict()).to(self.device)
        delta = (w_local_vec - w_global_vec).cpu()

        del model, global_params
        return ClientResult(
            client_id=self.client_id,
            delta=delta,
            num_samples=total_samples,
            loss=avg_loss,
            local_steps=local_steps,
        )


class SCAFFOLDClient(BaseClient):
    """
    SCAFFOLD client (Karimireddy et al., ICML 2020).
    Uses control variates (c_i, c) to correct for client drift.
    Local update step: g_corrected = g - c_i + c
    New c_i: c_i+ = c_i - c + 1/(K * eta) * (w_global - w_local)
    """

    def __init__(
        self,
        client_id: int,
        model_fn: Any,
        device: torch.device,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
    ):
        super().__init__(client_id, model_fn, device, learning_rate, weight_decay)
        self.c_i: Optional[torch.Tensor] = None

    def train(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        dataloader: DataLoader,
        epochs: int = 1,
        server_control_variate: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ClientResult:
        model: nn.Module = self.model_fn().to(self.device)
        model.load_state_dict(global_state_dict)
        model.train()

        w_global_vec = state_dict_to_vector(global_state_dict).to(self.device)
        param_dim = w_global_vec.numel()

        if self.c_i is None:
            self.c_i = torch.zeros(param_dim, device=self.device)
        else:
            self.c_i = self.c_i.to(self.device)

        if server_control_variate is None:
            c_server = torch.zeros(param_dim, device=self.device)
        else:
            c_server = server_control_variate.to(self.device)

        # Correction vector: c_server - c_i
        correction_vec = c_server - self.c_i

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        total_loss = 0.0
        total_samples = 0
        local_steps = 0

        # Pre-calculate parameter slices for matching correction_vec
        param_slices = []
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            param_slices.append((offset, offset + numel, p.shape))
            offset += numel

        for _ in range(epochs):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()

                # Apply SCAFFOLD drift correction to gradients: g = g - c_i + c
                for p, (start, end, shape) in zip(model.parameters(), param_slices):
                    if p.grad is not None:
                        p_corr = correction_vec[start:end].reshape(shape)
                        p.grad.data.add_(p_corr)

                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                local_steps += 1

        avg_loss = total_loss / max(1, total_samples)

        w_local_vec = state_dict_to_vector(model.state_dict()).to(self.device)
        delta_w = (w_local_vec - w_global_vec).cpu()

        # Update local control variate c_i+:
        # c_i+ = c_i - c_server + (w_global - w_local) / (local_steps * lr)
        if local_steps > 0:
            c_i_new = (
                self.c_i - c_server + (w_global_vec - w_local_vec) / (local_steps * self.lr)
            )
            delta_c = (c_i_new - self.c_i).cpu()
            self.c_i = c_i_new.detach()
        else:
            delta_c = torch.zeros_like(delta_w)

        del model
        return ClientResult(
            client_id=self.client_id,
            delta=delta_w,
            num_samples=total_samples,
            loss=avg_loss,
            control_variate_delta=delta_c,
            local_steps=local_steps,
        )


class FedNovaClient(BaseClient):
    """
    FedNova client (Wang et al., NeurIPS 2020).
    Normalizes local updates by effective step count tau_i to eliminate objective inconsistency.
    """

    def train(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        dataloader: DataLoader,
        epochs: int = 1,
        **kwargs,
    ) -> ClientResult:
        model: nn.Module = self.model_fn().to(self.device)
        model.load_state_dict(global_state_dict)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        total_loss = 0.0
        total_samples = 0
        local_steps = 0

        for _ in range(epochs):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                local_steps += 1

        avg_loss = total_loss / max(1, total_samples)

        w_global_vec = state_dict_to_vector(global_state_dict).to(self.device)
        w_local_vec = state_dict_to_vector(model.state_dict()).to(self.device)

        # FedNova normalization: normalized gradient update
        raw_delta = w_local_vec - w_global_vec
        tau_i = float(max(1, local_steps))
        normalized_delta = (raw_delta / tau_i).cpu()

        del model
        return ClientResult(
            client_id=self.client_id,
            delta=normalized_delta,
            num_samples=total_samples,
            loss=avg_loss,
            local_steps=local_steps,
            extra_metrics={"tau_i": tau_i},
        )
