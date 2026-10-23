"""
Server-side Optimizers for Federated Learning.
Supports:
- ServerSGD (standard FedAvg update with optional server learning rate & momentum)
- ServerAdam (Reddi et al., ICLR 2021, Adaptive Federated Optimization - FedAdam)
"""

from typing import Optional, Dict
import torch


class BaseServerOptimizer:
    """Abstract base class for server optimizers."""

    def step(self, global_weights: torch.Tensor, aggregated_delta: torch.Tensor) -> torch.Tensor:
        """
        Applies aggregated_delta (w_client - w_global) to global_weights.
        Returns the updated global_weights.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets any internal optimizer state."""
        pass


class ServerSGD(BaseServerOptimizer):
    """
    Server-side SGD optimizer.
    w_{t+1} = w_t + server_lr * velocity
    velocity = momentum * velocity_{prev} + aggregated_delta
    """

    def __init__(self, lr: float = 1.0, momentum: float = 0.0):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.lr = lr
        self.momentum = momentum
        self.velocity: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.velocity = None

    def step(self, global_weights: torch.Tensor, aggregated_delta: torch.Tensor) -> torch.Tensor:
        if self.momentum > 0.0:
            if self.velocity is None:
                self.velocity = aggregated_delta.clone()
            else:
                self.velocity = self.momentum * self.velocity + aggregated_delta
            effective_delta = self.velocity
        else:
            effective_delta = aggregated_delta

        return global_weights + self.lr * effective_delta


class ServerAdam(BaseServerOptimizer):
    """
    Server-side Adam optimizer (Adaptive Federated Optimization / FedAdam).
    Reference: Reddi et al., ICLR 2021.
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-4,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.t: int = 0

    def reset(self) -> None:
        self.m = None
        self.v = None
        self.t = 0

    def step(self, global_weights: torch.Tensor, aggregated_delta: torch.Tensor) -> torch.Tensor:
        self.t += 1
        if self.m is None:
            self.m = torch.zeros_like(aggregated_delta)
            self.v = torch.zeros_like(aggregated_delta)

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * aggregated_delta
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * torch.square(aggregated_delta)

        # Compute bias-corrected estimates
        m_hat = self.m / (1.0 - (self.beta1 ** self.t))
        v_hat = self.v / (1.0 - (self.beta2 ** self.t))

        step_direction = m_hat / (torch.sqrt(v_hat) + self.eps)
        return global_weights + self.lr * step_direction
