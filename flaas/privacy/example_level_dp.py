"""
Example-Level Differential Privacy (DP-SGD) for Federated Learning Clients.

Formulation (Abadi et al., 2016):
- Record universe: Individual training samples (x_j, y_j) in client's local dataset.
- Per-sample gradient clipping: For each sample in mini-batch of size B:
    g_j = grad(loss(w; x_j, y_j))
    g_j_clipped = g_j * min(1, C_sample / ||g_j||_2)
- Gaussian perturbation:
    g_batch = (1/B) * (Sum_{j=1}^B g_j_clipped + N(0, (sigma_sample * C_sample)^2 * I))
- Parameter update: w <- w - lr * g_batch.
- Accounting: Subsampled Gaussian mechanism with q_sample = B / N_client and steps = E * (N_client / B) * T.
"""

from typing import Tuple, Dict, Any, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flaas.fl.client_algos import ClientResult, state_dict_to_vector


class ExampleLevelDPSGDClient:
    """
    Client that executes Example-Level DP-SGD during local training rounds.
    Computes exact per-sample gradients using vectorized torch.func operations.
    """

    def __init__(
        self,
        client_id: int,
        model_fn: Callable[[], nn.Module],
        device: torch.device,
        learning_rate: float = 0.05,
        clip_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.client_id = client_id
        self.model_fn = model_fn
        self.device = device
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.rng = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def train(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        dataloader: DataLoader,
        epochs: int = 1,
        server_control_variate: Optional[torch.Tensor] = None,
    ) -> ClientResult:
        model = self.model_fn().to(self.device)
        model.load_state_dict({k: v.to(self.device) for k, v in global_state_dict.items()})
        model.train()

        total_loss = 0.0
        total_samples = 0
        steps = 0

        criterion = nn.CrossEntropyLoss(reduction="none")

        for epoch in range(epochs):
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.size(0)
                if batch_size == 0:
                    continue

                total_samples += batch_size
                steps += 1

                # Extract trainable parameters
                param_names = [name for name, p in model.named_parameters() if p.requires_grad]
                params = {name: p for name, p in model.named_parameters() if p.requires_grad}

                # Define per-sample loss function
                def compute_single_loss(p_dict, x_i, y_i):
                    out = torch.func.functional_call(model, p_dict, x_i.unsqueeze(0))
                    return nn.functional.cross_entropy(out, y_i.unsqueeze(0))

                # Vectorized per-sample gradients
                ft_grad_fn = torch.func.vmap(torch.func.grad(compute_single_loss), in_dims=(None, 0, 0))
                per_sample_grads = ft_grad_fn(params, x, y)

                # Compute per-sample L2 norm across all parameters
                flat_per_sample = torch.cat(
                    [per_sample_grads[name].reshape(batch_size, -1) for name in param_names],
                    dim=1,
                )
                sample_norms = torch.norm(flat_per_sample, p=2, dim=1)  # shape: (batch_size,)

                # Compute clipping scaling factors: min(1, C / ||g||_2)
                clip_factors = torch.clamp(self.clip_norm / torch.clamp(sample_norms, min=1e-12), max=1.0)
                # Reshape for broadcasting
                clip_factors = clip_factors.view(batch_size, 1)

                # Apply clipping to each parameter and sum over batch
                clipped_sum_grads = {}
                for name in param_names:
                    g = per_sample_grads[name]  # (batch_size, *param_shape)
                    view_shape = (batch_size,) + (1,) * (g.ndim - 1)
                    clipped_g = g * clip_factors.view(view_shape)
                    clipped_sum_grads[name] = clipped_g.sum(dim=0)

                # Add calibrated Gaussian noise to batch gradient if noise_multiplier > 0
                batch_grads = {}
                for name, p in params.items():
                    sum_g = clipped_sum_grads[name]
                    if self.noise_multiplier > 0.0:
                        std = self.noise_multiplier * self.clip_norm
                        noise = torch.randn(
                            sum_g.shape,
                            dtype=sum_g.dtype,
                            device=self.device,
                            generator=self.rng,
                        ) * std
                        noisy_sum = sum_g + noise
                    else:
                        noisy_sum = sum_g

                    batch_grads[name] = noisy_sum / float(batch_size)

                # Update model parameters via SGD
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in batch_grads:
                            p.data -= self.learning_rate * batch_grads[name]

                # Record batch loss
                with torch.no_grad():
                    out = model(x)
                    loss_val = nn.functional.cross_entropy(out, y).item()
                    total_loss += loss_val * batch_size

        avg_loss = total_loss / max(total_samples, 1)

        # Compute delta = w_local - w_global
        local_sd = model.state_dict()
        delta_vec = state_dict_to_vector(local_sd).to(self.device) - state_dict_to_vector(global_state_dict).to(self.device)

        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return ClientResult(
            client_id=self.client_id,
            delta=delta_vec.cpu(),
            num_samples=total_samples,
            loss=avg_loss,
            extra_metrics={"steps": steps, "clip_norm": self.clip_norm, "noise_multiplier": self.noise_multiplier},
        )
