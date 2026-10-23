"""
Federated Learning Simulator Loop with deterministic seeding,
device management, round 0 evaluation, and pluggable client/server modules.
"""

from typing import List, Dict, Any, Optional, Callable
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flaas.fl.device import get_device, clear_device_memory
from flaas.fl.server_rules import (
    BaseServerRule,
    FedAvgMean,
    state_dict_to_vector,
    vector_to_state_dict,
)
from flaas.fl.server_optimizers import BaseServerOptimizer, ServerSGD
from flaas.fl.client_algos import BaseClient, FedAvgClient, ClientResult
from flaas.fl.metrics import evaluate_model


def seed_everything(seed: int = 42) -> None:
    """Sets random seeds across python, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FLSimulator:
    """
    Coordinates federated learning simulation across rounds.
    Designed with small memory footprint for 4GB VRAM.
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        client_algos: Dict[int, BaseClient],
        client_dataloaders: Dict[int, DataLoader],
        test_dataloader: DataLoader,
        server_rule: Optional[BaseServerRule] = None,
        server_optimizer: Optional[BaseServerOptimizer] = None,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.seed = seed
        seed_everything(seed)

        self.device = device or get_device()
        self.model_fn = model_fn
        self.client_algos = client_algos
        self.client_dataloaders = client_dataloaders
        self.test_dataloader = test_dataloader

        self.server_rule = server_rule or FedAvgMean()
        self.server_optimizer = server_optimizer or ServerSGD(lr=1.0)

        # Initialize global model
        init_model = self.model_fn()
        self.global_state_dict = {
            k: v.cpu().clone() for k, v in init_model.state_dict().items()
        }
        del init_model

        # Server-side control variate (for SCAFFOLD)
        self.server_control_variate: Optional[torch.Tensor] = None

    def evaluate_global(
        self,
        criterion: nn.Module,
        compute_auprc: bool = False,
    ) -> Dict[str, float]:
        """Evaluates current global model state on the test dataloader."""
        eval_model = self.model_fn().to(self.device)
        eval_model.load_state_dict(
            {k: v.to(self.device) for k, v in self.global_state_dict.items()}
        )
        metrics = evaluate_model(
            model=eval_model,
            dataloader=self.test_dataloader,
            criterion=criterion,
            device=self.device,
            compute_auprc=compute_auprc,
        )
        del eval_model
        clear_device_memory()
        return metrics

    def run(
        self,
        num_rounds: int = 5,
        epochs_per_round: int = 1,
        participation_rate: float = 1.0,
        compute_auprc: bool = False,
        criterion: Optional[nn.Module] = None,
        verbose: bool = True,
        evaluate_initial: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes FL simulation for num_rounds.
        Returns metrics history.
        """
        eval_criterion = criterion or nn.CrossEntropyLoss()
        client_ids = list(self.client_dataloaders.keys())
        num_total_clients = len(client_ids)

        history: List[Dict[str, Any]] = []
        start_time = time.time()

        # Optional Round 0 (untrained initialization baseline)
        if evaluate_initial:
            r0_metrics = self.evaluate_global(eval_criterion, compute_auprc)
            r0_record = {
                "round": 0,
                "loss": r0_metrics["loss"],
                "accuracy": r0_metrics["accuracy"],
                "macro_f1": r0_metrics["macro_f1"],
                "round_time": 0.0,
            }
            if compute_auprc:
                r0_record["auprc"] = r0_metrics["auprc"]
            history.append(r0_record)
            if verbose:
                print(
                    f"Round 00/{num_rounds:02d} (Init) | "
                    f"Loss: {r0_metrics['loss']:.4f} | "
                    f"Acc: {r0_metrics['accuracy'] * 100:.2f}% | "
                    f"F1: {r0_metrics['macro_f1']:.4f}"
                )

        for round_idx in range(1, num_rounds + 1):
            round_start = time.time()

            # Determine participating clients
            if participation_rate >= 1.0:
                selected_clients = client_ids
            else:
                m = max(1, int(participation_rate * num_total_clients))
                selected_clients = random.sample(client_ids, m)

            # Local training on each client
            client_results: List[ClientResult] = []
            for cid in selected_clients:
                client = self.client_algos[cid]
                dloader = self.client_dataloaders[cid]

                res = client.train(
                    global_state_dict=self.global_state_dict,
                    dataloader=dloader,
                    epochs=epochs_per_round,
                    server_control_variate=self.server_control_variate,
                )
                client_results.append(res)

            # Extract update vectors and weights
            update_vectors = [res.delta for res in client_results]
            sample_counts = [res.num_samples for res in client_results]

            # Server aggregation rule
            aggregated_delta = self.server_rule.aggregate(
                updates=update_vectors,
                weights=[float(s) for s in sample_counts],
            )

            # Update SCAFFOLD server control variate if clients provided delta_c
            scaffold_deltas = [
                res.control_variate_delta
                for res in client_results
                if res.control_variate_delta is not None
            ]
            if scaffold_deltas:
                avg_delta_c = torch.stack(scaffold_deltas, dim=0).mean(dim=0)
                if self.server_control_variate is None:
                    self.server_control_variate = avg_delta_c * (len(selected_clients) / num_total_clients)
                else:
                    self.server_control_variate += avg_delta_c * (len(selected_clients) / num_total_clients)

            # Server optimizer step
            current_w_vec = state_dict_to_vector(self.global_state_dict)
            updated_w_vec = self.server_optimizer.step(current_w_vec, aggregated_delta)
            self.global_state_dict = vector_to_state_dict(updated_w_vec, self.global_state_dict)

            # Evaluation on global test set
            eval_metrics = self.evaluate_global(eval_criterion, compute_auprc)

            round_duration = time.time() - round_start
            round_record = {
                "round": round_idx,
                "loss": eval_metrics["loss"],
                "accuracy": eval_metrics["accuracy"],
                "macro_f1": eval_metrics["macro_f1"],
                "round_time": round_duration,
            }
            if compute_auprc:
                round_record["auprc"] = eval_metrics["auprc"]

            history.append(round_record)

            if verbose:
                acc_pct = eval_metrics["accuracy"] * 100.0
                f1 = eval_metrics["macro_f1"]
                print(
                    f"Round {round_idx:02d}/{num_rounds:02d} | "
                    f"Loss: {eval_metrics['loss']:.4f} | "
                    f"Acc: {acc_pct:.2f}% | "
                    f"F1: {f1:.4f} | "
                    f"Time: {round_duration:.2f}s"
                )

            # Clean memory (crucial for 4GB VRAM)
            clear_device_memory()

        total_time = time.time() - start_time
        return {
            "history": history,
            "final_metrics": history[-1] if history else {},
            "total_time": total_time,
            "device": str(self.device),
        }
