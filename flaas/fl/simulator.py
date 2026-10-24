"""
Federated Learning Simulator Loop with deterministic seeding,
device management, pluggable attacks, privacy mechanisms, compression,
and wire byte tracking.
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
from flaas.fl.client_algos import BaseClient, ClientResult
from flaas.fl.metrics import evaluate_model
from flaas.attacks.attacks import BaseAttack
from flaas.privacy.clip_noise import DPClipperNoise
from flaas.privacy.accountant import VerifiedRDPAccountant
from flaas.privacy.secure_agg import SimulatedSecureAggregation
from flaas.compression.compressors import BaseCompressor, IdentityCompressor
from flaas.compression.wire_meter import RoundWireMeter


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
    Includes pluggable attacks, differential privacy, compression, and wire accounting.
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
        # Stage 2 enhancements
        adversary_ids: Optional[List[int]] = None,
        attack: Optional[BaseAttack] = None,
        dp_clipper: Optional[DPClipperNoise] = None,
        dp_accountant: Optional[VerifiedRDPAccountant] = None,
        compressor: Optional[BaseCompressor] = None,
        enable_secure_aggregation: bool = False,
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

        # Stage 2 features
        self.adversary_ids = set(adversary_ids or [])
        self.attack = attack
        self.dp_clipper = dp_clipper
        self.dp_accountant = dp_accountant
        self.compressor = compressor or IdentityCompressor()
        self.enable_secure_aggregation = enable_secure_aggregation
        self.secagg = SimulatedSecureAggregation(seed=seed) if enable_secure_aggregation else None
        self.wire_meter = RoundWireMeter()

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
        Returns metrics history and cumulative wire meter stats.
        """
        eval_criterion = criterion or nn.CrossEntropyLoss()
        client_ids = list(self.client_dataloaders.keys())
        num_total_clients = len(client_ids)

        history: List[Dict[str, Any]] = []
        start_time = time.time()

        if evaluate_initial:
            r0_metrics = self.evaluate_global(eval_criterion, compute_auprc)
            r0_record = {
                "round": 0,
                "loss": r0_metrics["loss"],
                "accuracy": r0_metrics["accuracy"],
                "macro_f1": r0_metrics["macro_f1"],
                "round_time": 0.0,
                "epsilon": 0.0,
                "uplink_bytes": 0,
                "downlink_bytes": 0,
                "total_bytes": 0,
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

        global_model_bytes = state_dict_to_vector(self.global_state_dict).numel() * 4

        for round_idx in range(1, num_rounds + 1):
            round_start = time.time()
            round_uplink_bytes = 0

            # Determine participating clients
            if participation_rate >= 1.0:
                selected_clients = client_ids
            else:
                m = max(1, int(participation_rate * num_total_clients))
                selected_clients = random.sample(client_ids, m)

            # Downlink communication: broadcast global model to selected clients
            round_downlink_bytes = len(selected_clients) * global_model_bytes
            self.wire_meter.record_server_downlink(round_downlink_bytes)

            # Local training on each client
            client_results: List[ClientResult] = []
            processed_updates: Dict[int, torch.Tensor] = {}
            sample_counts: List[int] = []

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
                delta = res.delta

                # 1. Adversarial attack perturbation
                if cid in self.adversary_ids and self.attack is not None:
                    delta = self.attack.perturb(delta)

                # 2. Compression & Wire measurement
                compressed_delta, client_wire_bytes = self.compressor.compress(delta)
                round_uplink_bytes += client_wire_bytes
                self.wire_meter.record_client_uplink(client_wire_bytes)

                # 3. Differential privacy: client-level clipping (and optional local noise)
                if self.dp_clipper is not None:
                    if getattr(self.dp_clipper, "mode", "central") == "local":
                        compressed_delta, _ = self.dp_clipper.apply(compressed_delta)
                    else:
                        compressed_delta, _ = self.dp_clipper.clip_only(compressed_delta)

                processed_updates[cid] = compressed_delta
                sample_counts.append(res.num_samples)

            # 4. Secure aggregation simulation (pairwise zero-sum masking)
            if self.secagg is not None:
                processed_updates = self.secagg.mask_updates(processed_updates)

            # 5. Privacy accounting step (Client-Level DP: q = m / K)
            current_eps = 0.0
            if self.dp_accountant is not None and self.dp_clipper is not None:
                sample_rate = len(selected_clients) / float(num_total_clients)
                self.dp_accountant.step(
                    noise_multiplier=self.dp_clipper.noise_multiplier,
                    sample_rate=sample_rate,
                )
                current_eps = self.dp_accountant.get_epsilon()

            # Server aggregation rule
            update_vectors = [processed_updates[cid] for cid in selected_clients]
            aggregated_delta = self.server_rule.aggregate(
                updates=update_vectors,
                weights=[float(s) for s in sample_counts],
            )

            # Central Client-Level DP noise addition on the aggregated update
            if (
                self.dp_clipper is not None
                and getattr(self.dp_clipper, "mode", "central") == "central"
                and self.dp_clipper.noise_multiplier > 0.0
            ):
                aggregated_delta = self.dp_clipper.add_central_noise_to_average(
                    aggregated_avg_delta=aggregated_delta,
                    num_sampled_clients=len(selected_clients),
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

            # Global evaluation
            eval_metrics = self.evaluate_global(eval_criterion, compute_auprc)

            round_duration = time.time() - round_start
            round_record = {
                "round": round_idx,
                "loss": eval_metrics["loss"],
                "accuracy": eval_metrics["accuracy"],
                "macro_f1": eval_metrics["macro_f1"],
                "round_time": round_duration,
                "epsilon": current_eps,
                "uplink_bytes": round_uplink_bytes,
                "downlink_bytes": round_downlink_bytes,
                "total_bytes": self.wire_meter.total_bytes,
            }
            if compute_auprc:
                round_record["auprc"] = eval_metrics["auprc"]

            history.append(round_record)

            if verbose:
                acc_pct = eval_metrics["accuracy"] * 100.0
                f1 = eval_metrics["macro_f1"]
                eps_str = f" | Eps: {current_eps:.2f}" if self.dp_accountant is not None else ""
                kb_str = f" | Net: {round_uplink_bytes / 1024:.1f} KB"
                print(
                    f"Round {round_idx:02d}/{num_rounds:02d} | "
                    f"Loss: {eval_metrics['loss']:.4f} | "
                    f"Acc: {acc_pct:.2f}% | "
                    f"F1: {f1:.4f}"
                    f"{eps_str}{kb_str} | "
                    f"Time: {round_duration:.2f}s"
                )

            clear_device_memory()

        total_time = time.time() - start_time
        return {
            "history": history,
            "final_metrics": history[-1] if history else {},
            "total_time": total_time,
            "device": str(self.device),
            "wire_summary": self.wire_meter.get_summary(),
        }
