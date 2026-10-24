"""
Privacy Accountant wrapper utilizing Opacus verified Rényi Differential Privacy (RDP).
Provides exact epsilon computation for subsampled Gaussian mechanisms in Federated Learning.
"""

from typing import Tuple, Optional
import opacus
from opacus.accountants import RDPAccountant


class VerifiedRDPAccountant:
    """
    RDP Accountant wrapper using Opacus's mathematically audited implementation.
    Tracks privacy budget spend across federated communication rounds.
    """

    def __init__(self, target_delta: float = 1e-5):
        if target_delta <= 0 or target_delta >= 1:
            raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")
        self.target_delta = target_delta
        self.accountant = RDPAccountant()
        self.total_steps = 0

    def step(self, noise_multiplier: float, sample_rate: float) -> None:
        """
        Records one mechanism application (e.g. one FL communication round).
        noise_multiplier: sigma (ratio of standard deviation to L2 clip norm C).
        sample_rate: q = m / K (client sampling ratio) or sub-sampling ratio.
        """
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be strictly positive for DP accounting.")
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")

        self.accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        self.total_steps += 1

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Computes (epsilon, delta)-DP privacy budget spent so far.
        """
        if self.total_steps == 0:
            return 0.0
        d = delta if delta is not None else self.target_delta
        eps = float(self.accountant.get_epsilon(delta=d))
        return eps

    @staticmethod
    def compute_offline_epsilon(
        steps: int,
        noise_multiplier: float,
        sample_rate: float,
        target_delta: float = 1e-5,
    ) -> float:
        """
        Direct static computation of epsilon for a planned budget of steps.
        """
        if steps == 0 or noise_multiplier <= 0:
            return 0.0
        acc = RDPAccountant()
        for _ in range(steps):
            acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        return float(acc.get_epsilon(delta=target_delta))
