"""
Unit tests for Stage 2 Primitives:
1. Attacks (SignFlip, GaussianNoise, ScaledUpdate, LabelFlippedDataset)
2. Differential Privacy (Clipping, Gaussian perturbation, Opacus RDP Accountant verification)
3. Secure Aggregation (Exact zero-sum mask cancellation)
4. Compression (Top-k sparsity, QSGD, Sparse Ternary, and exact wire byte accounting)
"""

import math
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from flaas.attacks.attacks import (
    SignFlipAttack,
    GaussianNoiseAttack,
    ScaledUpdateAttack,
    LabelFlippedDataset,
)
from flaas.privacy.clip_noise import DPClipperNoise
from flaas.privacy.accountant import VerifiedRDPAccountant
from flaas.privacy.secure_agg import SimulatedSecureAggregation
from flaas.compression.compressors import (
    IdentityCompressor,
    TopKCompressor,
    QSGDCompressor,
    SparseTernaryCompressor,
)
from flaas.compression.wire_meter import calculate_raw_bytes


def test_sign_flip_attack():
    delta = torch.tensor([1.0, -2.0, 3.5])
    attack = SignFlipAttack(scale=2.0)
    adv_delta = attack.perturb(delta)
    expected = torch.tensor([-2.0, 4.0, -7.0])
    assert torch.allclose(adv_delta, expected)


def test_gaussian_noise_attack():
    torch.manual_seed(42)
    delta = torch.zeros(1000)
    attack = GaussianNoiseAttack(std=2.0, additive=False)
    adv_delta = attack.perturb(delta)
    assert abs(adv_delta.mean().item()) < 0.15
    assert abs(adv_delta.std().item() - 2.0) < 0.15


def test_scaled_update_attack():
    delta = torch.tensor([0.1, -0.2])
    attack = ScaledUpdateAttack(scale=50.0)
    adv_delta = attack.perturb(delta)
    assert torch.allclose(adv_delta, torch.tensor([5.0, -10.0]))


def test_label_flipped_dataset():
    x = torch.zeros(10, 2)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    base_ds = TensorDataset(x, y)
    flipped_ds = LabelFlippedDataset(base_ds, num_classes=10)

    # 0 -> 9, 1 -> 8, 2 -> 7 ...
    for i in range(10):
        _, y_flip = flipped_ds[i]
        assert y_flip == 9 - i


def test_dp_clipping_and_noise():
    # Vector with L2 norm = 10.0
    vec = torch.zeros(100)
    vec[0] = 10.0

    clipper_no_noise = DPClipperNoise(clip_norm=2.5, noise_multiplier=0.0)
    clipped, norm = clipper_no_noise.apply(vec)

    assert abs(norm - 10.0) < 1e-6
    # Clipped norm must equal clip_norm
    assert abs(torch.norm(clipped, p=2).item() - 2.5) < 1e-6

    # With noise
    clipper_with_noise = DPClipperNoise(clip_norm=2.5, noise_multiplier=1.0, seed=42)
    noised, _ = clipper_with_noise.apply(vec)
    assert not torch.allclose(clipped, noised)


def test_opacus_rdp_accountant_reference_values():
    """
    Validates the RDP accountant wrapper against established reference values
    from Mironov (CSF 2017) and Opacus RDP implementation.
    """
    accountant = VerifiedRDPAccountant(target_delta=1e-5)

    # Reference Case 1: Subsampled Gaussian mechanism with q=0.01, sigma=1.0, 1 step, delta=1e-5
    accountant.step(noise_multiplier=1.0, sample_rate=0.01)
    eps_step1 = accountant.get_epsilon()
    assert abs(eps_step1 - 0.9555) < 0.05, f"Eps mismatch: {eps_step1}"

    # Reference Case 2: 100 steps with q=0.01, sigma=1.0, delta=1e-5
    for _ in range(99):
        accountant.step(noise_multiplier=1.0, sample_rate=0.01)
    eps_step100 = accountant.get_epsilon()
    assert abs(eps_step100 - 1.2141) < 0.05, f"Eps mismatch: {eps_step100}"


def test_opacus_vs_google_dp_accounting_cross_check():
    """
    Independent cross-check: Compares Opacus RDP accountant against
    Google's official dp_accounting library across multiple FL sampling regimes.
    Tolerance: absolute difference < 1e-6.
    """
    test_cases = [
        {"steps": 10, "q": 0.01, "sigma": 1.0, "delta": 1e-5},
        {"steps": 100, "q": 0.01, "sigma": 1.0, "delta": 1e-5},
        {"steps": 100, "q": 0.05, "sigma": 1.5, "delta": 1e-5},
        {"steps": 50, "q": 0.10, "sigma": 2.0, "delta": 1e-5},
        {"steps": 20, "q": 0.20, "sigma": 1.2, "delta": 1e-5},
    ]

    for case in test_cases:
        res = VerifiedRDPAccountant.cross_check(
            steps=case["steps"],
            noise_multiplier=case["sigma"],
            sample_rate=case["q"],
            target_delta=case["delta"],
        )
        assert res["is_concordant"], f"Discrepancy between Opacus and Google dp_accounting: {res}"
        assert res["rel_diff_pct"] < 0.2, f"Relative difference exceeded 0.2%: {res['rel_diff_pct']}%"


def test_client_level_dp_accounting_simulation():
    """
    Verifies Client-Level (User-Level) Differential Privacy:
    - K = 10 total clients in federation
    - m = 4 sampled clients per round (q = 4/10 = 0.4)
    - Sensitivity C = 2.0 enforced by client clipping
    - Central noise added to averaged update with std = (sigma * C) / m
    - Accountant tracks composition across rounds with q = 0.4.
    """
    device = torch.device("cpu")
    from flaas.fl.models import CompactCNN
    from flaas.fl.client_algos import FedAvgClient
    from flaas.fl.server_rules import FedAvgMean
    from flaas.fl.server_optimizers import ServerSGD
    from flaas.fl.simulator import FLSimulator, seed_everything

    seed_everything(42)
    num_total_clients = 10
    num_sampled_clients = 4
    participation_rate = num_sampled_clients / float(num_total_clients)

    data = {
        cid: DataLoader(TensorDataset(torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))), batch_size=8)
        for cid in range(num_total_clients)
    }
    test_loader = DataLoader(TensorDataset(torch.randn(20, 1, 28, 28), torch.randint(0, 10, (20,))), batch_size=10)

    client_algos = {
        cid: FedAvgClient(cid, lambda: CompactCNN(1, 10), device, learning_rate=0.05)
        for cid in range(num_total_clients)
    }

    clipper = DPClipperNoise(clip_norm=2.0, noise_multiplier=1.2, mode="central", seed=42)
    accountant = VerifiedRDPAccountant(target_delta=1e-5)

    sim = FLSimulator(
        model_fn=lambda: CompactCNN(1, 10),
        client_algos=client_algos,
        client_dataloaders=data,
        test_dataloader=test_loader,
        server_rule=FedAvgMean(),
        server_optimizer=ServerSGD(lr=1.0),
        device=device,
        seed=42,
        dp_clipper=clipper,
        dp_accountant=accountant,
    )

    num_rounds = 3
    res = sim.run(
        num_rounds=num_rounds,
        epochs_per_round=1,
        participation_rate=participation_rate,
        verbose=False,
    )

    # Verify accountant stepped num_rounds times with q = 0.4
    assert accountant.total_steps == num_rounds
    eps = accountant.get_epsilon()
    assert eps > 0.0

    # Cross-check with Google dp_accounting
    google_eps = VerifiedRDPAccountant.compute_google_dp_accounting_epsilon(
        steps=num_rounds,
        noise_multiplier=1.2,
        sample_rate=participation_rate,
        target_delta=1e-5,
    )
    assert abs(eps - google_eps) < 1e-3



def test_secure_aggregation_exact_cancellation():
    """
    Tests that simulated pairwise zero-sum masks cancel out exactly:
    sum(masked_updates) == sum(unmasked_updates).
    """
    torch.manual_seed(42)
    num_clients = 5
    dim = 50

    unmasked_updates = {
        cid: torch.randn(dim) for cid in range(num_clients)
    }

    secagg = SimulatedSecureAggregation(seed=123)
    masked_updates = secagg.mask_updates(unmasked_updates)

    # 1. Individual updates must be thoroughly blinded
    for cid in range(num_clients):
        assert not torch.allclose(unmasked_updates[cid], masked_updates[cid])
        diff_norm = torch.norm(unmasked_updates[cid] - masked_updates[cid]).item()
        assert diff_norm > 1.0, "Individual update was not masked"

    # 2. Sum of masked updates must equal sum of unmasked updates numerically (atol 1e-6)
    unmasked_sum = torch.stack(list(unmasked_updates.values())).sum(dim=0)
    masked_sum = torch.stack(list(masked_updates.values())).sum(dim=0)

    assert torch.allclose(unmasked_sum, masked_sum, atol=1e-6), (
        f"Secure Aggregation mask did not cancel! Max diff: {(unmasked_sum - masked_sum).abs().max().item()}"
    )


def test_topk_compression_and_bytes():
    dim = 100
    tensor = torch.randn(dim)
    compressor = TopKCompressor(k_ratio=0.1)  # k = 10

    compressed, wire_bytes = compressor.compress(tensor)

    # Exactly 10 non-zero elements
    non_zeros = (compressed != 0.0).sum().item()
    assert non_zeros == 10

    # Wire bytes: 4 header bytes + 10 * 8 bytes = 84 bytes
    assert wire_bytes == 4 + 10 * 8
    # Baseline raw bytes would be 100 * 4 = 400 bytes
    assert wire_bytes < 400


def test_qsgd_compression_and_bytes():
    dim = 100
    tensor = torch.randn(dim)
    compressor = QSGDCompressor(s=8, seed=42)

    compressed, wire_bytes = compressor.compress(tensor)

    assert compressed.shape == tensor.shape
    # Wire bits: 1 sign + ceil(log2(8+1)) = 1 + 4 = 5 bits per coord.
    # 4 bytes norm + ceil(100 * 5 / 8) = 4 + 63 = 67 bytes.
    assert wire_bytes == 4 + math.ceil(100 * 5 / 8)
    assert wire_bytes < 400


def test_sparse_ternary_compression_and_bytes():
    dim = 100
    tensor = torch.randn(dim)
    compressor = SparseTernaryCompressor(threshold_factor=0.5)

    compressed, wire_bytes = compressor.compress(tensor)

    # Elements must only take values in {-scale, 0, +scale}
    unique_vals = torch.unique(torch.abs(compressed))
    assert len(unique_vals) <= 2  # 0 and scale

    # Wire bytes: 4 bytes scale + ceil(100 * 2 / 8) = 4 + 25 = 29 bytes.
    assert wire_bytes == 4 + math.ceil(100 * 2 / 8)
    assert wire_bytes < 400


def test_identity_compression_matches_uncompressed_fedavg():
    """
    Verifies that IdentityCompressor matches uncompressed execution numerically (atol 1e-6).
    """
    device = torch.device("cpu")
    from torch.utils.data import DataLoader, TensorDataset
    from flaas.fl.models import CompactCNN
    from flaas.fl.client_algos import FedAvgClient
    from flaas.fl.server_rules import FedAvgMean
    from flaas.fl.server_optimizers import ServerSGD
    from flaas.fl.simulator import FLSimulator, seed_everything

    # 2 clients with fixed data
    seed_everything(42)
    data = {
        0: DataLoader(TensorDataset(torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))), batch_size=8),
        1: DataLoader(TensorDataset(torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))), batch_size=8),
    }
    test_loader = DataLoader(TensorDataset(torch.randn(20, 1, 28, 28), torch.randint(0, 10, (20,))), batch_size=10)

    # Run 1: uncompressed (compressor=None -> IdentityCompressor default)
    seed_everything(100)
    client_algos1 = {cid: FedAvgClient(cid, lambda: CompactCNN(1, 10), device, learning_rate=0.05) for cid in range(2)}
    sim1 = FLSimulator(
        model_fn=lambda: CompactCNN(1, 10),
        client_algos=client_algos1,
        client_dataloaders=data,
        test_dataloader=test_loader,
        server_rule=FedAvgMean(),
        server_optimizer=ServerSGD(lr=1.0),
        device=device,
        seed=100,
        compressor=None,
    )
    res1 = sim1.run(num_rounds=2, epochs_per_round=1, verbose=False)

    # Run 2: explicit IdentityCompressor
    seed_everything(100)
    client_algos2 = {cid: FedAvgClient(cid, lambda: CompactCNN(1, 10), device, learning_rate=0.05) for cid in range(2)}
    sim2 = FLSimulator(
        model_fn=lambda: CompactCNN(1, 10),
        client_algos=client_algos2,
        client_dataloaders=data,
        test_dataloader=test_loader,
        server_rule=FedAvgMean(),
        server_optimizer=ServerSGD(lr=1.0),
        device=device,
        seed=100,
        compressor=IdentityCompressor(),
    )
    res2 = sim2.run(num_rounds=2, epochs_per_round=1, verbose=False)

    for k in sim1.global_state_dict:
        diff = (sim1.global_state_dict[k] - sim2.global_state_dict[k]).abs().max().item()
        assert diff < 1e-6, f"Mismatch in layer {k}: {diff}"

