"""
Unit tests for Stage 1 Server Optimizers:
- ServerSGD (lr, momentum)
- ServerAdam (moments, bias correction)
"""

import pytest
import torch

from flaas.fl.server_optimizers import ServerSGD, ServerAdam


def test_server_sgd():
    """Test server SGD without and with momentum."""
    # Without momentum
    opt_simple = ServerSGD(lr=0.5, momentum=0.0)
    w = torch.tensor([10.0, 20.0])
    delta = torch.tensor([2.0, -4.0])

    w_new = opt_simple.step(w, delta)
    # w + 0.5 * delta = [10+1, 20-2] = [11.0, 18.0]
    expected = torch.tensor([11.0, 18.0])
    assert torch.allclose(w_new, expected)

    # With momentum
    opt_mom = ServerSGD(lr=1.0, momentum=0.5)
    w_0 = torch.zeros(2)
    delta_1 = torch.tensor([1.0, 2.0])
    w_1 = opt_mom.step(w_0, delta_1)
    # velocity_1 = delta_1 = [1, 2], w_1 = [1, 2]
    assert torch.allclose(w_1, torch.tensor([1.0, 2.0]))

    delta_2 = torch.tensor([3.0, 4.0])
    w_2 = opt_mom.step(w_1, delta_2)
    # velocity_2 = 0.5 * [1, 2] + [3, 4] = [3.5, 5.0]
    # w_2 = [1, 2] + [3.5, 5.0] = [4.5, 7.0]
    assert torch.allclose(w_2, torch.tensor([4.5, 7.0]))


def test_server_adam():
    """Test server Adam optimizer tracks moments and updates weights."""
    opt = ServerAdam(lr=0.1, beta1=0.9, beta2=0.99, eps=1e-6)
    w = torch.zeros(2)
    delta = torch.tensor([1.0, -1.0])

    w_next = opt.step(w, delta)
    assert opt.t == 1
    assert opt.m is not None and opt.v is not None
    # Weights must update in positive/negative directions according to delta
    assert w_next[0] > 0
    assert w_next[1] < 0

    opt.reset()
    assert opt.t == 0
    assert opt.m is None
