"""Tests for RandomGaussianProjection."""

from __future__ import annotations

import math
import statistics

import pytest
import torch

from simlx.models.random_projection import RandomGaussianProjection


def test_random_projection_forward() -> None:
    proj = RandomGaussianProjection(in_features=512, out_features=64)
    x = torch.randn(2, 512, 32, 32)

    proj.train()
    y = proj(x)
    assert y.shape == (2, 64, 32, 32)

    proj.eval()
    with pytest.raises(RuntimeError):
        proj(x)


def test_no_gradients() -> None:
    proj = RandomGaussianProjection(512, 64)
    assert len(list(proj.parameters())) == 0

    x = torch.randn(2, 512, 32, 32, requires_grad=True)
    y = proj(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    if proj.weight is not None:
        assert proj.weight.grad is None


def test_batch_level_sampling() -> None:
    proj = RandomGaussianProjection(128, 32)
    proj.train()
    x = torch.randn(1, 128, 16, 16)
    y1 = proj(x)
    y2 = proj(x)

    assert not torch.allclose(y1, y2)
    assert y1.shape == (1, 32, 16, 16)
    assert y2.shape == (1, 32, 16, 16)


def test_set_projection() -> None:
    proj = RandomGaussianProjection(256, 64)
    x = torch.randn(2, 256, 32, 32)

    fixed_weight = torch.randn(64, 256) / math.sqrt(64)
    proj.set_projection(fixed_weight)

    proj.eval()
    y1 = proj(x)
    y2 = proj(x)
    assert torch.allclose(y1, y2)


def test_norm_preservation() -> None:
    proj = RandomGaussianProjection(512, 64)
    proj.train()

    grad_ratios: list[float] = []
    for _ in range(100):
        x = torch.randn(4, 512, 16, 16, requires_grad=True)
        y = proj(x)
        loss = (y**2).sum()
        loss.backward()

        grad_norm = x.grad.norm().item()
        input_norm = x.norm().item()
        grad_ratios.append(grad_norm / input_norm)

    expected = 2 * math.sqrt(512 / 64)
    actual = statistics.mean(grad_ratios)
    assert abs(actual - expected) < 0.5


def test_3d_input() -> None:
    proj = RandomGaussianProjection(128, 32)
    proj.train()
    x = torch.randn(2, 128, 16, 16, 16)

    y = proj(x)
    assert y.shape == (2, 32, 16, 16, 16)

