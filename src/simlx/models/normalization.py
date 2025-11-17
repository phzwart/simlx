"""Normalization layers for neural networks.

This module provides normalization layers including QuantileNorm2d, a quantile-based
normalization layer designed to work alongside BatchNorm for distribution refinement.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileNorm2d(nn.Module):
    """Quantile Normalization for 2D feature maps (BatchNorm cousin).

    Tracks sparse quantiles per channel and maps features to target distribution
    via differentiable quantile transformation. Designed to follow BatchNorm
    for distribution refinement.

    Args:
        num_features: Number of channels (C)
        num_quantiles: Number of quantiles to track (default: 5)
        percentiles: Which percentiles to track (default: [2, 25, 50, 75, 98])
        target_dist: Target distribution ('gaussian' or 'uniform')
        momentum: EMA momentum for quantile updates (default: 0.01)
        temperature: Softmax temperature for interpolation (default: 0.05)
        eps: Small constant for numerical stability (default: 1e-5)
        affine: Whether to learn affine transformation after normalization
        track_running_stats: Whether to track running quantiles (like BatchNorm)

    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)

    Examples:
        >>> # Drop-in after BatchNorm
        >>> norm = nn.Sequential(
        >>>     nn.BatchNorm2d(256),
        >>>     QuantileNorm2d(256)
        >>> )
        >>> x = torch.randn(32, 256, 64, 64)
        >>> y = norm(x)
    """

    def __init__(
        self,
        num_features: int,
        num_quantiles: int = 5,
        percentiles: list[float] | None = None,
        target_dist: str = "gaussian",
        momentum: float = 0.01,
        temperature: float = 0.05,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_quantiles = num_quantiles
        self.momentum = momentum
        self.temperature = temperature
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Default percentiles: [2, 25, 50, 75, 98]
        if percentiles is None:
            if num_quantiles == 5:
                percentiles = [2, 25, 50, 75, 98]
            elif num_quantiles == 7:
                percentiles = [2, 16, 33, 50, 67, 84, 98]
            else:
                percentiles = list(torch.linspace(2, 98, num_quantiles).tolist())

        if len(percentiles) != num_quantiles:
            raise ValueError(f"Number of percentiles ({len(percentiles)}) must match num_quantiles ({num_quantiles})")

        self.register_buffer("percentiles", torch.tensor(percentiles, dtype=torch.float32))

        # Target distribution quantiles
        self.target_dist = target_dist
        self.register_buffer("target_quantiles", self._get_target_quantiles(target_dist, percentiles))

        if self.track_running_stats:
            # Running quantiles: [C, Q]
            # Initialize to Gaussian (assumes input from BatchNorm)
            init_quantiles = self._get_target_quantiles("gaussian", percentiles)
            self.register_buffer(
                "running_quantiles",
                init_quantiles.unsqueeze(0).repeat(num_features, 1),
            )

            # Running variance for blank detection: [C]
            self.register_buffer("running_variance", torch.ones(num_features))

            # Number of batches tracked
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_quantiles", None)
            self.register_buffer("running_variance", None)
            self.register_buffer("num_batches_tracked", None)

        # Affine parameters (like BatchNorm's weight and bias)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        if self.track_running_stats:
            init_quantiles = self._get_target_quantiles("gaussian", self.percentiles.tolist())
            self.running_quantiles.fill_(0.0)
            self.running_quantiles.copy_(init_quantiles.unsqueeze(0).repeat(self.num_features, 1))
            self.running_variance.fill_(1.0)
            self.num_batches_tracked.zero_()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def _get_target_quantiles(self, dist: str, percentiles: list[float]) -> torch.Tensor:
        """Get quantiles for target distribution."""
        p = torch.tensor(percentiles, dtype=torch.float32) / 100.0

        if dist == "gaussian":
            # Standard normal quantiles
            # Using inverse error function: Φ^(-1)(p) = √2 * erf^(-1)(2p - 1)
            quantiles = torch.erfinv(2 * p - 1) * 1.4142135623730951  # √2
        elif dist == "uniform":
            # Uniform in [-1, 1]
            quantiles = 2 * p - 1
        else:
            raise ValueError(f"Unknown target distribution: {dist}")

        return quantiles

    def _compute_batch_quantiles(self, x: torch.Tensor, channel: int) -> torch.Tensor:
        """Compute quantiles for a single channel across batch and spatial dims.

        Args:
            x: [B, H, W] tensor for one channel

        Returns:
            quantiles: [Q] tensor
        """
        x_flat = x.reshape(-1)  # Flatten batch and spatial

        # Use torch.quantile (differentiable as of PyTorch 1.7+)
        quantiles = torch.quantile(x_flat, self.percentiles / 100.0, interpolation="linear")

        return quantiles

    def _update_running_stats(self, x: torch.Tensor) -> None:
        """Update running quantiles and variance (EMA-based streaming)."""
        if not self.training or not self.track_running_stats:
            return

        _B, C, _H, _W = x.shape

        # Exponential moving average momentum
        momentum = self.momentum

        with torch.no_grad():
            for c in range(C):
                x_c = x[:, c, :, :]  # [B, H, W]

                # Compute batch statistics
                batch_var = x_c.var().item()

                # Update running variance
                self.running_variance[c] = (1 - momentum) * self.running_variance[c] + momentum * batch_var

                # Skip quantile update for nearly-uniform channels
                if batch_var < self.eps:
                    continue

                # Compute batch quantiles
                batch_quantiles = self._compute_batch_quantiles(x_c, c)

                # Update running quantiles (EMA)
                self.running_quantiles[c] = (1 - momentum) * self.running_quantiles[c] + momentum * batch_quantiles

            self.num_batches_tracked += 1

    def _quantile_transform(
        self,
        x: torch.Tensor,
        source_quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        variance: float,
    ) -> torch.Tensor:
        """Transform values through quantile mapping (differentiable).

        Args:
            x: [B, H, W] values to transform
            source_quantiles: [Q] empirical quantiles
            target_quantiles: [Q] target distribution quantiles
            variance: channel variance (for blank detection)

        Returns:
            x_transformed: [B, H, W]
        """
        # Handle blank/uniform channels
        if variance < self.eps:
            # Pass through unchanged (or could apply simple standardization)
            return x

        B, H, W = x.shape
        x_flat = x.reshape(-1, 1)  # [B*H*W, 1]

        # Compute soft distances to each quantile
        source_quantiles = source_quantiles.unsqueeze(0)  # [1, Q]
        distances = torch.abs(x_flat - source_quantiles)  # [B*H*W, Q]

        # Soft weights via temperature-scaled softmax (differentiable!)
        weights = F.softmax(-distances / self.temperature, dim=-1)  # [B*H*W, Q]

        # Interpolate percentiles (find empirical percentile rank)
        (weights * self.percentiles.unsqueeze(0)).sum(dim=-1)  # [B*H*W]

        # Map percentiles to target distribution via interpolation
        # Linear interpolation through target quantiles
        target_quantiles = target_quantiles.unsqueeze(0)  # [1, Q]
        x_transformed = (weights * target_quantiles).sum(dim=-1)  # [B*H*W]

        return x_transformed.reshape(B, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Normalized tensor [B, C, H, W]
        """
        _B, C, _H, _W = x.shape

        if self.num_features != C:
            raise ValueError(f"Expected {self.num_features} channels, got {C}")

        # Update running statistics in training mode
        if self.training:
            self._update_running_stats(x)

        # Use running stats (train) or fixed stats (eval)
        if self.track_running_stats:
            quantiles = self.running_quantiles  # [C, Q]
            variances = self.running_variance  # [C]
        else:
            # Compute batch quantiles on-the-fly (no tracking)
            quantiles = torch.stack([self._compute_batch_quantiles(x[:, c], c) for c in range(C)])  # [C, Q]
            variances = x.var(dim=(0, 2, 3))  # [C]

        # Apply quantile transformation per channel
        x_normalized = []
        for c in range(C):
            x_c = x[:, c, :, :]  # [B, H, W]

            x_c_norm = self._quantile_transform(
                x_c,
                source_quantiles=quantiles[c],
                target_quantiles=self.target_quantiles,
                variance=variances[c].item(),
            )

            x_normalized.append(x_c_norm.unsqueeze(1))

        x_normalized = torch.cat(x_normalized, dim=1)  # [B, C, H, W]

        # Apply affine transformation (like BatchNorm)
        if self.affine:
            x_normalized = x_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        return x_normalized

    def extra_repr(self) -> str:
        """String representation (like BatchNorm)."""
        return (
            f"{self.num_features}, "
            f"quantiles={self.num_quantiles}, "
            f"target={self.target_dist}, "
            f"momentum={self.momentum}, "
            f"affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}"
        )


def CascadedNorm2d(
    num_features: int,
    bn_momentum: float = 0.1,
    bn_affine: bool = True,
    bn_track_running_stats: bool = True,
    **kwargs: Any,
) -> nn.Sequential:
    """BatchNorm → QuantileNorm cascade (recommended usage).

    Args:
        num_features: Number of channels
        bn_momentum: Momentum for BatchNorm (default: 0.1)
        bn_affine: Whether BatchNorm has learnable affine parameters (default: True)
        bn_track_running_stats: Whether BatchNorm tracks running stats (default: True)
        **kwargs: Additional arguments for QuantileNorm2d

    Returns:
        Sequential module with BatchNorm → QuantileNorm

    Example:
        >>> norm = CascadedNorm2d(256)
        >>> x = torch.randn(32, 256, 64, 64)
        >>> y = norm(x)
    """
    return nn.Sequential(
        nn.BatchNorm2d(
            num_features,
            momentum=bn_momentum,
            affine=bn_affine,
            track_running_stats=bn_track_running_stats,
        ),
        QuantileNorm2d(num_features, **kwargs),
    )
