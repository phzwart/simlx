"""Random Gaussian projection module for Matryoshka U-Net.

Example:
    >>> module = RandomGaussianProjection(in_features=512, out_features=64)
    >>> module.train()
    >>> y = module(torch.randn(2, 512, 16, 16))
    >>> module.set_projection(torch.randn(64, 512) / math.sqrt(64))
    >>> module.eval()
    >>> y = module(torch.randn(2, 512, 16, 16))

    >>> # Fixed deterministic Gaussian projection
    >>> fixed_module = FixedGaussianProjection(in_features=512, out_features=64, seed=42)
    >>> y = fixed_module(torch.randn(2, 512, 16, 16))  # Same projection every time
"""

from __future__ import annotations

import math

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]


class RandomGaussianProjection(nn.Module):
    """Random Gaussian projection matrix sampler with optional fixed weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        if bias:
            raise ValueError("RandomGaussianProjection does not support bias in Stage 1.")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer("weight", None, persistent=False)
        self._is_fixed = False

    def _sample_projection(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.out_features)
        projection = torch.randn((self.out_features, self.in_features), device=device, dtype=dtype)
        projection.mul_(scale)
        return projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError("Input tensor must have at least 2 dimensions (B, C, ...).")
        if x.shape[1] != self.in_features:
            raise ValueError(f"Expected input with {self.in_features} channels, got {x.shape[1]}.")

        if self.training:
            weight = self._sample_projection(x.device, x.dtype)
            self.weight = weight
            self._is_fixed = False
        else:
            if self.weight is None or not self._is_fixed:
                raise RuntimeError(
                    "RandomGaussianProjection in eval mode requires fixed projection. "
                    "Call set_projection() after training or use model.train()."
                )
            weight = self.weight

        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]
        x_flat = x.flatten(2)
        projected = torch.einsum("oc,bci->boi", weight, x_flat)
        output = projected.view(batch_size, self.out_features, *spatial_shape)
        return output

    def set_projection(self, weight: torch.Tensor) -> None:
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"weight must have shape ({self.out_features}, {self.in_features}), got {tuple(weight.shape)}."
            )
        if not weight.is_floating_point():
            raise TypeError("weight must be a floating point tensor.")
        weight = weight.detach().contiguous()
        self.weight = weight
        self._is_fixed = True

    def extra_repr(self) -> str:
        projection_type = "fixed" if self._is_fixed and self.weight is not None else "random"
        return f"in_features={self.in_features}, out_features={self.out_features}, projection={projection_type}"


class FixedGaussianProjection(nn.Module):
    """Fixed deterministic Gaussian projection matrix with no gradients.

    Generates a Gaussian projection matrix using a fixed seed, ensuring the same
    projection is used across all networks. The projection matrix is registered
    as a buffer (not a parameter) and requires_grad=False, so it does not receive
    gradients during training.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        seed: Random seed for generating the projection matrix (default: 42)
        scale: Scaling factor for the projection (default: 1/sqrt(out_features))

    Example:
        >>> module = FixedGaussianProjection(in_features=512, out_features=64, seed=42)
        >>> x = torch.randn(2, 512, 16, 16)
        >>> y = module(x)  # Same projection every time, no gradients
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        seed: int = 42,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.seed = seed

        if scale is None:
            scale = 1.0 / math.sqrt(self.out_features)
        self.scale = scale

        # Generate fixed projection matrix
        projection = self._generate_projection()

        # Register as buffer (not parameter) with requires_grad=False
        self.register_buffer("weight", projection, persistent=True)

    def _generate_projection(self) -> torch.Tensor:
        """Generate deterministic Gaussian projection matrix."""
        # Save current RNG state
        rng_state = torch.get_rng_state()

        try:
            # Set fixed seed for deterministic generation
            torch.manual_seed(self.seed)
            projection = torch.randn(
                (self.out_features, self.in_features),
                dtype=torch.float32,
            )
            projection.mul_(self.scale)
        finally:
            # Restore RNG state
            torch.set_rng_state(rng_state)

        return projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fixed projection matrix.

        Args:
            x: Input tensor of shape (B, in_features, H, W) or (B, in_features, ...)

        Returns:
            Projected tensor of shape (B, out_features, H, W) or (B, out_features, ...)
        """
        if x.dim() < 2:
            raise ValueError("Input tensor must have at least 2 dimensions (B, C, ...).")
        if x.shape[1] != self.in_features:
            raise ValueError(f"Expected input with {self.in_features} channels, got {x.shape[1]}.")

        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]
        x_flat = x.flatten(2)  # (B, in_features, H*W)
        projected = torch.einsum("oc,bci->boi", self.weight, x_flat)  # (B, out_features, H*W)
        output = projected.view(batch_size, self.out_features, *spatial_shape)
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"seed={self.seed}, "
            f"scale={self.scale:.6f}"
        )
