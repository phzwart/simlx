"""Barlow Twins model for self-supervised feature learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from simlx.models.base import BaseModel


@dataclass
class BarlowTwinsConfig:
    """Configuration for :class:`BarlowTwins` model."""

    encoder_name: str = "resnet18"
    encoder_output_dim: int = 512
    projection_dim: int = 128
    projection_hidden_dim: int = 2048
    use_batch_norm: bool = True
    use_bias: bool = False


class ProjectionHead(nn.Module):
    """MLP projection head for Barlow Twins."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_batch_norm: bool = True,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.bn2 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dimensions if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def _create_encoder(encoder_name: str, encoder_output_dim: int) -> nn.Module:
    """Create encoder backbone from name."""
    encoder_name_lower = encoder_name.lower()

    if encoder_name_lower.startswith("resnet"):
        try:
            import torchvision.models as models  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for ResNet encoders. "
                "Install it with: pip install torchvision"
            ) from exc

        if encoder_name_lower == "resnet18":
            encoder = models.resnet18(weights=None)
            encoder.fc = nn.Identity()
            # Override with custom output dimension if needed
            if encoder_output_dim != 512:
                encoder.fc = nn.Linear(512, encoder_output_dim)
        elif encoder_name_lower == "resnet34":
            encoder = models.resnet34(weights=None)
            encoder.fc = nn.Identity()
            if encoder_output_dim != 512:
                encoder.fc = nn.Linear(512, encoder_output_dim)
        elif encoder_name_lower == "resnet50":
            encoder = models.resnet50(weights=None)
            encoder.fc = nn.Identity()
            if encoder_output_dim != 2048:
                encoder.fc = nn.Linear(2048, encoder_output_dim)
        else:
            raise ValueError(f"Unsupported ResNet variant: {encoder_name}")
        return encoder

    raise ValueError(f"Unsupported encoder: {encoder_name}")


class BarlowTwins(BaseModel, nn.Module):
    """Barlow Twins model with encoder and projection head."""

    def __init__(self, config: BarlowTwinsConfig | None = None) -> None:
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.config = config or BarlowTwinsConfig()

        self.encoder = _create_encoder(self.config.encoder_name, self.config.encoder_output_dim)
        self.projection_head = ProjectionHead(
            input_dim=self.config.encoder_output_dim,
            hidden_dim=self.config.projection_hidden_dim,
            output_dim=self.config.projection_dim,
            use_batch_norm=self.config.use_batch_norm,
            use_bias=self.config.use_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and projection head."""
        features = self.encoder(x)
        # Handle different encoder output shapes
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        projections = self.projection_head(features)
        return projections

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature representation (without projection)."""
        features = self.encoder(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return features

    def state_dict(self) -> dict[str, torch.Tensor]:
        return nn.Module.state_dict(self)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:  # type: ignore[override]
        nn.Module.load_state_dict(self, state)

