"""Configurable 2D U-Net with residual blocks and LoRA kernel modulation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from simlx.models.base import BaseModel


@dataclass
class UNet2DConfig:
    """Configuration parameters for :class:`UNet2D`."""

    in_channels: int = 3
    out_channels: int = 1
    feature_sizes: Sequence[int] = (64, 128, 256, 512)
    bottleneck_features: int = 1024
    kernel_size: int = 3
    padding: int = 1
    activation: str = "relu"
    dropout: float = 0.0
    lora_rank: int = 4
    lora_scale: float = 1.0
    lora_variational: bool = False
    upsample_mode: str = "bilinear"
    align_corners: bool = False


class LoRAConv2d(nn.Module):
    """Conv2d layer with additive LoRA-style low-rank kernel noise."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        rank: int = 4,
        scale: float = 1.0,
        variational: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.rank = rank
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.variational = variational

        weight_shape = self.conv.weight.shape
        flattened_dim = weight_shape[1] * weight_shape[2] * weight_shape[3]

        self.lora_down = nn.Parameter(torch.zeros(weight_shape[0], rank))
        self.lora_up_mu = nn.Parameter(torch.zeros(rank, flattened_dim))
        if variational:
            self.lora_up_logvar = nn.Parameter(torch.full((rank, flattened_dim), -10.0))
        else:
            self.register_parameter("lora_up_logvar", None)

        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up_mu)

    def extra_repr(self) -> str:
        scale = self.scale.detach().cpu().item()
        return f"rank={self.rank}, scale={scale:.3f}, variational={self.variational}"

    def lora_weight(self) -> torch.Tensor:
        if self.variational and self.lora_up_logvar is not None and self.training:
            std = torch.exp(0.5 * self.lora_up_logvar)
            noise = torch.randn_like(std)
            lora_up = self.lora_up_mu + noise * std
        else:
            lora_up = self.lora_up_mu

        delta = self.lora_down @ lora_up
        delta = delta.view_as(self.conv.weight)
        return delta * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv.weight + self.lora_weight()
        return F.conv2d(
            x,
            weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
        )


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'")


class ResidualBlock(nn.Module):
    """Two-layer residual block with LoRA-enabled convolutions."""

    def __init__(self, in_ch: int, out_ch: int, config: UNet2DConfig) -> None:
        super().__init__()
        self.needs_projection = in_ch != out_ch
        self.projection = (
            LoRAConv2d(
                in_ch,
                out_ch,
                kernel_size=1,
                padding=0,
                bias=False,
                rank=config.lora_rank,
                scale=config.lora_scale,
                variational=config.lora_variational,
            )
            if self.needs_projection
            else None
        )

        self.conv1 = LoRAConv2d(
            in_ch,
            out_ch,
            kernel_size=config.kernel_size,
            padding=config.padding,
            bias=True,
            rank=config.lora_rank,
            scale=config.lora_scale,
            variational=config.lora_variational,
        )
        self.conv2 = LoRAConv2d(
            out_ch,
            out_ch,
            kernel_size=config.kernel_size,
            padding=config.padding,
            bias=True,
            rank=config.lora_rank,
            scale=config.lora_scale,
            variational=config.lora_variational,
        )

        self.activation = get_activation(config.activation)
        self.dropout = nn.Dropout2d(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.needs_projection and self.projection is not None:
            identity = self.projection(identity)

        out = out + identity
        out = self.activation(out)
        return out


class DownBlock(nn.Module):
    """Encoder step returning the residual block output and pooled tensor."""

    def __init__(self, in_ch: int, out_ch: int, config: UNet2DConfig) -> None:
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch, config)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.block(x)
        return out, self.pool(out)


class UpBlock(nn.Module):
    """Decoder step combining interpolation upsampling with a residual block."""

    def __init__(self, in_ch: int, out_ch: int, config: UNet2DConfig) -> None:
        super().__init__()
        self.config = config
        self.block = ResidualBlock(in_ch, out_ch, config)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upscaled = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode=self.config.upsample_mode,
            align_corners=self.config.align_corners,
        )
        concatenated = torch.cat([upscaled, skip], dim=1)
        return self.block(concatenated)


class UNet2D(BaseModel, nn.Module):
    """Configurable 2D U-Net with residual connections and LoRA convolutions."""

    def __init__(self, config: UNet2DConfig | None = None) -> None:
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.config = config or UNet2DConfig()

        features = list(self.config.feature_sizes)
        if not features:
            raise ValueError("feature_sizes must contain at least one value")

        self.input_block = ResidualBlock(
            self.config.in_channels,
            features[0],
            self.config,
        )

        self.down_blocks = nn.ModuleList()
        for idx in range(len(features) - 1):
            self.down_blocks.append(DownBlock(features[idx], features[idx + 1], self.config))

        self.bottleneck = ResidualBlock(
            features[-1],
            self.config.bottleneck_features,
            self.config,
        )

        reversed_features = list(reversed(features))
        decoder_in_channels = [
            self.config.bottleneck_features + reversed_features[0],
            *[reversed_features[i] + reversed_features[i + 1] for i in range(len(reversed_features) - 1)],
        ]

        self.up_blocks = nn.ModuleList()
        for idx, out_ch in enumerate(reversed_features):
            in_ch = decoder_in_channels[idx]
            self.up_blocks.append(UpBlock(in_ch, out_ch, self.config))

        self.output_conv = LoRAConv2d(
            reversed_features[-1],
            self.config.out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
            rank=self.config.lora_rank,
            scale=self.config.lora_scale,
            variational=self.config.lora_variational,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        current = self.input_block(x)
        skip_connections.append(current)

        for down in self.down_blocks:
            skip, current = down(current)
            skip_connections.append(skip)

        current = self.bottleneck(current)

        for up, skip in zip(self.up_blocks, reversed(skip_connections)):
            current = up(current, skip)

        return self.output_conv(current)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return nn.Module.state_dict(self)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:  # type: ignore[override]
        nn.Module.load_state_dict(self, state)
