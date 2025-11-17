"""Matryoshka U-Net architecture with multi-scale feature attribution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]

from simlx.models.base import BaseModel
from simlx.models.normalization import CascadedNorm2d
from simlx.models.random_projection import FixedGaussianProjection, RandomGaussianProjection


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _create_norm(
    norm_type: str | None,
    num_channels: int,
    num_groups: int,
) -> nn.Module:
    if norm_type is None or norm_type == "identity":
        return nn.Identity()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "group":
        groups = min(num_groups, num_channels)
        groups = groups if groups > 0 else 1
        return nn.GroupNorm(groups, num_channels)
    if norm_type == "layer":
        return nn.GroupNorm(1, num_channels)
    raise ValueError(f"Unsupported normalization: {norm_type}")


class ConvBlock(nn.Module):
    """Two-layer convolutional block with optional normalization and dropout."""

    def __init__(self, in_channels: int, out_channels: int, config: MatryoshkaUNetConfig) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=config.kernel_size,
            padding=config.padding,
            bias=config.normalization in (None, "identity"),
        )
        self.norm1 = _create_norm(config.normalization, out_channels, config.norm_groups)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=config.kernel_size,
            padding=config.padding,
            bias=config.normalization in (None, "identity"),
        )
        self.norm2 = _create_norm(config.normalization, out_channels, config.norm_groups)
        self.activation = _get_activation(config.activation)
        self.dropout = nn.Dropout2d(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        return out


class DownBlock(nn.Module):
    """Encoder block with pooling."""

    def __init__(self, in_channels: int, out_channels: int, config: MatryoshkaUNetConfig) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels, config)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.block(x)
        return features, self.pool(features)


class UpBlock(nn.Module):
    """Decoder block combining interpolation upsampling with a ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int, config: MatryoshkaUNetConfig) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels, config)
        self.upsample_mode = config.upsample_mode
        self.align_corners = (
            config.align_corners
            if config.upsample_mode
            in {
                "linear",
                "bilinear",
                "bicubic",
                "trilinear",
            }
            else None
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.align_corners is None:
            upscaled = F.interpolate(x, size=skip.shape[2:], mode=self.upsample_mode)
        else:
            upscaled = F.interpolate(
                x,
                size=skip.shape[2:],
                mode=self.upsample_mode,
                align_corners=self.align_corners,
            )
        concatenated = torch.cat([upscaled, skip], dim=1)
        return self.block(concatenated)


class ProjectionHead(nn.Module):
    """Projection head applied to each scale before upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: "MatryoshkaUNetConfig",
    ) -> None:
        super().__init__()
        if config.use_fixed_gaussian_projections:
            # Fixed deterministic Gaussian projection (no gradients, same for all networks)
            self.head = FixedGaussianProjection(
                in_features=in_channels,
                out_features=out_channels,
                seed=config.fixed_gaussian_seed,
            )
        elif config.use_random_projections:
            self.head = RandomGaussianProjection(
                in_features=in_channels,
                out_features=out_channels,
            )
        elif config.projection_head_factory is not None:
            module = config.projection_head_factory(in_channels, out_channels)
            if not isinstance(module, nn.Module):
                raise TypeError("projection_head_factory must return nn.Module")
            self.head = module
        else:
            self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def _make_scale_normalizer(config: "MatryoshkaUNetConfig", num_channels: int) -> nn.Module:
    """Factory for scale-dependent global normalizers.
    
    For exposed feature maps, uses BatchNorm → QuantileNorm cascade when
    scale_normalization="quantile".
    """

    if config.scale_normalization == "batchnorm":
        return nn.BatchNorm2d(
            num_channels,
            momentum=config.scale_bn_momentum,
            affine=False,
            track_running_stats=True,
        )
    if config.scale_normalization == "quantile":
        # BatchNorm → QuantileNorm cascade for exposed feature maps
        return CascadedNorm2d(
            num_channels,
            bn_momentum=config.scale_bn_momentum,
            bn_affine=False,
            bn_track_running_stats=True,
            num_quantiles=config.quantile_num_quantiles,
            percentiles=(
                list(config.quantile_percentiles)
                if config.quantile_percentiles is not None
                else None
            ),
            target_dist=config.quantile_target_dist,
            momentum=config.quantile_momentum,
            temperature=config.quantile_temperature,
            eps=config.quantile_eps,
            affine=config.quantile_affine,
            track_running_stats=config.quantile_track_running_stats,
        )
    return nn.Identity()


@dataclass
class MatryoshkaOutput:
    """Structured output containing concatenated features and attribution map."""

    features: torch.Tensor
    scale_map: dict[str, slice]


@dataclass
class MatryoshkaUNetConfig:
    """Configuration for :class:`MatryoshkaUNet`."""

    spatial_dims: int = 2
    in_channels: int = 3
    feature_sizes: Sequence[int] = (64, 128, 256, 512)
    bottleneck_features: int = 1024
    scale_channels: dict[str, int] = field(default_factory=dict)
    total_output_channels: int | None = None

    conv_mode: str = "standard"
    kernel_size: int = 3
    padding: int = 1

    activation: str = "relu"
    dropout: float = 0.0
    normalization: str | None = "group"
    norm_groups: int = 32

    upsample_mode: str = "bilinear"
    align_corners: bool | None = False
    projection_head_factory: Callable[[int, int], nn.Module] | None = None
    use_fixed_gaussian_projections: bool = False
    fixed_gaussian_seed: int = 42
    scale_normalization: str | None = "batchnorm"
    scale_bn_momentum: float = 0.01
    # QuantileNorm hyperparameters (used when scale_normalization="quantile")
    quantile_num_quantiles: int = 5
    quantile_percentiles: Sequence[float] | None = None
    quantile_target_dist: str = "gaussian"
    quantile_momentum: float = 0.01
    quantile_temperature: float = 0.05
    quantile_eps: float = 1e-5
    quantile_affine: bool = True
    quantile_track_running_stats: bool = True
    use_random_projections: bool = False

    def __post_init__(self) -> None:
        self._validate_spatial_dims()
        self._validate_feature_sizes()
        expected_scales = self._expected_scales()
        if self.scale_channels:
            self._validate_provided_scales(expected_scales)
        else:
            self.scale_channels = self._build_default_scales(expected_scales)
        self._validate_upsample_mode()
        self._validate_normalization()
        self._validate_conv_mode()
        self._validate_scale_normalization()
        self._validate_projection_settings()

    def _validate_spatial_dims(self) -> None:
        if self.spatial_dims != 2:
            raise ValueError("Only spatial_dims=2 supported")

    def _validate_feature_sizes(self) -> None:
        if not self.feature_sizes:
            raise ValueError("feature_sizes must not be empty")

    def _expected_scales(self) -> list[str]:
        return ["bottleneck"] + [f"up{i}" for i in range(len(self.feature_sizes))]

    def _validate_provided_scales(self, expected_scales: Sequence[str]) -> None:
        extra = sorted(set(self.scale_channels) - set(expected_scales))
        if extra:
            raise ValueError(f"Unexpected scale keys: {extra}")
        missing = sorted(set(expected_scales) - set(self.scale_channels))
        if missing:
            raise ValueError(f"Missing scale keys: {missing}")
        for key, value in self.scale_channels.items():
            if value <= 0:
                raise ValueError(f"scale_channels[{key!r}] must be positive")

    def _build_default_scales(self, expected_scales: Sequence[str]) -> dict[str, int]:
        total_scales = len(expected_scales)
        total_channels = self.total_output_channels
        if total_channels is None:
            base_default = self.feature_sizes[0]
            total_channels = base_default * total_scales
        if total_channels < total_scales:
            raise ValueError("total_output_channels too small for number of scales")
        base, remainder = divmod(total_channels, total_scales)
        allocations: list[int] = []
        for idx in range(total_scales):
            value = base + (1 if idx < remainder else 0)
            allocations.append(max(1, value))
        return {scale_name: allocations[idx] for idx, scale_name in enumerate(expected_scales)}

    def _validate_upsample_mode(self) -> None:
        allowed = {"nearest", "linear", "bilinear", "bicubic", "trilinear", "area"}
        if self.upsample_mode not in allowed:
            raise ValueError(f"Unsupported upsample_mode: {self.upsample_mode}")

    def _validate_normalization(self) -> None:
        allowed = {None, "identity", "batch", "group", "layer"}
        if self.normalization not in allowed:
            raise ValueError(f"Unsupported normalization: {self.normalization}")

    def _validate_conv_mode(self) -> None:
        if self.conv_mode != "standard":
            raise ValueError("Only conv_mode='standard' supported in Stage 1")

    def _validate_scale_normalization(self) -> None:
        allowed = {None, "batchnorm", "quantile"}
        if self.scale_normalization not in allowed:
            raise ValueError(f"Unsupported scale_normalization: {self.scale_normalization}")
        if self.scale_bn_momentum <= 0 or self.scale_bn_momentum > 1:
            raise ValueError("scale_bn_momentum must be between 0 and 1")
        if self.scale_normalization == "quantile":
            if self.quantile_num_quantiles < 2:
                raise ValueError("quantile_num_quantiles must be at least 2")
            if self.quantile_percentiles is not None:
                if len(self.quantile_percentiles) != self.quantile_num_quantiles:
                    raise ValueError(
                        f"quantile_percentiles length ({len(self.quantile_percentiles)}) "
                        f"must match quantile_num_quantiles ({self.quantile_num_quantiles})"
                    )
            if self.quantile_target_dist not in {"gaussian", "uniform"}:
                raise ValueError(
                    f"quantile_target_dist must be 'gaussian' or 'uniform', "
                    f"got {self.quantile_target_dist}"
                )
            if self.quantile_momentum <= 0 or self.quantile_momentum > 1:
                raise ValueError("quantile_momentum must be between 0 and 1")
            if self.quantile_temperature <= 0:
                raise ValueError("quantile_temperature must be positive")
            if self.quantile_eps <= 0:
                raise ValueError("quantile_eps must be positive")

    def _validate_projection_settings(self) -> None:
        projection_strategies = sum(
            [
                self.use_fixed_gaussian_projections,
                self.use_random_projections,
                self.projection_head_factory is not None,
            ]
        )
        if projection_strategies > 1:
            raise ValueError(
                "Cannot use multiple projection strategies simultaneously. "
                "Choose one: use_fixed_gaussian_projections, use_random_projections, "
                "or projection_head_factory."
            )


class MatryoshkaUNet(BaseModel, nn.Module):
    """Configurable Matryoshka U-Net with multi-scale feature attribution."""

    def __init__(self, config: MatryoshkaUNetConfig | None = None) -> None:
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self.config = config or MatryoshkaUNetConfig()

        features = list(self.config.feature_sizes)

        self.input_block = ConvBlock(self.config.in_channels, features[0], self.config)

        self.down_blocks = nn.ModuleList()
        for idx in range(len(features) - 1):
            self.down_blocks.append(DownBlock(features[idx], features[idx + 1], self.config))

        self.bottleneck = ConvBlock(features[-1], self.config.bottleneck_features, self.config)

        reversed_features = list(reversed(features))
        decoder_in_channels = [
            self.config.bottleneck_features + reversed_features[0],
            *[reversed_features[i] + reversed_features[i + 1] for i in range(len(reversed_features) - 1)],
        ]

        self.up_blocks = nn.ModuleList()
        for idx, out_channels in enumerate(reversed_features):
            in_channels = decoder_in_channels[idx]
            self.up_blocks.append(UpBlock(in_channels, out_channels, self.config))

        self.scale_heads = nn.ModuleDict()
        # bottleneck head
        self.scale_heads["bottleneck"] = ProjectionHead(
            self.config.bottleneck_features,
            self.config.scale_channels["bottleneck"],
            self.config,
        )

        for idx, out_channels in enumerate(reversed_features):
            scale_name = f"up{idx}"
            self.scale_heads[scale_name] = ProjectionHead(
                out_channels,
                self.config.scale_channels[scale_name],
                self.config,
            )

        self.scale_norms = nn.ModuleDict()
        for scale_name in self.config.scale_channels:
            channels = self.config.scale_channels[scale_name]
            self.scale_norms[scale_name] = _make_scale_normalizer(self.config, channels)

    def _upsample_to(self, tensor: torch.Tensor, target_size: Sequence[int]) -> torch.Tensor:
        mode = self.config.upsample_mode
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            align_corners = self.config.align_corners
            return F.interpolate(tensor, size=target_size, mode=mode, align_corners=align_corners)
        return F.interpolate(tensor, size=target_size, mode=mode)

    def forward(self, x: torch.Tensor) -> MatryoshkaOutput:
        original_size = x.shape[2:]
        skip_connections: list[torch.Tensor] = []

        current = self.input_block(x)
        skip_connections.append(current)

        for down in self.down_blocks:
            skip, current = down(current)
            skip_connections.append(skip)

        current = self.bottleneck(current)
        bottleneck_features = self.scale_heads["bottleneck"](current)
        bottleneck_features = self.scale_norms["bottleneck"](bottleneck_features)
        bottleneck_upsampled = self._upsample_to(bottleneck_features, original_size)

        scale_tensors: list[torch.Tensor] = [bottleneck_upsampled]
        scale_map: dict[str, slice] = {}
        offset = 0
        channel_count = bottleneck_upsampled.shape[1]
        scale_map["bottleneck"] = slice(offset, offset + channel_count)
        offset += channel_count

        reversed_skips = list(reversed(skip_connections))
        for idx, up_block in enumerate(self.up_blocks):
            skip = reversed_skips[idx]
            current = up_block(current, skip)
            scale_name = f"up{idx}"
            projected = self.scale_heads[scale_name](current)
            projected = self.scale_norms[scale_name](projected)
            upsampled = self._upsample_to(projected, original_size)
            scale_tensors.append(upsampled)
            channel_count = upsampled.shape[1]
            scale_map[scale_name] = slice(offset, offset + channel_count)
            offset += channel_count

        concatenated = torch.cat(scale_tensors, dim=1)
        return MatryoshkaOutput(features=concatenated, scale_map=scale_map)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return nn.Module.state_dict(self)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:  # type: ignore[override]
        nn.Module.load_state_dict(self, state)


def _matryoshka_unet_factory(**kwargs: object) -> BaseModel:
    """Factory function for registry integration."""

    config = kwargs.pop("config", None)
    if config is not None and kwargs:
        raise ValueError("Pass config or keyword overrides, not both")
    if config is None:
        config = MatryoshkaUNetConfig(**kwargs)  # type: ignore[arg-type]
    if not isinstance(config, MatryoshkaUNetConfig):
        raise TypeError("config must be a MatryoshkaUNetConfig instance")
    return MatryoshkaUNet(config=config)
