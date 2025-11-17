from __future__ import annotations

from typing import Optional, Tuple  # noqa: UP035  # Python 3.9 compatibility

import pytest
import torch

from simlx.models.matryoshka_unet import (
    MatryoshkaOutput,
    MatryoshkaUNet,
    MatryoshkaUNetConfig,
)
from simlx.models.registry import create_model


def _build_config() -> MatryoshkaUNetConfig:
    return MatryoshkaUNetConfig(
        in_channels=3,
        feature_sizes=(32, 64),
        bottleneck_features=128,
        scale_channels={"bottleneck": 16, "up0": 32, "up1": 48},
    )


def test_matryoshka_output_shapes() -> None:
    config = _build_config()
    model = MatryoshkaUNet(config)
    x = torch.randn(2, 3, 64, 64)
    output = model(x)

    assert isinstance(output, MatryoshkaOutput)
    expected_channels = sum(config.scale_channels.values())
    assert output.features.shape == (2, expected_channels, 64, 64)
    assert output.scale_map["bottleneck"] == slice(0, config.scale_channels["bottleneck"])


def test_scale_attribution_alignment() -> None:
    config = _build_config()
    model = MatryoshkaUNet(config)
    x = torch.randn(1, 3, 32, 32)
    output = model(x)

    bottleneck_slice = output.scale_map["bottleneck"]
    fine_slice = output.scale_map["up1"]

    bottleneck_features = output.features[:, bottleneck_slice, :, :]
    fine_features = output.features[:, fine_slice, :, :]

    assert bottleneck_features.shape[1] == config.scale_channels["bottleneck"]
    assert fine_features.shape[1] == config.scale_channels["up1"]


def test_custom_projection_head_is_used() -> None:
    class DummyHead(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
        projection_head_factory=lambda in_ch, out_ch: DummyHead(in_ch, out_ch),
    )

    model = MatryoshkaUNet(config)

    for head in model.scale_heads.values():
        assert isinstance(head.head, DummyHead)


def test_registry_integration_with_kwargs() -> None:
    model = create_model(
        "matryoshka_unet",
        in_channels=3,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
    )

    assert isinstance(model, MatryoshkaUNet)


def test_registry_integration_with_config() -> None:
    config = _build_config()
    model = create_model("matryoshka_unet", config=config)
    assert isinstance(model, MatryoshkaUNet)
    assert model.config is config


# Config validation tests
def test_config_validation_spatial_dims() -> None:
    with pytest.raises(ValueError, match="Only spatial_dims=2 supported"):
        MatryoshkaUNetConfig(spatial_dims=3)


def test_config_validation_empty_feature_sizes() -> None:
    with pytest.raises(ValueError, match="feature_sizes must not be empty"):
        MatryoshkaUNetConfig(feature_sizes=())


def test_config_validation_missing_scale_keys() -> None:
    with pytest.raises(ValueError, match="Missing scale keys"):
        MatryoshkaUNetConfig(
            feature_sizes=(32, 64),
            scale_channels={"bottleneck": 16},  # Missing up0, up1
        )


def test_config_validation_extra_scale_keys() -> None:
    with pytest.raises(ValueError, match="Unexpected scale keys"):
        MatryoshkaUNetConfig(
            feature_sizes=(32, 64),
            scale_channels={"bottleneck": 16, "up0": 32, "up1": 48, "invalid": 10},
        )


def test_config_validation_negative_scale_channels() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        MatryoshkaUNetConfig(
            feature_sizes=(32, 64),
            scale_channels={"bottleneck": 16, "up0": 32, "up1": -1},
        )


def test_config_validation_invalid_upsample_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported upsample_mode"):
        MatryoshkaUNetConfig(upsample_mode="invalid_mode")


def test_config_validation_invalid_normalization() -> None:
    with pytest.raises(ValueError, match="Unsupported normalization"):
        MatryoshkaUNetConfig(normalization="invalid_norm")


def test_config_validation_invalid_conv_mode() -> None:
    with pytest.raises(ValueError, match="Only conv_mode='standard' supported"):
        MatryoshkaUNetConfig(conv_mode="invalid")


def test_config_validation_invalid_scale_normalization() -> None:
    with pytest.raises(ValueError, match="Unsupported scale_normalization"):
        MatryoshkaUNetConfig(scale_normalization="invalid")


def test_config_validation_scale_bn_momentum_range() -> None:
    with pytest.raises(ValueError, match="scale_bn_momentum must be between 0 and 1"):
        MatryoshkaUNetConfig(scale_bn_momentum=1.5)
    with pytest.raises(ValueError, match="scale_bn_momentum must be between 0 and 1"):
        MatryoshkaUNetConfig(scale_bn_momentum=0.0)


def test_config_validation_quantile_num_quantiles() -> None:
    with pytest.raises(ValueError, match="quantile_num_quantiles must be at least 2"):
        MatryoshkaUNetConfig(scale_normalization="quantile", quantile_num_quantiles=1)


def test_config_validation_quantile_percentiles_length() -> None:
    with pytest.raises(ValueError, match="quantile_percentiles length"):
        MatryoshkaUNetConfig(
            scale_normalization="quantile",
            quantile_num_quantiles=5,
            quantile_percentiles=[0.2, 0.4, 0.6],  # Wrong length
        )


def test_config_validation_quantile_target_dist() -> None:
    with pytest.raises(ValueError, match="quantile_target_dist must be 'gaussian' or 'uniform'"):
        MatryoshkaUNetConfig(scale_normalization="quantile", quantile_target_dist="invalid")


def test_config_validation_quantile_momentum() -> None:
    with pytest.raises(ValueError, match="quantile_momentum must be between 0 and 1"):
        MatryoshkaUNetConfig(scale_normalization="quantile", quantile_momentum=1.5)


def test_config_validation_quantile_temperature() -> None:
    with pytest.raises(ValueError, match="quantile_temperature must be positive"):
        MatryoshkaUNetConfig(scale_normalization="quantile", quantile_temperature=0.0)


def test_config_validation_quantile_eps() -> None:
    with pytest.raises(ValueError, match="quantile_eps must be positive"):
        MatryoshkaUNetConfig(scale_normalization="quantile", quantile_eps=0.0)


def test_config_validation_multiple_projection_strategies() -> None:
    with pytest.raises(ValueError, match="Cannot use multiple projection strategies"):
        MatryoshkaUNetConfig(
            use_fixed_gaussian_projections=True,
            use_random_projections=True,
        )


def test_config_validation_projection_factory_with_others() -> None:
    with pytest.raises(ValueError, match="Cannot use multiple projection strategies"):
        MatryoshkaUNetConfig(
            use_fixed_gaussian_projections=True,
            projection_head_factory=lambda in_ch, out_ch: torch.nn.Conv2d(in_ch, out_ch, 1),
        )


# Activation function tests
@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "gelu"])
def test_different_activations(activation: str) -> None:
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
        activation=activation,
    )
    model = MatryoshkaUNet(config)
    x = torch.randn(1, 1, 32, 32)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)


def test_invalid_activation() -> None:
    config = _build_config()
    config.activation = "invalid"
    with pytest.raises(ValueError, match="Unsupported activation"):
        MatryoshkaUNet(config)


# Normalization tests
@pytest.mark.parametrize("normalization", [None, "identity", "batch", "group", "layer"])
def test_different_normalizations(normalization: Optional[str]) -> None:  # noqa: UP045  # Python 3.9 compatibility
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
        normalization=normalization,
    )
    model = MatryoshkaUNet(config)
    x = torch.randn(1, 1, 32, 32)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)


# Upsample mode tests
@pytest.mark.parametrize("upsample_mode", ["nearest", "bilinear", "bicubic", "area"])
def test_different_upsample_modes(upsample_mode: str) -> None:
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
        upsample_mode=upsample_mode,
    )
    model = MatryoshkaUNet(config)
    x = torch.randn(1, 1, 32, 32)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)
    assert output.features.shape[2:] == x.shape[2:]


# Scale normalization tests
@pytest.mark.parametrize("scale_normalization", [None, "batchnorm", "quantile"])
def test_different_scale_normalizations(scale_normalization: Optional[str]) -> None:  # noqa: UP045  # Python 3.9 compatibility
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
        scale_normalization=scale_normalization,
    )
    model = MatryoshkaUNet(config)
    x = torch.randn(1, 1, 32, 32)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)


def test_quantile_normalization_with_percentiles() -> None:
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
        scale_normalization="quantile",
        quantile_num_quantiles=5,
        quantile_percentiles=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    model = MatryoshkaUNet(config)
    x = torch.randn(1, 1, 32, 32)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)


# Projection strategy tests
def test_fixed_gaussian_projections() -> None:
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
        use_fixed_gaussian_projections=True,
        fixed_gaussian_seed=42,
    )
    model = MatryoshkaUNet(config)
    x = torch.randn(1, 1, 32, 32)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)
    # Check that FixedGaussianProjection is used
    from simlx.models.random_projection import FixedGaussianProjection

    assert isinstance(model.scale_heads["bottleneck"].head, FixedGaussianProjection)


def test_random_projections() -> None:
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
        use_random_projections=True,
    )
    model = MatryoshkaUNet(config)
    x = torch.randn(1, 1, 32, 32)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)
    # Check that RandomGaussianProjection is used
    from simlx.models.random_projection import RandomGaussianProjection

    assert isinstance(model.scale_heads["bottleneck"].head, RandomGaussianProjection)


def test_default_conv_projection() -> None:
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
    )
    model = MatryoshkaUNet(config)
    # Check that default Conv2d is used
    assert isinstance(model.scale_heads["bottleneck"].head, torch.nn.Conv2d)


# Default scale_channels generation
def test_default_scale_channels() -> None:
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        total_output_channels=96,  # Will be divided among 3 scales
    )
    assert "bottleneck" in config.scale_channels
    assert "up0" in config.scale_channels
    assert "up1" in config.scale_channels
    assert sum(config.scale_channels.values()) == 96


def test_default_scale_channels_from_feature_size() -> None:
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        # No total_output_channels or scale_channels specified
    )
    # Should use feature_sizes[0] * num_scales as default
    assert len(config.scale_channels) == 3  # bottleneck, up0, up1
    assert all(v > 0 for v in config.scale_channels.values())


def test_default_scale_channels_too_small() -> None:
    with pytest.raises(ValueError, match="total_output_channels too small"):
        MatryoshkaUNetConfig(
            in_channels=1,
            feature_sizes=(16, 32),
            bottleneck_features=64,
            total_output_channels=2,  # Too small for 3 scales
        )


# Different input sizes
@pytest.mark.parametrize("input_size", [(32, 32), (64, 64), (128, 128), (16, 32), (32, 16)])
def test_different_input_sizes(input_size: Tuple[int, int]) -> None:  # noqa: UP006  # Python 3.9 compatibility
    config = _build_config()
    model = MatryoshkaUNet(config)
    x = torch.randn(2, 3, *input_size)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)
    assert output.features.shape[2:] == input_size


# State dict tests
def test_state_dict_save_load() -> None:
    config = _build_config()
    model1 = MatryoshkaUNet(config)
    model2 = MatryoshkaUNet(config)

    x = torch.randn(1, 3, 32, 32)
    output1 = model1(x)

    # Save and load state dict
    state_dict = model1.state_dict()
    model2.load_state_dict(state_dict)

    # Models should produce same output
    output2 = model2(x)
    torch.testing.assert_close(output1.features, output2.features)


# Dropout tests
def test_dropout() -> None:
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
        dropout=0.5,
    )
    model = MatryoshkaUNet(config)
    x = torch.randn(1, 1, 32, 32)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)


# Kernel size and padding tests
@pytest.mark.parametrize("kernel_size,padding", [(3, 1), (5, 2), (7, 3)])
def test_different_kernel_sizes(kernel_size: int, padding: int) -> None:
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
        kernel_size=kernel_size,
        padding=padding,
    )
    model = MatryoshkaUNet(config)
    x = torch.randn(1, 1, 32, 32)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)


# Align corners tests
@pytest.mark.parametrize("align_corners", [True, False, None])
def test_align_corners(align_corners: Optional[bool]) -> None:  # noqa: UP045  # Python 3.9 compatibility
    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
        upsample_mode="bilinear",
        align_corners=align_corners,
    )
    model = MatryoshkaUNet(config)
    x = torch.randn(1, 1, 32, 32)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)


# Multiple batch sizes
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_different_batch_sizes(batch_size: int) -> None:
    config = _build_config()
    model = MatryoshkaUNet(config)
    x = torch.randn(batch_size, 3, 64, 64)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)
    assert output.features.shape[0] == batch_size


# Test that all scale maps are contiguous and non-overlapping
def test_scale_map_contiguity() -> None:
    config = _build_config()
    model = MatryoshkaUNet(config)
    x = torch.randn(1, 3, 64, 64)
    output = model(x)

    # Check that slices are contiguous and cover all channels
    slices = sorted(output.scale_map.values(), key=lambda s: s.start)
    assert slices[0].start == 0
    for i in range(len(slices) - 1):
        assert slices[i].stop == slices[i + 1].start
    assert slices[-1].stop == output.features.shape[1]


# Test with None config (should use defaults)
def test_none_config() -> None:
    model = MatryoshkaUNet(None)
    x = torch.randn(1, 3, 64, 64)
    output = model(x)
    assert isinstance(output, MatryoshkaOutput)


# Test projection head factory type validation
def test_projection_head_factory_type_error() -> None:
    def invalid_factory(in_ch: int, out_ch: int) -> str:  # type: ignore[return]
        return "not a module"

    config = MatryoshkaUNetConfig(
        in_channels=1,
        feature_sizes=(16, 32),
        bottleneck_features=64,
        scale_channels={"bottleneck": 8, "up0": 12, "up1": 16},
        projection_head_factory=invalid_factory,  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match=r"projection_head_factory must return nn\.Module"):
        MatryoshkaUNet(config)
