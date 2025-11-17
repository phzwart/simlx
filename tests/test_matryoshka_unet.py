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
