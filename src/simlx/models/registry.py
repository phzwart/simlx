"""Model registry for wiring base models into training workflows."""

from __future__ import annotations

from collections.abc import Callable

from simlx.models.base import BaseModel
from simlx.models.barlow_twins import BarlowTwins, BarlowTwinsConfig
from simlx.models.matryoshka_unet import _matryoshka_unet_factory
from simlx.models.unet2d import UNet2D, UNet2DConfig

_REGISTRY: dict[str, Callable[..., BaseModel]] = {}


def register_model(name: str, factory: Callable[..., BaseModel]) -> None:
    if name in _REGISTRY:
        raise KeyError(f"Model '{name}' is already registered")
    _REGISTRY[name] = factory


def create_model(name: str, **kwargs) -> BaseModel:
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Model '{name}' is not registered") from exc
    return factory(**kwargs)


def available_models() -> dict[str, Callable[..., BaseModel]]:
    return dict(_REGISTRY)


def _unet2d_factory(**kwargs) -> BaseModel:
    config = kwargs.pop("config", None)
    if config is not None and kwargs:
        raise ValueError("Pass config or keyword overrides, not both")
    if config is None:
        config = UNet2DConfig(**kwargs)
    return UNet2D(config=config)


def _barlow_twins_factory(**kwargs) -> BaseModel:
    config = kwargs.pop("config", None)
    if config is not None and kwargs:
        raise ValueError("Pass config or keyword overrides, not both")
    if config is None:
        config = BarlowTwinsConfig(**kwargs)
    if not isinstance(config, BarlowTwinsConfig):
        raise TypeError("config must be a BarlowTwinsConfig instance")
    return BarlowTwins(config=config)


register_model("unet2d", _unet2d_factory)
register_model("matryoshka_unet", _matryoshka_unet_factory)
register_model(
    "matryoshka_unet_2d",
    lambda **kwargs: _matryoshka_unet_factory(spatial_dims=2, **kwargs),
)
register_model(
    "matryoshka_unet_3d",
    lambda **kwargs: _matryoshka_unet_factory(spatial_dims=3, **kwargs),
)
register_model("barlow_twins", _barlow_twins_factory)
