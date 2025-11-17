"""Model registry for wiring base models into training workflows."""

from __future__ import annotations

from collections.abc import Callable

from simlx.models.base import BaseModel
from simlx.models.matryoshka_unet import _matryoshka_unet_factory

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


register_model("matryoshka_unet", _matryoshka_unet_factory)
register_model(
    "matryoshka_unet_2d",
    lambda **kwargs: _matryoshka_unet_factory(spatial_dims=2, **kwargs),
)
register_model(
    "matryoshka_unet_3d",
    lambda **kwargs: _matryoshka_unet_factory(spatial_dims=3, **kwargs),
)
