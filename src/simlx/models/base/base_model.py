"""Base model abstractions coupled with MLflow integration hooks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

try:
    import mlflow  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore[assignment]


class MLflowMixin:
    """Lightweight mixin offering MLflow convenience helpers."""

    mlflow_run: str | None = None

    def log_model(self, artifact_path: str, **kwargs: Any) -> None:
        if mlflow is None:
            raise RuntimeError("MLflow unavailable")
        mlflow.sklearn.log_model(self, artifact_path, **kwargs)  # type: ignore[arg-type]

    def log_params(self, params: dict[str, Any]) -> None:
        if mlflow is None:
            raise RuntimeError("MLflow unavailable")
        mlflow.log_params(params)


class BaseModel(ABC, MLflowMixin):
    """Shared interface for models participating in peer learning."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """Run the model forward pass."""

    def state_dict(self) -> dict[str, Any]:
        """Return serializable model state."""
        raise NotImplementedError

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore model state from a serialized form."""
        raise NotImplementedError
