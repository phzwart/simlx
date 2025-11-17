"""Base protocol for experiment tracking adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ExperimentTracker(ABC):
    """Protocol for experiment tracking backends."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics at a given step."""

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric."""

    @abstractmethod
    def log_artifact(
        self,
        file_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Log a file artifact."""

    @abstractmethod
    def log_model(self, model_path: str | Path, model_name: str) -> None:
        """Log a model checkpoint."""

    @abstractmethod
    def log_figure(self, figure: Any, name: str, step: int | None = None) -> None:
        """Log a matplotlib/plotly figure."""

    @abstractmethod
    def set_tag(self, key: str, value: str) -> None:
        """Set a tag for the experiment."""

    @abstractmethod
    def start_run(self, run_name: str | None = None, **kwargs: Any) -> None:
        """Start a new experiment run."""

    @abstractmethod
    def end_run(self) -> None:
        """End the current experiment run."""

    @property
    @abstractmethod
    def run_id(self) -> str | None:
        """Get current run ID."""


class NoOpTracker(ExperimentTracker):
    """No-op tracker for testing or when tracking is disabled."""

    def log_params(self, params: dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        return None

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        return None

    def log_artifact(
        self,
        file_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        return None

    def log_model(self, model_path: str | Path, model_name: str) -> None:
        return None

    def log_figure(self, figure: Any, name: str, step: int | None = None) -> None:
        return None

    def set_tag(self, key: str, value: str) -> None:
        return None

    def start_run(self, run_name: str | None = None, **kwargs: Any) -> None:
        return None

    def end_run(self) -> None:
        return None

    @property
    def run_id(self) -> str | None:
        return None
