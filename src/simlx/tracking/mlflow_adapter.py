"""MLflow experiment tracking adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import mlflow  # type: ignore[import-untyped]

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False

from simlx.tracking.base import ExperimentTracker


class MLflowAdapter(ExperimentTracker):
    """Adapter for MLflow experiment tracking."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        if not MLFLOW_AVAILABLE:
            raise ImportError("Install mlflow to enable MLflowAdapter")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        if experiment_name:
            mlflow.set_experiment(experiment_name)

        self._run_id = run_id
        self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_artifact(
        self,
        file_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        mlflow.log_artifact(str(file_path), artifact_path=artifact_path)

    def log_model(self, model_path: str | Path, model_name: str) -> None:
        mlflow.log_artifact(str(model_path), artifact_path=model_name)

    def log_figure(self, figure: Any, name: str, step: int | None = None) -> None:
        mlflow.log_figure(figure, f"{name}.png")

    def set_tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)  # type: ignore[union-attr]

    def start_run(self, run_name: str | None = None, **kwargs: Any) -> None:
        self._active_run = mlflow.start_run(run_name=run_name, run_id=self._run_id, **kwargs)  # type: ignore[union-attr]
        if self._active_run:
            self._run_id = self._active_run.info.run_id

    def end_run(self) -> None:
        mlflow.end_run()  # type: ignore[union-attr]
        self._active_run = None

    @property
    def run_id(self) -> str | None:
        return self._run_id
