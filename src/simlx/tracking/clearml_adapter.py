"""ClearML experiment tracking adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from clearml import Task

    CLEARML_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Task = None  # type: ignore[assignment]
    CLEARML_AVAILABLE = False

from simlx.tracking.base import ExperimentTracker


class ClearMLAdapter(ExperimentTracker):
    """Adapter for ClearML experiment tracking."""

    def __init__(
        self,
        project_name: str | None = None,
        task_name: str | None = None,
        task_id: str | None = None,
    ) -> None:
        if not CLEARML_AVAILABLE:
            raise ImportError("Install clearml to enable ClearMLAdapter")

        self._task = None
        self._project_name = project_name
        self._task_name = task_name
        self._task_id = task_id

    def log_params(self, params: dict[str, Any]) -> None:
        if self._task:
            self._task.connect(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._task:
            logger = self._task.get_logger()
            for key, value in metrics.items():
                logger.report_scalar(
                    title=key,
                    series=key,
                    value=value,
                    iteration=step if step is not None else 0,
                )

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if self._task:
            logger = self._task.get_logger()
            logger.report_scalar(
                title=key,
                series=key,
                value=value,
                iteration=step if step is not None else 0,
            )

    def log_artifact(
        self,
        file_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        if self._task:
            name = artifact_path or Path(file_path).name
            self._task.upload_artifact(name=name, artifact_object=str(file_path))

    def log_model(self, model_path: str | Path, model_name: str) -> None:
        if self._task:
            self._task.upload_artifact(name=model_name, artifact_object=str(model_path))

    def log_figure(self, figure: Any, name: str, step: int | None = None) -> None:
        if self._task:
            logger = self._task.get_logger()
            logger.report_matplotlib_figure(
                title=name,
                series=name,
                figure=figure,
                iteration=step if step is not None else 0,
            )

    def set_tag(self, key: str, value: str) -> None:
        if self._task:
            self._task.set_parameter(f"tags/{key}", value)

    def start_run(self, run_name: str | None = None, **kwargs: Any) -> None:
        if Task is None:  # pragma: no cover - defensive
            raise RuntimeError("ClearML Task unavailable")

        task_name = run_name or self._task_name or "Training Run"
        self._task = Task.init(
            project_name=self._project_name or "Default",
            task_name=task_name,
            task_id=self._task_id,
            **kwargs,
        )

    def end_run(self) -> None:
        if self._task:
            self._task.close()
            self._task = None

    @property
    def run_id(self) -> str | None:
        return self._task.id if self._task else self._task_id
