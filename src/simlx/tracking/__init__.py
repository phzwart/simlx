"""Experiment tracking adapters."""

from simlx.tracking.base import ExperimentTracker, NoOpTracker
from simlx.tracking.clearml_adapter import ClearMLAdapter
from simlx.tracking.mlflow_adapter import MLflowAdapter

__all__ = [
    "ClearMLAdapter",
    "ExperimentTracker",
    "MLflowAdapter",
    "NoOpTracker",
    "create_tracker",
]


def create_tracker(backend: str, **kwargs: object) -> ExperimentTracker:
    """Factory for creating experiment trackers.

    Args:
        backend: One of ``"mlflow"``, ``"clearml"``, ``"none"``.
        **kwargs: Backend specific configuration options.
    """

    backend_lower = backend.lower()

    if backend_lower == "mlflow":
        return MLflowAdapter(**kwargs)  # type: ignore[arg-type]
    if backend_lower == "clearml":
        return ClearMLAdapter(**kwargs)  # type: ignore[arg-type]
    if backend_lower == "none":
        return NoOpTracker()
    raise ValueError(f"Unknown tracking backend: {backend}")
