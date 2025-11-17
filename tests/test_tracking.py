import importlib.util

import pytest

from simlx.tracking import (
    ClearMLAdapter,
    MLflowAdapter,
    NoOpTracker,
    create_tracker,
)


def test_tracker_factory_noop() -> None:
    tracker = create_tracker("none")
    assert isinstance(tracker, NoOpTracker)
    tracker.start_run()
    tracker.log_params({"foo": "bar"})
    tracker.log_metric("loss", 0.1, step=1)
    tracker.end_run()


@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="MLflow not installed")  # type: ignore[attr-defined]
def test_tracker_factory_mlflow() -> None:
    tracker = create_tracker("mlflow")
    assert isinstance(tracker, MLflowAdapter)


@pytest.mark.skipif(importlib.util.find_spec("clearml") is None, reason="ClearML not installed")  # type: ignore[attr-defined]
def test_tracker_factory_clearml() -> None:
    tracker = create_tracker("clearml")
    assert isinstance(tracker, ClearMLAdapter)


def test_tracker_factory_invalid_backend() -> None:
    with pytest.raises(ValueError):
        create_tracker("unknown-backend")
