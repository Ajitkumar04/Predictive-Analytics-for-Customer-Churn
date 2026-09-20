from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow


def _normalize_tracking_uri(tracking_uri: str | None) -> str | None:
    if tracking_uri is None:
        return None

    if tracking_uri.startswith("file://"):
        candidate = tracking_uri.replace("file://", "", 1)
        if not candidate.startswith("/"):
            return Path(candidate).as_uri()

    return tracking_uri


class MLflowExperimentTracker:
    """Small wrapper for logging model metadata and evaluation metrics with MLflow."""

    def __init__(self, experiment_name: str, tracking_uri: str | None = None):
        self.experiment_name = experiment_name
        self.tracking_uri = _normalize_tracking_uri(tracking_uri)
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.latest_run = None

    @contextmanager
    def start_run(self):
        with mlflow.start_run(run_name=self.experiment_name) as run:
            self.latest_run = run
            yield run

    def log_metric(self, name: str, value: float) -> None:
        mlflow.log_metric(name, float(value))

    def log_param(self, name: str, value: Any) -> None:
        mlflow.log_param(name, str(value))

    def log_model_metadata(self, metadata: dict[str, Any]) -> None:
        for key, value in metadata.items():
            self.log_param(key, value)
