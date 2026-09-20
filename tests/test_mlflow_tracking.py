from __future__ import annotations

from pathlib import Path

from src.utils.mlflow_tracker import MLflowExperimentTracker


def test_mlflow_tracker_logs_metrics_and_model_metadata(tmp_path):
    tracking_root = tmp_path / "mlruns"
    tracker = MLflowExperimentTracker(
        experiment_name="customer-churn-test",
        tracking_uri=tracking_root.as_uri(),
    )

    with tracker.start_run() as run:
        tracker.log_metric("accuracy", 0.91)
        tracker.log_metric("f1", 0.85)
        tracker.log_param("model", "logistic_regression")
        tracker.log_param("random_state", 42)
        tracker.log_model_metadata({"version": "v1.0.0", "target": "churn"})

    assert tracker.experiment_name == "customer-churn-test"
    assert run.info.run_id
    assert tracker.latest_run is not None
