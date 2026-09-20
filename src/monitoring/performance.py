from __future__ import annotations

from typing import Any

import numpy as np


def summarize_prediction_metrics(probabilities: list[float], actual_labels: list[int]) -> dict[str, float]:
    """Summarize average probability, prediction rate, and error rate."""
    arr_prob = np.asarray(probabilities, dtype=float)
    arr_labels = np.asarray(actual_labels, dtype=int)

    if arr_prob.size == 0:
        return {
            "average_probability": 0.0,
            "prediction_rate": 0.0,
            "error_rate": 0.0,
        }

    predictions = (arr_prob >= 0.5).astype(int)
    error_rate = float(np.mean(predictions != arr_labels)) if arr_labels.size else 0.0

    return {
        "average_probability": float(arr_prob.mean()),
        "prediction_rate": float(np.mean(predictions)),
        "error_rate": error_rate,
    }


def retraining_required(drift_report: dict[str, dict[str, Any]], average_probability: float, min_probability: float = 0.45) -> bool:
    """Simple retraining trigger based on drift or poor risk signal."""
    drift_detected = any(
        item.get("status") in {"drift_detected", "warning"}
        for item in drift_report.values()
    )
    low_average_probability = average_probability < min_probability
    return drift_detected or low_average_probability
