import pandas as pd
import pytest

from src.monitoring.drift import compute_psi, detect_data_drift
from src.monitoring.performance import retraining_required, summarize_prediction_metrics


def test_compute_psi_returns_low_value_for_similar_distributions():
    reference = pd.Series([10, 20, 30, 40, 50] * 20)
    current = pd.Series([11, 21, 29, 41, 49] * 20)

    psi_value = compute_psi(reference, current)

    assert psi_value < 0.05


def test_detect_data_drift_flags_severe_shift():
    reference = pd.DataFrame({"monthly_charges": [10, 15, 20, 25, 30] * 10})
    current = pd.DataFrame({"monthly_charges": [60, 65, 70, 75, 80] * 10})

    drift_report = detect_data_drift(reference, current, ["monthly_charges"])

    assert "monthly_charges" in drift_report
    assert drift_report["monthly_charges"]["status"] == "drift_detected"
    assert drift_report["monthly_charges"]["psi"] > 0.3


def test_summarize_prediction_metrics_returns_summary():
    summary = summarize_prediction_metrics([0.10, 0.40, 0.80], [0, 0, 1])

    assert summary["average_probability"] == pytest.approx(0.4333333333, rel=1e-5)
    assert summary["prediction_rate"] == pytest.approx(1 / 3, rel=1e-5)
    assert summary["error_rate"] == pytest.approx(0.0, abs=1e-9)


def test_retraining_required_when_drift_or_low_average_probability_is_seen():
    drift_report = {"monthly_charges": {"status": "drift_detected", "psi": 0.68}}

    assert retraining_required(drift_report, average_probability=0.35) is True
    assert retraining_required({"monthly_charges": {"status": "stable", "psi": 0.01}}, average_probability=0.35) is True
