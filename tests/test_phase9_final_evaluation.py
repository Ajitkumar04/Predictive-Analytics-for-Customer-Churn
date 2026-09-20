import numpy as np

from src.models.evaluate import assess_business_tradeoff, evaluate_model, generate_model_card


def test_evaluate_model_returns_core_metrics_and_confusion_matrix():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.2, 0.8, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4])

    result = evaluate_model(y_true, y_prob, threshold=0.5)

    assert set({"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix", "threshold"}).issubset(result)
    assert 0 <= result["accuracy"] <= 1
    assert result["threshold"] == 0.5
    assert result["confusion_matrix"].shape == (2, 2)


def test_generate_model_card_returns_summary_for_reporting():
    metrics = {
        "accuracy": 0.89,
        "precision": 0.83,
        "recall": 0.78,
        "f1": 0.80,
        "roc_auc": 0.86,
        "pr_auc": 0.82,
    }

    card = generate_model_card("logistic_regression", metrics)

    assert card["model_name"] == "logistic_regression"
    assert card["performance"]["accuracy"] == 0.89
    assert "limitations" in card
    assert len(card["limitations"]) >= 2


def test_assess_business_tradeoff_returns_recommendation():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.25, 0.82, 0.48, 0.77, 0.66, 0.31])

    report = assess_business_tradeoff(y_true, y_prob)

    assert "recommended_threshold" in report
    assert "tradeoff_summary" in report
    assert "thresholds" in report
    assert report["recommended_threshold"] >= 0
