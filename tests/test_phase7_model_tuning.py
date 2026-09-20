import numpy as np
import pandas as pd

from src.models.tune import optimize_threshold, tune_model


def _make_binary_dataset():
    rng = np.random.default_rng(42)
    n = 180
    X = pd.DataFrame({
        "numeric_a": rng.normal(0, 1, n),
        "numeric_b": rng.normal(2, 1.5, n),
        "category_x": rng.choice(["A", "B", "C"], size=n),
        "category_y": rng.choice(["Low", "High"], size=n),
    })
    y = ((X["numeric_a"] + X["numeric_b"] / 2) > 1.5).astype(int)
    return X, y


def test_tune_model_returns_best_estimator_and_scores():
    X, y = _make_binary_dataset()
    X_train = X.iloc[:140]
    y_train = y.iloc[:140]

    tuned = tune_model(X_train, y_train, model_name="logistic_regression", cv=3)

    assert hasattr(tuned, "best_estimator_")
    assert hasattr(tuned, "best_params_")
    assert hasattr(tuned, "best_score_")
    assert tuned.best_score_ >= 0


def test_optimize_threshold_returns_metrics_table():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])

    results = optimize_threshold(y_true, y_proba, thresholds=[0.3, 0.5, 0.7])

    assert set(["threshold", "precision", "recall", "f1", "accuracy"]).issubset(results.columns)
    assert len(results) == 3
    assert results["threshold"].tolist() == [0.3, 0.5, 0.7]
