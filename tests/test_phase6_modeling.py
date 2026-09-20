import numpy as np
import pandas as pd

from src.models.train import get_baseline_models, train_baseline_models
from src.models.evaluate import compare_models


def _make_binary_dataset():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "numeric_a": rng.normal(0, 1, n),
        "numeric_b": rng.normal(2, 1.5, n),
        "category_x": rng.choice(["A", "B", "C"], size=n),
        "category_y": rng.choice(["Low", "High"], size=n),
    })
    y = ((X["numeric_a"] + X["numeric_b"] / 2) > 1.5).astype(int)
    return X, y


def test_baseline_model_registry_contains_expected_models():
    models = get_baseline_models()

    expected = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "naive_bayes",
        "gradient_boosting",
    ]

    assert set(expected).issubset(models.keys())


def test_train_baseline_models_returns_fitted_models():
    X, y = _make_binary_dataset()
    X_train = X.iloc[:140]
    y_train = y.iloc[:140]

    trained = train_baseline_models(X_train, y_train)

    assert set(trained.keys()) == set(get_baseline_models().keys())
    for model in trained.values():
        assert hasattr(model, "predict")


def test_compare_models_returns_metrics_table():
    X, y = _make_binary_dataset()
    X_train = X.iloc[:140]
    X_test = X.iloc[140:]
    y_train = y.iloc[:140]
    y_test = y.iloc[140:]

    results = compare_models(X_train, X_test, y_train, y_test)

    required_columns = {"model", "accuracy", "precision", "recall", "f1", "roc_auc"}
    assert required_columns.issubset(results.columns)
    assert len(results) >= 3
    assert results["accuracy"].between(0, 1).all()
