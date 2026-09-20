import numpy as np
import pandas as pd

from src.models.explainability import explain_prediction, summarize_feature_importance


def test_summarize_feature_importance_returns_sorted_results():
    feature_names = ["support_tickets", "monthly_charges", "account_age", "user_rating"]
    importances = np.array([0.4, 0.3, 0.2, 0.1])

    result = summarize_feature_importance(feature_names, importances)

    assert list(result["feature"]) == ["support_tickets", "monthly_charges", "account_age", "user_rating"]
    assert result["importance"].iloc[0] == 0.4
    assert result["importance"].sum() == 1.0


def test_explain_prediction_returns_contribution_summary():
    X = pd.DataFrame(
        {
            "support_tickets": [2],
            "monthly_charges": [45.0],
            "account_age": [12],
            "user_rating": [2],
        }
    )

    explanation = explain_prediction(X, probability=0.82, top_n=3)

    assert "prediction_probability" in explanation
    assert explanation["prediction_probability"] == 0.82
    assert "top_features" in explanation
    assert len(explanation["top_features"]) <= 3
    assert all("feature" in item and "contribution" in item for item in explanation["top_features"])
