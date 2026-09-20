import json

import pandas as pd

from src.models.serialization import load_model_artifact, save_model_artifact


def test_save_and_reload_model_artifact(tmp_path):
    X = pd.DataFrame(
        {
            "monthly_charges": [15.0, 25.0, 40.0, 60.0],
            "subscription_type": ["Basic", "Premium", "Premium", "Basic"],
        }
    )
    y = pd.Series([0, 0, 1, 1])

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                ["monthly_charges"],
            ),
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]),
                ["subscription_type"],
            ),
        ]
    )

    model = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=500))])
    model.fit(X, y)

    artifacts_dir = tmp_path / "models"
    result = save_model_artifact(
        model=model,
        output_dir=artifacts_dir,
        model_name="churn_model",
        metadata={"model_name": "logistic_regression", "version": "v1.0.0"},
    )

    assert result["model_path"].exists()
    assert result["metadata_path"].exists()

    loaded_model = load_model_artifact(result["model_path"])
    assert hasattr(loaded_model, "predict")

    metadata = json.loads(result["metadata_path"].read_text())
    assert metadata["model_name"] == "logistic_regression"
