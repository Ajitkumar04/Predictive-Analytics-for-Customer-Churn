from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.serialization import load_model_artifact, save_model_artifact


def test_model_artifact_round_trip_saves_and_loads(tmp_path):
    X = pd.DataFrame(
        {
            "monthly_charges": [10.0, 20.0, 35.0, 50.0],
            "subscription_type": ["Basic", "Premium", "Basic", "Premium"],
        }
    )
    y = pd.Series([0, 0, 1, 1])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["monthly_charges"],
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["subscription_type"],
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )
    model.fit(X, y)

    artifact = save_model_artifact(
        model=model,
        output_dir=tmp_path / "artifacts",
        model_name="churn_model",
        metadata={"model_name": "logistic_regression", "version": "v1.0.0"},
    )

    assert artifact["model_path"].exists()
    assert artifact["metadata_path"].exists()

    loaded_model = load_model_artifact(artifact["model_path"])
    predictions = loaded_model.predict(X)
    probabilities = loaded_model.predict_proba(X)

    assert set(predictions.tolist()).issubset({0, 1})
    assert probabilities.shape == (len(X), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))

    metadata = artifact["metadata_path"].read_text(encoding="utf-8")
    assert "logistic_regression" in metadata
