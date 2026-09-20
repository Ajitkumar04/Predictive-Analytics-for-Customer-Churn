from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature lists for the preprocessing pipeline."""
    numeric_columns = list(df.select_dtypes(include=["number"]).columns)
    categorical_columns = list(df.select_dtypes(exclude=["number"]).columns)
    return numeric_columns, categorical_columns


def create_preprocessing_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer that handles mixed numeric and categorical features."""
    numeric_columns, categorical_columns = get_feature_types(df)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_transformer, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_transformer, categorical_columns))

    if not transformers:
        raise ValueError("DataFrame must contain at least one numeric or categorical feature column.")

    return ColumnTransformer(transformers=transformers, remainder="drop")
