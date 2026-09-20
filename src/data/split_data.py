from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    target_column: str = "Churn",
    id_column: str | None = "CustomerID",
    test_size: float = 0.2,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Split the churn dataset into train/validation/test sets while preserving class balance."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found.")

    X = df.drop(columns=[target_column])
    if id_column and id_column in X.columns:
        X = X.drop(columns=[id_column])
    y = df[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    validation_fraction = validation_size / (1.0 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1.0 - validation_fraction,
        stratify=y_temp,
        random_state=random_state,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
