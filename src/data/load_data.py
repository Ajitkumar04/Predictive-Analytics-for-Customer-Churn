from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw churn dataset and validate it can be read as a DataFrame."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Dataset at {file_path} is empty.")

    return df


def summarize_dataset(
    df: pd.DataFrame,
    target_column: str,
    id_column: str | None = None,
) -> dict[str, Any]:
    """Return key information about the dataset for the data understanding phase."""
    target_distribution = df[target_column].value_counts().astype(int).to_dict()
    if not target_distribution:
        target_distribution = {}

    summary: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "target_column": target_column,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df[id_column].duplicated().sum()) if id_column and id_column in df.columns else 0,
        "target_distribution": {str(k): int(v) for k, v in target_distribution.items()},
        "columns": list(df.columns),
        "numerical_columns": list(df.select_dtypes(include=["number"]).columns),
        "categorical_columns": list(df.select_dtypes(exclude=["number"]).columns),
    }
    return summary
