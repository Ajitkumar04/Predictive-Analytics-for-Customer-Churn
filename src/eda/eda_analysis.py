from __future__ import annotations

from typing import Any

import pandas as pd


def compute_target_distribution(df: pd.DataFrame, target_column: str = "Churn") -> dict[str, Any]:
    """Compute the class balance summary for the target variable."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe.")

    counts = df[target_column].value_counts().sort_index().to_dict()
    total_rows = int(df.shape[0])
    positive_rate = float(counts.get(1, 0) / total_rows) if total_rows else 0.0

    return {
        "total_rows": total_rows,
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "positive_rate": positive_rate,
        "negative_rate": 1.0 - positive_rate,
    }


def summarize_numeric_features(df: pd.DataFrame, exclude_columns: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Summarize numeric feature distributions with key statistics."""
    exclude_columns = set(exclude_columns or [])
    numeric_columns = [col for col in df.select_dtypes(include=["number"]).columns if col not in exclude_columns]

    summary: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        series = df[column]
        summary[column] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
            "missing": int(series.isna().sum()),
        }
    return summary


def summarize_categorical_features(df: pd.DataFrame, exclude_columns: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Summarize categorical feature value distributions."""
    exclude_columns = set(exclude_columns or [])
    categorical_columns = [col for col in df.select_dtypes(exclude=["number"]).columns if col not in exclude_columns]

    summary: dict[str, dict[str, Any]] = {}
    for column in categorical_columns:
        counts = df[column].value_counts(dropna=False).to_dict()
        summary[column] = {
            "unique_values": int(df[column].nunique(dropna=False)),
            "top_values": {str(k): int(v) for k, v in list(counts.items())[:5]},
            "missing": int(df[column].isna().sum()),
        }
    return summary


def compute_missing_value_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return total and per-column missing value statistics."""
    missing_counts = df.isna().sum().to_dict()
    return {
        "total_missing": int(sum(missing_counts.values())),
        "per_column": {str(k): int(v) for k, v in missing_counts.items()},
    }


def compute_correlation_summary(df: pd.DataFrame, target_column: str = "Churn") -> dict[str, Any]:
    """Compute target correlation summary and rank the strongest positive/negative numeric features."""
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if target_column not in numeric_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in numeric features.")

    corr = numeric_df.corr(numeric_only=True)[target_column].drop(target_column).sort_values(ascending=False)
    top_positive = corr[corr > 0].head(5).to_dict()
    top_negative = corr[corr < 0].tail(5).sort_values(ascending=True).to_dict()

    return {
        "top_positive_features": {str(k): float(v) for k, v in top_positive.items()},
        "top_negative_features": {str(k): float(v) for k, v in top_negative.items()},
    }


def build_eda_report(df: pd.DataFrame, target_column: str = "Churn", id_column: str | None = None) -> dict[str, Any]:
    """Build a reusable EDA report structure for analysis and reporting."""
    report = {
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "target_distribution": compute_target_distribution(df, target_column=target_column),
        "missing_values": compute_missing_value_summary(df),
        "numeric_summary": summarize_numeric_features(df, exclude_columns=[target_column, id_column] if id_column else [target_column]),
        "categorical_summary": summarize_categorical_features(df, exclude_columns=[target_column, id_column] if id_column else [target_column]),
        "correlation_summary": compute_correlation_summary(df, target_column=target_column),
    }
    return report
