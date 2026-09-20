from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "AccountAge",
    "MonthlyCharges",
    "TotalCharges",
    "SubscriptionType",
    "PaymentMethod",
    "PaperlessBilling",
    "ContentType",
    "MultiDeviceAccess",
    "DeviceRegistered",
    "ViewingHoursPerWeek",
    "AverageViewingDuration",
    "ContentDownloadsPerMonth",
    "GenrePreference",
    "UserRating",
    "SupportTicketsPerMonth",
    "Gender",
    "WatchlistSize",
    "ParentalControl",
    "SubtitlesEnabled",
    "CustomerID",
    "Churn",
]


def validate_dataset(
    df: pd.DataFrame,
    target_column: str = "Churn",
    id_column: str = "CustomerID",
) -> dict[str, Any]:
    """Run core schema and data-quality validation checks for the dataset."""
    required_columns_ok = all(column in df.columns for column in REQUIRED_COLUMNS)
    missing_values_ok = bool(df.isna().sum().sum() == 0)
    duplicate_customer_ids_ok = bool(df[id_column].duplicated().sum() == 0) if id_column in df.columns else False
    target_values_ok = bool(set(df[target_column].dropna().unique()).issubset({0, 1})) if target_column in df.columns else False

    return {
        "required_columns_ok": required_columns_ok,
        "missing_values_ok": missing_values_ok,
        "duplicate_customer_ids_ok": duplicate_customer_ids_ok,
        "target_values_ok": target_values_ok,
        "missing_value_count": int(df.isna().sum().sum()),
        "duplicate_customer_id_count": int(df[id_column].duplicated().sum()) if id_column in df.columns else 0,
        "target_unique_values": sorted(df[target_column].dropna().unique().tolist()) if target_column in df.columns else [],
    }
