from __future__ import annotations

import pandas as pd


class FeatureEngineeringTransformer:
    """Create domain-driven churn features from the raw customer dataset.

    The features are intentionally simple and explainable, based on the business
    signals described in the project README and the dataset columns used in the
    churn problem. They are designed to support the modeling pipeline without
    inventing arbitrary variables.
    """

    def __init__(self) -> None:
        self.feature_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineeringTransformer":
        """Store the feature columns that the transformer will engineer."""
        self.feature_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of X with domain-aligned engineered features added."""
        if X.empty:
            return X.copy()

        output = X.copy()

        numeric_columns = output.select_dtypes(include=["number"]).columns
        id_like_columns = [col for col in output.columns if "customer" in col.lower() or col.lower() == "customerid"]

        for column in numeric_columns:
            if output[column].isna().any():
                output[column] = output[column].fillna(0)

        if {"MonthlyCharges", "ViewingHoursPerWeek", "AverageViewingDuration"}.issubset(output.columns):
            output["EngagementScore"] = (
                output["ViewingHoursPerWeek"]
                * output["AverageViewingDuration"]
                / 10
                + output["MonthlyCharges"]
                / 10
            )

        if {"ViewingHoursPerWeek", "AverageViewingDuration"}.issubset(output.columns):
            output["ViewingIntensity"] = (
                output["ViewingHoursPerWeek"] * output["AverageViewingDuration"]
            )

        if {"ContentDownloadsPerMonth", "MonthlyCharges"}.issubset(output.columns):
            output["DownloadIntensity"] = (
                output["ContentDownloadsPerMonth"] / (output["MonthlyCharges"] + 1e-6)
            )

        if {"SupportTicketsPerMonth", "UserRating"}.issubset(output.columns):
            output["SupportPressure"] = (
                output["SupportTicketsPerMonth"] * (6 - output["UserRating"])
            )

        if {"MonthlyCharges", "ContentDownloadsPerMonth"}.issubset(output.columns):
            output["EstimatedMonthlyValue"] = (
                output["MonthlyCharges"] * (1 + output["ContentDownloadsPerMonth"] / 10)
            )

        if {"ViewingHoursPerWeek", "WatchlistSize"}.issubset(output.columns):
            output["ViewingPerMonth"] = output["ViewingHoursPerWeek"] * output["WatchlistSize"]

        for col in id_like_columns:
            if col in output.columns:
                output = output.drop(columns=[col])

        default_columns = [
            "EngagementScore",
            "ViewingIntensity",
            "DownloadIntensity",
            "SupportPressure",
            "EstimatedMonthlyValue",
            "ViewingPerMonth",
        ]

        for col in default_columns:
            if col not in output.columns:
                output[col] = 0.0

        return output

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit the transformer and return the engineered dataset."""
        return self.fit(X, y).transform(X)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the feature engineering logic to a raw churn dataframe."""
    transformer = FeatureEngineeringTransformer()
    return transformer.fit_transform(df)
