import pandas as pd

from src.data.features.feature_engineering import FeatureEngineeringTransformer, engineer_features


def test_engineer_features_adds_business_relevant_columns():
    df = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            "AccountAge": [12, 36, 48],
            "MonthlyCharges": [20.0, 35.0, 50.0],
            "ViewingHoursPerWeek": [8, 12, 20],
            "AverageViewingDuration": [30, 45, 60],
            "ContentDownloadsPerMonth": [2, 5, 10],
            "SupportTicketsPerMonth": [1, 3, 6],
            "UserRating": [4, 3, 2],
            "WatchlistSize": [10, 8, 15],
        }
    )

    transformed = engineer_features(df)

    expected_columns = {
        "EngagementScore",
        "ViewingIntensity",
        "DownloadIntensity",
        "SupportPressure",
        "EstimatedMonthlyValue",
        "ViewingPerMonth",
    }

    assert expected_columns.issubset(transformed.columns)
    assert transformed["EngagementScore"].notna().all()
    assert transformed["ViewingIntensity"].ge(0).all()
    assert transformed["SupportPressure"].ge(0).all()


def test_feature_engineering_transformer_fit_transform_returns_dataframe():
    df = pd.DataFrame(
        {
            "AccountAge": [10, 20, 30],
            "MonthlyCharges": [15.0, 25.0, 35.0],
            "ViewingHoursPerWeek": [5, 10, 15],
            "AverageViewingDuration": [20, 30, 40],
            "ContentDownloadsPerMonth": [1, 2, 3],
            "SupportTicketsPerMonth": [0, 2, 4],
            "UserRating": [5, 4, 2],
        }
    )

    transformer = FeatureEngineeringTransformer()
    transformed = transformer.fit_transform(df)

    assert isinstance(transformed, pd.DataFrame)
    assert set({"EngagementScore", "EstimatedMonthlyValue"}).issubset(transformed.columns)
    assert transformed.shape[0] == df.shape[0]
