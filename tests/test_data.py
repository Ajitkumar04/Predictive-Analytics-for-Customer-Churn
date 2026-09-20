from __future__ import annotations

import pandas as pd

from src.data.load_data import load_raw_data, summarize_dataset
from src.data.validate_data import validate_dataset


def test_load_raw_data_reads_csv(tmp_path):
    csv_path = tmp_path / "customer_churn_sample.csv"
    sample = pd.DataFrame(
        {
            "CustomerID": [1, 2],
            "AccountAge": [12, 18],
            "MonthlyCharges": [15.0, 25.0],
            "TotalCharges": [180.0, 300.0],
            "SubscriptionType": ["Basic", "Premium"],
            "PaymentMethod": ["Credit Card", "Bank Transfer"],
            "PaperlessBilling": ["Yes", "No"],
            "ContentType": ["Movies", "Series"],
            "MultiDeviceAccess": ["Yes", "Yes"],
            "DeviceRegistered": ["TV", "Phone"],
            "ViewingHoursPerWeek": [10.5, 14.0],
            "AverageViewingDuration": [30.0, 42.0],
            "ContentDownloadsPerMonth": [2, 5],
            "GenrePreference": ["Drama", "Comedy"],
            "UserRating": [3, 4],
            "SupportTicketsPerMonth": [1, 0],
            "Gender": ["Male", "Female"],
            "WatchlistSize": [5, 8],
            "ParentalControl": ["No", "Yes"],
            "SubtitlesEnabled": ["Yes", "No"],
            "Churn": [0, 1],
        }
    )
    sample.to_csv(csv_path, index=False)

    loaded = load_raw_data(csv_path)

    assert list(loaded.columns) == list(sample.columns)
    assert loaded.shape == (2, 22)


def test_validate_dataset_reports_clean_schema():
    df = pd.DataFrame(
        {
            "CustomerID": [101, 102, 103],
            "AccountAge": [12, 18, 22],
            "MonthlyCharges": [15.0, 20.0, 18.5],
            "TotalCharges": [180.0, 240.0, 222.0],
            "SubscriptionType": ["Basic", "Premium", "Basic"],
            "PaymentMethod": ["Credit Card", "Bank Transfer", "Credit Card"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "ContentType": ["Movies", "Series", "Movies"],
            "MultiDeviceAccess": ["Yes", "No", "Yes"],
            "DeviceRegistered": ["TV", "Phone", "Tablet"],
            "ViewingHoursPerWeek": [10.5, 14.0, 11.0],
            "AverageViewingDuration": [30.0, 41.0, 33.0],
            "ContentDownloadsPerMonth": [2, 6, 3],
            "GenrePreference": ["Drama", "Comedy", "Drama"],
            "UserRating": [3, 4, 2],
            "SupportTicketsPerMonth": [1, 0, 2],
            "Gender": ["Male", "Female", "Male"],
            "WatchlistSize": [5, 9, 6],
            "ParentalControl": ["No", "Yes", "No"],
            "SubtitlesEnabled": ["Yes", "No", "Yes"],
            "Churn": [0, 1, 0],
        }
    )

    report = validate_dataset(df)

    assert report["required_columns_ok"] is True
    assert report["missing_values_ok"] is True
    assert report["duplicate_customer_ids_ok"] is True
    assert report["target_values_ok"] is True
    assert report["missing_value_count"] == 0
    assert report["duplicate_customer_id_count"] == 0
    assert report["target_unique_values"] == [0, 1]


def test_summarize_dataset_reports_key_metadata():
    df = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4],
            "AccountAge": [12, 18, 22, 15],
            "MonthlyCharges": [15.0, 25.0, 18.0, 30.0],
            "Churn": [0, 1, 0, 1],
        }
    )

    summary = summarize_dataset(df, target_column="Churn", id_column="CustomerID")

    assert summary["n_rows"] == 4
    assert summary["n_columns"] == 4
    assert summary["target_distribution"] == {"0": 2, "1": 2}
    assert "CustomerID" in summary["columns"]
    assert summary["duplicate_customer_ids"] == 0
