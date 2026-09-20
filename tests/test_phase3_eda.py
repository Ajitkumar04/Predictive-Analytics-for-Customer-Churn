from src.data.load_data import load_raw_data
from src.eda.eda_analysis import (
    build_eda_report,
    compute_correlation_summary,
    compute_missing_value_summary,
    compute_target_distribution,
    summarize_categorical_features,
    summarize_numeric_features,
)


DATA_PATH = "data/row/train.csv"


def test_target_distribution_summary_is_valid():
    df = load_raw_data(DATA_PATH)
    summary = compute_target_distribution(df, target_column="Churn")

    assert summary["total_rows"] == df.shape[0]
    assert summary["class_counts"][0] > 0
    assert summary["class_counts"][1] > 0
    assert summary["positive_rate"] > 0
    assert summary["positive_rate"] < 1


def test_numeric_and_categorical_summaries_are_generated():
    df = load_raw_data(DATA_PATH)

    numeric_summary = summarize_numeric_features(df, exclude_columns=["Churn", "CustomerID"])
    categorical_summary = summarize_categorical_features(df, exclude_columns=["Churn", "CustomerID"])

    assert len(numeric_summary) > 0
    assert len(categorical_summary) > 0
    assert "MonthlyCharges" in numeric_summary
    assert "SubscriptionType" in categorical_summary


def test_missing_values_and_correlation_checks_are_available():
    df = load_raw_data(DATA_PATH)

    missing_summary = compute_missing_value_summary(df)
    corr_summary = compute_correlation_summary(df, target_column="Churn")

    assert missing_summary["total_missing"] >= 0
    assert len(corr_summary["top_positive_features"]) >= 0
    assert len(corr_summary["top_negative_features"]) >= 0


def test_eda_report_contains_expected_sections():
    df = load_raw_data(DATA_PATH)
    report = build_eda_report(df, target_column="Churn", id_column="CustomerID")

    assert "dataset_shape" in report
    assert "target_distribution" in report
    assert "missing_values" in report
    assert "numeric_summary" in report
    assert "categorical_summary" in report
    assert "correlation_summary" in report
