from pathlib import Path

from src.data.load_data import load_raw_data, summarize_dataset
from src.data.validate_data import validate_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "row" / "train.csv"


def test_load_raw_data_returns_dataframe():
    df = load_raw_data(DATA_PATH)

    assert df is not None
    assert not df.empty
    assert "CustomerID" in df.columns
    assert "Churn" in df.columns
    assert df.shape[0] > 0


def test_dataset_summary_reports_key_metadata():
    df = load_raw_data(DATA_PATH)
    summary = summarize_dataset(df, target_column="Churn", id_column="CustomerID")

    assert summary["n_rows"] == df.shape[0]
    assert summary["n_columns"] == df.shape[1]
    assert summary["target_column"] == "Churn"
    assert summary["missing_values"] >= 0
    assert summary["duplicate_rows"] >= 0
    assert summary["target_distribution"]["0"] > 0
    assert summary["target_distribution"]["1"] > 0


def test_validate_dataset_detects_no_schema_issues():
    df = load_raw_data(DATA_PATH)
    result = validate_dataset(df, target_column="Churn", id_column="CustomerID")

    assert result["required_columns_ok"] is True
    assert result["missing_values_ok"] is True
    assert result["duplicate_customer_ids_ok"] is True
    assert result["target_values_ok"] is True
