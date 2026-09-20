from pathlib import Path

import pandas as pd

from src.data.load_data import load_raw_data
from src.data.split_data import split_dataset
from src.preprocessing.preprocessing_pipeline import create_preprocessing_pipeline


DATA_PATH = Path("data/row/train.csv")


def test_split_dataset_returns_train_validation_and_test_sets():
    df = load_raw_data(DATA_PATH)
    split = split_dataset(
        df,
        target_column="Churn",
        id_column="CustomerID",
        test_size=0.2,
        validation_size=0.15,
        random_state=42,
    )

    assert set(split.keys()) == {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test"}
    assert split["X_train"].shape[0] > 0
    assert split["X_val"].shape[0] > 0
    assert split["X_test"].shape[0] > 0
    assert split["y_train"].shape[0] == split["X_train"].shape[0]
    assert split["y_val"].shape[0] == split["X_val"].shape[0]
    assert split["y_test"].shape[0] == split["X_test"].shape[0]


def test_preprocessor_handles_mixed_numeric_and_categorical_data():
    df = load_raw_data(DATA_PATH)
    X = df.drop(columns=["Churn", "CustomerID"])

    preprocessor = create_preprocessing_pipeline(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == X.shape[0]
    assert transformed.shape[1] > 0
