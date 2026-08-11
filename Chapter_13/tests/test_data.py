import pandas as pd

from src.train import (
    prepare_features,
    validate_data,
)


def sample_data():
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=20,
                freq="D",
            ),
            "sales": range(100, 120),
        }
    )


def test_validation():
    df = sample_data()

    validate_data(df)


def test_feature_creation():
    df = sample_data()

    result = prepare_features(df)

    expected = {
        "lag_1",
        "lag_7",
        "rolling_7",
    }

    assert expected.issubset(
        result.columns
    )


def test_features_have_no_missing_values():
    df = sample_data()

    result = prepare_features(df)

    assert not result[
        [
            "lag_1",
            "lag_7",
            "rolling_7",
        ]
    ].isna().any().any()