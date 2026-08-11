from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


FEATURES = [
    "lag_1",
    "lag_7",
    "rolling_7",
]


def load_data(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    return df.sort_values("date")


def validate_data(df):
    required_columns = {
        "date",
        "sales",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df["sales"].isna().any():
        raise ValueError(
            "Sales contains missing values."
        )

    if (df["sales"] < 0).any():
        raise ValueError(
            "Sales cannot contain negative values."
        )


def prepare_features(df):
    df = df.copy()

    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)

    df["rolling_7"] = (
        df["sales"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    return df.dropna()


def train_model(df):
    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    validation = df.iloc[split:]

    X_train = train[FEATURES]
    y_train = train["sales"]

    X_validation = validation[FEATURES]
    y_validation = validation["sales"]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    return model, mae