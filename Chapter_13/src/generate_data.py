from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path("data/sales.csv")


def generate_sales_data(n_days=500, random_state=42):
    rng = np.random.default_rng(random_state)

    dates = pd.date_range(
        start="2025-01-01",
        periods=n_days,
        freq="D",
    )

    trend = np.linspace(400, 550, n_days)

    weekly = 40 * np.sin(
        2 * np.pi * np.arange(n_days) / 7
    )

    noise = rng.normal(
        loc=0,
        scale=20,
        size=n_days,
    )

    sales = trend + weekly + noise
    sales = np.maximum(sales, 0).round(0)

    return pd.DataFrame(
        {
            "date": dates,
            "sales": sales,
        }
    )


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = generate_sales_data()
    df.to_csv(OUTPUT_PATH, index=False)

    print(
        f"Generated {len(df)} rows "
        f"in {OUTPUT_PATH}"
    )