from sklearn.metrics import mean_absolute_error

from src.pipeline import forecasting_pipeline


RETRAINING_THRESHOLD = 60.0


def check_model_performance(
    actual,
    predicted,
):
    mae = mean_absolute_error(
        actual,
        predicted,
    )

    print(
        f"Recent production MAE: {mae:.2f}"
    )

    if mae > RETRAINING_THRESHOLD:
        print(
            "Performance threshold exceeded."
        )
        print(
            "Starting retraining pipeline..."
        )

        return forecasting_pipeline()

    print(
        "Model performance is acceptable."
    )

    return None


if __name__ == "__main__":
    # Simulated recent production observations.
    actual = [
        470,
        510,
        495,
        530,
        560,
    ]

    predicted = [
        400,
        430,
        420,
        445,
        460,
    ]

    check_model_performance(
        actual,
        predicted,
    )