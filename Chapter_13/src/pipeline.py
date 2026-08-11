import mlflow
import mlflow.sklearn

from prefect import flow, task

from src.train import (
    load_data,
    prepare_features,
    train_model,
    validate_data,
)


DATA_PATH = "data/sales.csv"
EXPERIMENT_NAME = "sales-forecasting"

REGISTERED_MODEL_NAME = "SalesForecastModel"

MAX_ALLOWED_MAE = 50.0


@task(retries=2, retry_delay_seconds=5)
def load_and_prepare():
    df = load_data(DATA_PATH)

    validate_data(df)

    df = prepare_features(df)

    return df


@task
def train(df):
    model, mae = train_model(df)

    return model, mae


@task
def evaluate_and_register(model, mae):
    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run() as run:
        mlflow.log_metric(
            "validation_mae",
            mae,
        )

        mlflow.log_param(
            "model_type",
            "RandomForestRegressor",
        )

        mlflow.log_param(
            "quality_threshold",
            MAX_ALLOWED_MAE,
        )

        print(
            f"Validation MAE: {mae:.2f}"
        )

        if mae > MAX_ALLOWED_MAE:
            print(
                "Candidate rejected: "
                "MAE exceeds threshold."
            )

            return {
                "registered": False,
                "mae": mae,
                "run_id": run.info.run_id,
            }

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=(
                REGISTERED_MODEL_NAME
            ),
        )

        print(
            "Candidate registered as "
            f"{REGISTERED_MODEL_NAME}"
        )

        return {
            "registered": True,
            "mae": mae,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
        }


@flow(name="sales-forecast-training")
def forecasting_pipeline():
    df = load_and_prepare()

    model, mae = train(df)

    result = evaluate_and_register(
        model,
        mae,
    )

    return result


if __name__ == "__main__":
    result = forecasting_pipeline()

    print(result)