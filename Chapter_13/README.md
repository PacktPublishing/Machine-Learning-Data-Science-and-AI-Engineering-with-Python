# Sales Forecasting MLOps Pipeline

A compact example combining DVC, Prefect, MLflow, model registration, and a simple retraining trigger.

## Install

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Generate the Data
```bash
python -m src.generate_data
```

This creates data/sales.csv.

## Initialize DVC

```bash
git init
dvc init
dvc add data/sales.csv
```

For a simple local DVC remote:

```bash
mkdir dvc-storage
dvc remote add -d storage ./dvc-storage
dvc push
```

## Run the Tests
```bash
pytest
```

## Start MLflow

Run the MLflow server in a separate terminal:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Set the tracking URI on Linux/macOS:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

or Windows PowerShell:

```bash
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

The MLflow UI is available at:

http://127.0.0.1:5000

## Run the Training Pipeline

```bash
python -m src.pipeline
```

Prefect loads and validates the data, creates the forecasting features, trains the model, and evaluates its validation MAE. Models satisfying the quality threshold are logged and registered in MLflow as SalesForecastModel.

## Simulate Monitoring and Retraining

```bash
python -m src.monitor
```

The example calculates the MAE of simulated production predictions. If it exceeds the configured threshold, the training pipeline is triggered again.

## Update the Versioned Data

After changing data/sales.csv:

```bash
dvc add data/sales.csv
dvc push
```

Commit the updated DVC metadata together with the corresponding source-code changes:


```bash
git add .
git commit -m "Update sales forecasting pipeline"
```
