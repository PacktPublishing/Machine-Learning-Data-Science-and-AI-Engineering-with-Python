import json
import logging
import time
import uuid
from pathlib import Path

import joblib
import torch
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from app.model import IrisClassifier
from app.schemas import IrisRequest, IrisResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iris_prediction_service")

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "iris_model.pt"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"


app = FastAPI(
    title="Iris Classification API",
    description="A small ML deployment project using PyTorch, FastAPI, monitoring, and Docker.",
    version="1.0.0"
)


PREDICTION_COUNTER = Counter(
    "iris_predictions_total",
    "Total number of predictions by model version and class",
    ["model_version", "predicted_class"]
)

PREDICTION_LATENCY = Histogram(
    "iris_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_version"]
)


def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)


metadata = load_metadata()
MODEL_VERSION = metadata["model_version"]
CLASS_NAMES = metadata["classes"]

model = IrisClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

scaler = joblib.load(SCALER_PATH)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION
    }


@app.get("/metadata")
def get_metadata():
    return metadata


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )


@app.post("/predict", response_model=IrisResponse)
def predict(request: IrisRequest):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        raw_features = [[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]]

        scaled_features = scaler.transform(raw_features)

        features_tensor = torch.tensor(
            scaled_features,
            dtype=torch.float32
        )

        with torch.no_grad():
            logits = model(features_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, class_id = torch.max(probabilities, dim=1)

        predicted_class = CLASS_NAMES[class_id.item()]
        latency = time.time() - start_time

        PREDICTION_COUNTER.labels(
            model_version=MODEL_VERSION,
            predicted_class=predicted_class
        ).inc()

        PREDICTION_LATENCY.labels(
            model_version=MODEL_VERSION
        ).observe(latency)

        logger.info({
            "event": "prediction_success",
            "request_id": request_id,
            "model_version": MODEL_VERSION,
            "predicted_class": predicted_class,
            "confidence": round(confidence.item(), 4),
            "latency_ms": round(latency * 1000, 2)
        })

        return IrisResponse(
            prediction=predicted_class,
            class_id=class_id.item(),
            confidence=round(confidence.item(), 4),
            model_version=MODEL_VERSION,
            request_id=request_id,
            latency_ms=round(latency * 1000, 2)
        )

    except Exception as error:
        logger.exception({
            "event": "prediction_error",
            "request_id": request_id,
            "error": str(error)
        })

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )