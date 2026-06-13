# Iris Classification API

A complete end-to-end machine learning deployment project built with **PyTorch**, **FastAPI**, and **Docker**.

This project demonstrates many of the concepts covered in this chapter:

- Model training with PyTorch
- Model serialization
- Consistent preprocessing during inference
- REST API deployment using FastAPI
- Request validation with Pydantic
- Prediction logging
- Monitoring with Prometheus metrics
- Health checks
- Docker containerization
- Basic support for A/B testing and model versioning

The deployed service receives four Iris flower measurements and predicts the corresponding species.

---

# Project Structure

```text
iris-api/
├── app/
│   ├── main.py
│   ├── model.py
│   └── schemas.py
├── artifacts/
│   ├── iris_model.pt
│   ├── scaler.joblib
│   └── metadata.json
├── train.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

# Example Request

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

# Example Response

```json
{
  "prediction": "setosa",
  "class_id": 0,
  "confidence": 0.99,
  "model_version": "1.0.0"
}
```

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training the Model

Train the classifier and generate deployment artifacts:

```bash
python train.py
```

This creates:

```text
artifacts/
├── iris_model.pt
├── scaler.joblib
└── metadata.json
```

These files are required by the inference service.

---

# Running the API Locally

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

Alternative documentation:

```text
http://localhost:8000/redoc
```

---

# Testing Predictions

Example request using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'
```

Expected output:

```json
{
  "prediction": "setosa",
  "class_id": 0,
  "confidence": 0.99,
  "model_version": "1.0.0",
  "request_id": "54b9a69d-d0a3-4c27-a17a-b532c0435e36",
  "latency_ms": 15.78
}
```

Another example:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "sepal_length": 6.7,
  "sepal_width": 3.0,
  "petal_length": 5.2,
  "petal_width": 2.3
}'
```

Expected prediction:

```json
{
  "prediction": "virginica",
  "class_id": 2,
  "confidence": 0.97,
  "model_version": "1.0.0",
  "request_id": "54b9a69d-d0a3-4c27-a17a-a647d0325e81",
  "latency_ms": 11.21
}
```

---

# Available Endpoints

## Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_version": "1.0.0"
}
```

---

## Model Metadata

```http
GET /metadata
```

Returns model version, class names, and training metadata.

---

## Prediction Endpoint

```http
POST /predict
```

Performs inference using the deployed model.

---

## Prometheus Metrics

```http
GET /metrics
```

Exposes monitoring metrics such as:

- Prediction counts
- Prediction latency
- Model version statistics

---

# Docker Deployment

## Build the Docker Image

```bash
docker build -t iris-classifier-api .
```

## Run the Container

```bash
docker run -p 8000:8000 iris-classifier-api
```

The service becomes available at:

```text
http://localhost:8000
```

Documentation:

```text
http://localhost:8000/docs
```

---

# Docker Health Check

The Docker image includes a health check based on:

```http
GET /health
```

Container orchestration systems can use this endpoint to verify that the service is operational.

---

# Monitoring

The project includes:

- Structured prediction logging
- Latency monitoring
- Prediction counters
- Prometheus metrics export

Example log entry:

```json
{
  "event": "prediction_success",
  "model_version": "1.0.0",
  "predicted_class": "setosa",
  "confidence": 0.99,
  "latency_ms": 2.4
}
```

---

# A/B Testing and Model Versioning

The project is designed to support multiple model versions.

A simple routing function can be used to direct a percentage of traffic to a newer model:

```python
import random

def choose_model_version():

    if random.random() < 0.10:
        return "v2"

    return "v1"
```

This enables:

- Canary deployments
- A/B testing
- Shadow deployments
- Safe rollouts

without changing the API contract.

---

# Learning Objectives

With this project, you learn about how to:

- Train a PyTorch model
- Save and load model artifacts
- Deploy inference through FastAPI
- Validate requests with Pydantic
- Monitor prediction services
- Add logging and observability
- Containerize applications using Docker
- Prepare services for production deployment
- Implement basic model rollout strategies

Although the example uses the Iris dataset, the same deployment pattern can be applied to image classifiers, recommendation systems, fraud detectors, customer churn models, and many other machine learning applications.