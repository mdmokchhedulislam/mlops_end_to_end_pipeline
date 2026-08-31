import os

import mlflow
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
)

# ============================================================
# MLflow Configuration
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = "models:/fraud-detection-model/1"

print("=" * 70)
print("Loading MLflow model...")
print(f"Model URI: {MODEL_URI}")
print("=" * 70)

model = mlflow.sklearn.load_model(MODEL_URI)

print("Model loaded successfully!")
print("=" * 70)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
)


# ============================================================
# Prometheus
# ============================================================

Instrumentator().instrument(app).expose(app)


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():
    return {
        "service": "fraud-detection-api",
        "version": "1.0.0",
        "model": "fraud-detection-model",
        "model_version": "1",
    }


# ============================================================
# Prediction
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    # Convert request into DataFrame
    input_data = pd.DataFrame([
        {
            "amount": request.amount,
            "account_age": request.account_age,
            "transaction_count": request.transaction_count,
        }
    ])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),
    }