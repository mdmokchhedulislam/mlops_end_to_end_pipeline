import os

import mlflow
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from dotenv import load_dotenv
load_dotenv()

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
)


# ============================================================
# MLflow Configuration
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)

print("url is", MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model using MLflow "champion" alias
MODEL_NAME = "fraud-detection-model"
MODEL_ALIAS = "champion"

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


print("=" * 70)
print("Loading MLflow model...")
print(f"MLflow Tracking URI : {MLFLOW_TRACKING_URI}")
print(f"Model Name          : {MODEL_NAME}")
print(f"Model Alias         : {MODEL_ALIAS}")
print(f"Model URI            : {MODEL_URI}")
print("=" * 70)


# ============================================================
# Load Champion Model
# ============================================================

try:
    model = mlflow.sklearn.load_model(MODEL_URI)

    print("Model loaded successfully!")
    print(f"Loaded model: {MODEL_NAME}@{MODEL_ALIAS}")

except Exception as e:
    print("=" * 70)
    print("ERROR: Failed to load MLflow model")
    print(f"Model URI: {MODEL_URI}")
    print(f"Error: {e}")
    print("=" * 70)

    raise


print("=" * 70)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
)


# ============================================================
# Prometheus Metrics
# ============================================================

Instrumentator().instrument(app).expose(app)


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
    }


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "service": "fraud-detection-api",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }


# ============================================================
# Prediction
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    # --------------------------------------------------------
    # Convert request into DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                "amount": request.amount,
                "account_age": request.account_age,
                "transaction_count": request.transaction_count,
            }
        ]
    )

    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    # --------------------------------------------------------
    # Fraud Probability
    # --------------------------------------------------------

    probability = model.predict_proba(input_data)[0][1]

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),
    }