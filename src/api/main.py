
import os

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from dotenv import load_dotenv

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
)


# ============================================================
# Environment Configuration
# ============================================================

load_dotenv()


# ============================================================
# MLflow Configuration
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)

print("=" * 70)
print("MLflow Configuration")
print("=" * 70)
print(f"MLflow Tracking URI : {MLFLOW_TRACKING_URI}")

# Set MLflow Tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ============================================================
# Model Configuration
# ============================================================

MODEL_NAME = "fraud-detection-model"
MODEL_ALIAS = "champion"

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


# ============================================================
# Load Champion Model
# ============================================================

print("=" * 70)
print("Loading MLflow model...")
print(f"MLflow Tracking URI : {MLFLOW_TRACKING_URI}")
print(f"Model Name          : {MODEL_NAME}")
print(f"Model Alias         : {MODEL_ALIAS}")
print(f"Model URI           : {MODEL_URI}")
print("=" * 70)


try:
    # --------------------------------------------------------
    # Load model directly from MLflow Model Registry
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # We intentionally do NOT use download_artifacts()
    # here.
    #
    # This avoids converting a Windows path such as:
    #
    # C:\Users\...
    #
    # into an MLflow artifact URI.
    #
    # MLflow will resolve the model URI and download the
    # required artifacts using its configured artifact store.
    # --------------------------------------------------------

    model = mlflow.pyfunc.load_model(
        MODEL_URI
    )

    print("=" * 70)
    print("Model loaded successfully!")
    print(f"Loaded model: {MODEL_NAME}@{MODEL_ALIAS}")
    print("=" * 70)


except Exception as e:

    print("=" * 70)
    print("ERROR: Failed to load MLflow model")
    print(f"Model URI: {MODEL_URI}")
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error: {e}")
    print("=" * 70)

    raise


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Fraud Detection API",
    description="Fraud detection inference API powered by MLflow",
    version="1.0.0",
)


# ============================================================
# Prometheus Metrics
# ============================================================

Instrumentator().instrument(app).expose(app)


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
# Prediction Endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    # --------------------------------------------------------
    # Convert request to DataFrame
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
    # API Response
    # --------------------------------------------------------

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),
    }
