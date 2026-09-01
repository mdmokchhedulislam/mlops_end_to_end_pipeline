import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

# ==========================================
# Configuration
# ==========================================

MLFLOW_TRACKING_URI = "http://localhost:5000"

MODEL_NAME = "fraud-detection-model"

MODEL_VERSION = "1"


# ==========================================
# MLflow
# ==========================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ==========================================
# Load Registered Model
# ==========================================

MODEL_URI = (
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)

model = mlflow.pyfunc.load_model(
    MODEL_URI
)


# ==========================================
# FastAPI
# ==========================================

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0"
)


# ==========================================
# Request Schema
# ==========================================

class Transaction(BaseModel):

    amount: float

    account_age: int

    transaction_count: int


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


# ==========================================
# Prediction
# ==========================================

@app.post("/predict")
def predict(
    transaction: Transaction
):

    input_data = [[
        transaction.amount,
        transaction.account_age,
        transaction.transaction_count
    ]]

    prediction = model.predict(
        input_data
    )

    return {
        "prediction": int(prediction[0]),
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }