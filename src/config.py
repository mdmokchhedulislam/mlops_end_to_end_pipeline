import os

# ============================================================
# Data Configuration
# ============================================================

PROCESSED_DATA_DIR = os.getenv(
    "PROCESSED_DATA_DIR",
    "data/processed",
)

X_TRAIN_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "X_train.csv",
)

X_TEST_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "X_test.csv",
)

Y_TRAIN_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "y_train.csv",
)

Y_TEST_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "y_test.csv",
)


# ============================================================
# MLflow
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT",
    "fraud-detection",
)

MLFLOW_MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "fraud-detection-model",
)

MLFLOW_MODEL_ALIAS = os.getenv(
    "MLFLOW_MODEL_ALIAS",
    "champion",
)


# ============================================================
# Quality Gate Thresholds
# ============================================================

MIN_ACCURACY = float(
    os.getenv(
        "MIN_ACCURACY",
        "0.90",
    )
)

MIN_PRECISION = float(
    os.getenv(
        "MIN_PRECISION",
        "0.90",
    )
)

MIN_RECALL = float(
    os.getenv(
        "MIN_RECALL",
        "0.85",
    )
)

MIN_F1_SCORE = float(
    os.getenv(
        "MIN_F1_SCORE",
        "0.90",
    )
)