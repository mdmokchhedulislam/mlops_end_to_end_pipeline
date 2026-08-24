import os

import mlflow
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# from src.train import get_models, train_model

from src.training.train import get_models, train_model


def test_get_models():
    models = get_models()

    expected_models = {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    }

    assert set(models.keys()) == expected_models
    assert len(models) == 5


def test_train_model(tmp_path, monkeypatch):
    # -------------------------
    # Create test dataset
    # -------------------------

    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    X = pd.DataFrame(
        X,
        columns=["feature_1", "feature_2", "feature_3", "feature_4"],
    )

    y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # -------------------------
    # Use temporary model directory
    # -------------------------

    model_dir = tmp_path / "models"
    model_dir.mkdir()

    monkeypatch.setattr(
        "src.training.train.MODEL_DIR",
        str(model_dir),
    )

    # -------------------------
    # Prevent real MLflow model upload
    # -------------------------

    monkeypatch.setattr(
        mlflow.sklearn,
        "log_model",
        lambda *args, **kwargs: None,
    )

    # -------------------------
    # Use local MLflow tracking
    # -------------------------

    mlflow.set_tracking_uri(
        f"file://{tmp_path}/mlruns"
    )

    mlflow.set_experiment(
        "test-fraud-detection"
    )

    # -------------------------
    # Train model
    # -------------------------

    model = LogisticRegression(
        max_iter=1000
    )

    train_model(
        "test_logistic_regression",
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # -------------------------
    # Check model file
    # -------------------------

    model_path = (
        model_dir / "test_logistic_regression.pkl"
    )

    assert model_path.exists()
    assert model_path.stat().st_size > 0