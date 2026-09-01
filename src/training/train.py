
# # test_train.py

# from pathlib import Path

# import mlflow
# import mlflow.sklearn
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     precision_score,
#     recall_score,
# )
# from sklearn.model_selection import train_test_split


# # ============================================================
# # Configuration
# # ============================================================

# MLFLOW_TRACKING_URI = "http://192.168.1.112:5000"
# EXPERIMENT_NAME = "mlflow-metrics"


# # ============================================================
# # MLflow Setup
# # ============================================================

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# mlflow.set_experiment(EXPERIMENT_NAME)


# # ============================================================
# # Dataset
# # ============================================================

# X, y = make_classification(
#     n_samples=2000,
#     n_features=5,
#     n_informative=4,
#     n_redundant=1,
#     n_classes=2,
#     random_state=42,
# )

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y,
# )


# # ============================================================
# # Model
# # ============================================================

# model = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=10,
#     random_state=42,
#     n_jobs=-1,
# )


# # ============================================================
# # Training + MLflow Logging
# # ============================================================

# with mlflow.start_run(run_name="random_forest_metrics_test") as run:

#     # --------------------------------------------------------
#     # Parameters
#     # --------------------------------------------------------

#     mlflow.log_params(
#         {
#             "algorithm": "random_forest",
#             "n_estimators": 100,
#             "max_depth": 10,
#             "random_state": 42,
#             "training_samples": len(X_train),
#             "test_samples": len(X_test),
#             "feature_count": X.shape[1],
#         }
#     )

#     # --------------------------------------------------------
#     # Train
#     # --------------------------------------------------------

#     model.fit(X_train, y_train)

#     # --------------------------------------------------------
#     # Prediction
#     # --------------------------------------------------------

#     y_pred = model.predict(X_test)

#     # --------------------------------------------------------
#     # Metrics
#     # --------------------------------------------------------

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     # --------------------------------------------------------
#     # IMPORTANT:
#     # These are RUN LEVEL metrics.
#     # --------------------------------------------------------

#     mlflow.log_metrics(
#         {
#             "accuracy": accuracy,
#             "precision": precision,
#             "recall": recall,
#             "f1_score": f1,
#         }
#     )

#     # --------------------------------------------------------
#     # Tags
#     # --------------------------------------------------------

#     mlflow.set_tags(
#         {
#             "project": "mlflow-metrics-test",
#             "pipeline_stage": "training",
#             "model_name": "random_forest",
#         }
#     )

#     # --------------------------------------------------------
#     # Log Model
#     # --------------------------------------------------------

#     model_info = mlflow.sklearn.log_model(
#         sk_model=model,
#         name="random_forest_model",
#     )

#     # --------------------------------------------------------
#     # Print useful information
#     # --------------------------------------------------------

#     print("=" * 70)
#     print("MLflow Training Test Completed")
#     print("=" * 70)

#     print(f"Run ID       : {run.info.run_id}")
#     print(f"Experiment ID: {run.info.experiment_id}")

#     print()
#     print("Metrics")
#     print("-" * 70)
#     print(f"Accuracy     : {accuracy:.4f}")
#     print(f"Precision    : {precision:.4f}")
#     print(f"Recall       : {recall:.4f}")
#     print(f"F1 Score     : {f1:.4f}")

#     print()
#     print("Model")
#     print("-" * 70)
#     print(f"Model URI    : {model_info.model_uri}")

#     if hasattr(model_info, "model_id"):
#         print(f"Model ID     : {model_info.model_id}")

#     print("=" * 70)



















import logging
import os
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
)

# ============================================================
# Logging
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# Models
# ============================================================


def get_models() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


# ============================================================
# Load Data
# ============================================================


def load_data(
    x_train_path: str,
    x_test_path: str,
    y_train_path: str,
    y_test_path: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    required_files = [
        x_train_path,
        x_test_path,
        y_train_path,
        y_test_path,
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# Log Parameters
# ============================================================


def log_model_parameters(
    model: Any,
) -> None:
    for name, value in model.get_params().items():
        if value is None:
            continue

        try:
            mlflow.log_param(
                name,
                str(value),
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Failed to log MLflow parameter '%s': %s",
                name,
                error,
            )


# ============================================================
# Train One Model
# ============================================================


def train_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    with mlflow.start_run(run_name=name) as run:

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        mlflow.set_tags(
            {
                "project": "fraud-detection",
                "model_name": name,
                "pipeline_stage": "training",
            }
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        print("Training model...")

        model.fit(
            X_train,
            y_train,
        )

        print("Training completed.")

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "algorithm": name,
                "training_samples": len(X_train),
                "feature_count": X_train.shape[1],
            }
        )

        mlflow.set_tag(
            "features",
            ",".join(
                str(column)
                for column in X_train.columns
            ),
        )

        log_model_parameters(model)

        # ----------------------------------------------------
        # Signature
        # ----------------------------------------------------

        predictions = model.predict(X_train)

        signature = infer_signature(
            X_train,
            predictions,
        )

        # ----------------------------------------------------
        # Log Model
        # ----------------------------------------------------

        print("Logging model to MLflow...")

        if isinstance(model, XGBClassifier):
            model_info = mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5),
            )
        else:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5),
            )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = {
            "model_name": name,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "model": model,
        }

        print(f"Run ID: {result['run_id']}")
        print(f"Model URI: {result['model_uri']}")

        return result


# ============================================================
# Train All Models
# ============================================================


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list[dict[str, Any]]:
    models = get_models()
    results = []

    for name, model in models.items():
        result = train_model(
            name=name,
            model=model,
            X_train=X_train,
            y_train=y_train,
        )

        results.append(result)

    return results


# ============================================================
# Main
# ============================================================


def main() -> list[dict[str, Any]]:
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    (
        X_train,
        _,
        y_train,
        _,
    ) = load_data(
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    )

    results = train_all_models(
        X_train,
        y_train,
    )

    print("\nTraining completed.")

    return results


if __name__ == "__main__":
    main()

