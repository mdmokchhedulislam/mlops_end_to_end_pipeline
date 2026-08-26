import os
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd

from mlflow.models import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from xgboost import XGBClassifier


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
):

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

    X_train = pd.read_csv(
        x_train_path
    )

    X_test = pd.read_csv(
        x_test_path
    )

    y_train = pd.read_csv(
        y_train_path
    ).squeeze()

    y_test = pd.read_csv(
        y_test_path
    ).squeeze()

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"y_test : {y_test.shape}"
    )

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

        except Exception:

            continue


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

    with mlflow.start_run(
        run_name=name
    ) as run:

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

        print(
            "Training model..."
        )

        model.fit(
            X_train,
            y_train,
        )

        print(
            "Training completed."
        )

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

        log_model_parameters(
            model
        )

        # ----------------------------------------------------
        # Signature
        # ----------------------------------------------------

        predictions = model.predict(
            X_train
        )

        signature = infer_signature(
            X_train,
            predictions,
        )

        # ----------------------------------------------------
        # Log Model
        # ----------------------------------------------------

        print(
            "Logging model to MLflow..."
        )

        if isinstance(
            model,
            XGBClassifier,
        ):

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

        print(
            f"Run ID: {result['run_id']}"
        )

        print(
            f"Model URI: {result['model_uri']}"
        )

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

        results.append(
            result
        )

    return results


# ============================================================
# Main
# ============================================================

def main():

    from src.config import (
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
        MLFLOW_TRACKING_URI,
        MLFLOW_EXPERIMENT,
    )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
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