import os
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from mlflow.models import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# Configuration
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
# MLflow Configuration
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
# Quality Gate
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
# Calculate Metrics
# ============================================================

def calculate_metrics(
    y_test: pd.Series,
    predictions,
) -> dict[str, float]:

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


# ============================================================
# Log Model Parameters
# ============================================================

def log_model_parameters(
    model: Any,
) -> None:

    model_params = model.get_params()

    for param_name, param_value in model_params.items():

        try:

            if param_value is None:
                continue

            mlflow.log_param(
                param_name,
                str(param_value),
            )

        except Exception:
            # Do not fail training because of
            # an unsupported MLflow parameter.
            continue


# ============================================================
# Train Single Model
# ============================================================

def train_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:

    print("\n" + "=" * 70)
    print(f"Training Model: {name}")
    print("=" * 70)

    with mlflow.start_run(
        run_name=name,
    ) as run:

        # ----------------------------------------------------
        # Basic metadata
        # ----------------------------------------------------

        mlflow.set_tags(
            {
                "project": "fraud-detection",
                "model_name": name,
                "stage": "training",
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
        # Prediction
        # ----------------------------------------------------

        predictions = model.predict(
            X_test,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        # ----------------------------------------------------
        # MLflow Parameters
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "algorithm": name,
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "feature_count": X_train.shape[1],
            }
        )

        # Feature names are logged as a tag instead of
        # parameter because parameter values should remain
        # simple and bounded.

        mlflow.set_tag(
            "features",
            ",".join(
                str(column)
                for column in X_train.columns
            ),
        )

        # ----------------------------------------------------
        # Model Parameters
        # ----------------------------------------------------

        log_model_parameters(
            model,
        )

        # ----------------------------------------------------
        # MLflow Metrics
        # ----------------------------------------------------

        mlflow.log_metrics(
            metrics,
        )

        # ----------------------------------------------------
        # Model Signature
        # ----------------------------------------------------

        signature = infer_signature(
            X_train,
            model.predict(X_train),
        )

        # ----------------------------------------------------
        # Log Model to MLflow
        # ----------------------------------------------------

        print("Logging model to MLflow...")

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
        )

        # ----------------------------------------------------
        # Artifact URI
        # ----------------------------------------------------

        artifact_uri = model_info.model_uri

        # ----------------------------------------------------
        # Print Results
        # ----------------------------------------------------

        print("\nModel Results:")

        print(
            f"Accuracy  : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision : {metrics['precision']:.4f}"
        )

        print(
            f"Recall    : {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score  : {metrics['f1_score']:.4f}"
        )

        print(
            f"Run ID    : {run.info.run_id}"
        )

        print(
            f"Model URI : {artifact_uri}"
        )

        # ----------------------------------------------------
        # Return Result
        # ----------------------------------------------------

        return {
            "name": name,
            "model": model,
            "run_id": run.info.run_id,
            "artifact_uri": artifact_uri,
            **metrics,
        }


# ============================================================
# Load Processed Dataset
# ============================================================

def load_processed_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:

    print("\n" + "=" * 70)
    print("Loading processed dataset")
    print("=" * 70)

    required_files = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    X_train = pd.read_csv(
        X_TRAIN_PATH,
    )

    X_test = pd.read_csv(
        X_TEST_PATH,
    )

    y_train = pd.read_csv(
        Y_TRAIN_PATH,
    ).squeeze()

    y_test = pd.read_csv(
        Y_TEST_PATH,
    ).squeeze()

    print(
        f"X_train shape: {X_train.shape}"
    )

    print(
        f"X_test shape : {X_test.shape}"
    )

    print(
        f"y_train shape: {y_train.shape}"
    )

    print(
        f"y_test shape : {y_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# Quality Gate
# ============================================================

def quality_gate(
    result: dict[str, Any],
) -> bool:

    print("\n" + "=" * 70)
    print("                    QUALITY GATE")
    print("=" * 70)

    checks = {
        "accuracy": (
            result["accuracy"],
            MIN_ACCURACY,
        ),
        "precision": (
            result["precision"],
            MIN_PRECISION,
        ),
        "recall": (
            result["recall"],
            MIN_RECALL,
        ),
        "f1_score": (
            result["f1_score"],
            MIN_F1_SCORE,
        ),
    }

    passed = True

    for metric_name, (
        actual,
        minimum,
    ) in checks.items():

        status = "PASS" if actual >= minimum else "FAIL"

        print(
            f"{metric_name:<12} "
            f"actual={actual:.4f} "
            f"required={minimum:.4f} "
            f"[{status}]"
        )

        if actual < minimum:
            passed = False

    if passed:

        print(
            "\nQUALITY GATE: PASSED ✓"
        )

    else:

        print(
            "\nQUALITY GATE: FAILED ✗"
        )

    return passed


# ============================================================
# Register Best Model
# ============================================================

def register_best_model(
    best_result: dict[str, Any],
) -> None:

    print("\n" + "=" * 70)
    print("               MODEL REGISTRATION")
    print("=" * 70)

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI,
    )

    # --------------------------------------------------------
    # Create Registered Model if it does not exist
    # --------------------------------------------------------

    try:

        client.get_registered_model(
            MLFLOW_MODEL_NAME,
        )

        print(
            f"Registered model already exists: "
            f"{MLFLOW_MODEL_NAME}"
        )

    except Exception:

        print(
            f"Creating registered model: "
            f"{MLFLOW_MODEL_NAME}"
        )

        client.create_registered_model(
            name=MLFLOW_MODEL_NAME,
            description=(
                "Fraud detection model "
                "managed by automated MLOps pipeline."
            ),
        )

    # --------------------------------------------------------
    # Create Model Version
    # --------------------------------------------------------

    print(
        "Creating model version..."
    )

    model_version = client.create_model_version(
        name=MLFLOW_MODEL_NAME,
        source=best_result["artifact_uri"],
        run_id=best_result["run_id"],
        description=(
            f"Automatically registered best model. "
            f"Algorithm={best_result['name']}, "
            f"F1={best_result['f1_score']:.4f}"
        ),
    )

    version = model_version.version

    print(
        f"Registered model version: {version}"
    )

    # --------------------------------------------------------
    # Model Version Tags
    # --------------------------------------------------------

    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=version,
        key="algorithm",
        value=best_result["name"],
    )

    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=version,
        key="accuracy",
        value=f"{best_result['accuracy']:.6f}",
    )

    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=version,
        key="precision",
        value=f"{best_result['precision']:.6f}",
    )

    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=version,
        key="recall",
        value=f"{best_result['recall']:.6f}",
    )

    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=version,
        key="f1_score",
        value=f"{best_result['f1_score']:.6f}",
    )

    # --------------------------------------------------------
    # Set Champion Alias
    # --------------------------------------------------------

    print(
        f"Assigning alias '@{MLFLOW_MODEL_ALIAS}'..."
    )

    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias=MLFLOW_MODEL_ALIAS,
        version=version,
    )

    print(
        f"\nModel version {version} is now "
        f"@{MLFLOW_MODEL_ALIAS} ✓"
    )

    print(
        "\nProduction model URI:"
    )

    print(
        f"models:/{MLFLOW_MODEL_NAME}@"
        f"{MLFLOW_MODEL_ALIAS}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("\n")
    print("=" * 70)
    print("        FRAUD DETECTION MODEL TRAINING")
    print("=" * 70)

    # ========================================================
    # MLflow Configuration
    # ========================================================

    print("\n" + "=" * 70)
    print("Configuring MLflow")
    print("=" * 70)

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI,
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT,
    )

    print(
        f"MLflow URI      : {MLFLOW_TRACKING_URI}"
    )

    print(
        f"Experiment      : {MLFLOW_EXPERIMENT}"
    )

    print(
        f"Registered Model: {MLFLOW_MODEL_NAME}"
    )

    print(
        f"Model Alias     : @{MLFLOW_MODEL_ALIAS}"
    )

    # ========================================================
    # Load Data
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_processed_data()

    # ========================================================
    # Get Models
    # ========================================================

    models = get_models()

    print("\nModels to train:")

    for model_name in models:

        print(
            f"  - {model_name}"
        )

    # ========================================================
    # Train Models
    # ========================================================

    results: list[dict[str, Any]] = []

    for name, model in models.items():

        result = train_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(
            result
        )

    # ========================================================
    # Model Comparison
    # ========================================================

    print("\n")
    print("=" * 70)
    print("                MODEL COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(
        [
            {
                "model": result["name"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
            }
            for result in results
        ]
    )

    results_df = results_df.sort_values(
        by="f1_score",
        ascending=False,
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Find Best Model
    # ========================================================

    best_result = max(
        results,
        key=lambda result: result["f1_score"],
    )

    print("\n")
    print("=" * 70)
    print("                 BEST MODEL")
    print("=" * 70)

    print(
        f"Model     : {best_result['name']}"
    )

    print(
        f"Accuracy  : {best_result['accuracy']:.4f}"
    )

    print(
        f"Precision : {best_result['precision']:.4f}"
    )

    print(
        f"Recall    : {best_result['recall']:.4f}"
    )

    print(
        f"F1 Score  : {best_result['f1_score']:.4f}"
    )

    print(
        f"Run ID    : {best_result['run_id']}"
    )

    print(
        f"Model URI : {best_result['artifact_uri']}"
    )

    # ========================================================
    # Quality Gate
    # ========================================================

    if not quality_gate(
        best_result,
    ):

        print("\n")
        print("=" * 70)
        print("        MODEL TRAINING REJECTED ✗")
        print("=" * 70)

        print(
            "\nBest model did not pass "
            "the production quality gate."
        )

        # IMPORTANT:
        # Do NOT register or promote rejected model.

        raise RuntimeError(
            "Model failed production quality gate."
        )

    # ========================================================
    # Register Best Model
    # ========================================================

    register_best_model(
        best_result,
    )

    # ========================================================
    # Completed
    # ========================================================

    print("\n")
    print("=" * 70)
    print("        MODEL TRAINING COMPLETED ✓")
    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("        MODEL TRAINING FAILED ✗")
        print("=" * 70)

        print(
            f"\nError: {error}"
        )

        raise