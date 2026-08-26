from typing import Any

import mlflow
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
)


# ============================================================
# Test Dataset Paths
# ============================================================

X_TEST_PATH = f"{PROCESSED_DATA_DIR}/X_test.csv"
Y_TEST_PATH = f"{PROCESSED_DATA_DIR}/y_test.csv"


# ============================================================
# Load Test Data
# ============================================================

def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load test features and labels."""

    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    try:
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Test dataset not found: {error}"
        ) from error

    if X_test.empty:
        raise ValueError("X_test dataset is empty.")

    if y_test.empty:
        raise ValueError("y_test dataset is empty.")

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test have different number of samples."
        )

    print(f"X_test shape : {X_test.shape}")
    print(f"y_test shape : {y_test.shape}")

    return X_test, y_test


# ============================================================
# Calculate Metrics
# ============================================================

def calculate_metrics(
    y_true: pd.Series,
    predictions: Any,
) -> dict[str, float]:
    """Calculate classification metrics."""

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
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
# Get Training Runs
# ============================================================

def get_training_runs() -> list[Any]:
    """
    Get completed training runs from MLflow.

    IMPORTANT:
    Training code uses:
        pipeline_stage=training

    Therefore we must search using the same tag.
    """

    client = mlflow.MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    experiment = client.get_experiment_by_name(
        MLFLOW_EXPERIMENT
    )

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment does not exist: "
            f"{MLFLOW_EXPERIMENT}"
        )

    runs = client.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        filter_string=(
            "attributes.status = 'FINISHED' "
            "and tags.pipeline_stage = 'training'"
        ),
        order_by=[
            "start_time DESC"
        ],
    )

    if not runs:
        raise RuntimeError(
            "No completed training runs found in MLflow. "
            "Run the training step first."
        )

    return runs


# ============================================================
# Evaluate One Model
# ============================================================

def evaluate_model(
    run_id: str,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Load model from MLflow and evaluate it."""

    print("\n" + "=" * 70)
    print(f"EVALUATING MODEL: {model_name}")
    print("=" * 70)

    model_uri = f"runs:/{run_id}/model"

    print(f"Run ID    : {run_id}")
    print(f"Model URI : {model_uri}")

    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------

    try:
        # pyfunc works for both:
        # sklearn models
        # XGBoost models
        model = mlflow.pyfunc.load_model(
            model_uri
        )

    except Exception as error:
        raise RuntimeError(
            f"Failed to load model from MLflow. "
            f"Run ID: {run_id}"
        ) from error

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # Log Metrics Into SAME Training Run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_id=run_id
    ):

        mlflow.log_metrics(
            {
                "eval_accuracy": metrics["accuracy"],
                "eval_precision": metrics["precision"],
                "eval_recall": metrics["recall"],
                "eval_f1_score": metrics["f1_score"],
            }
        )

        # IMPORTANT TAGS

        mlflow.set_tags(
            {
                "evaluation_status": "completed",
                "pipeline_stage": "evaluation",
                "evaluated_model": model_name,
            }
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nEvaluation Results:")

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

    return {
        "name": model_name,
        "run_id": run_id,
        "model_uri": model_uri,
        **metrics,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("\n")
    print("=" * 70)
    print("       FRAUD DETECTION MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    print(
        f"\nMLflow URI : "
        f"{MLFLOW_TRACKING_URI}"
    )

    print(
        f"Experiment : "
        f"{MLFLOW_EXPERIMENT}"
    )

    # --------------------------------------------------------
    # Test Data
    # --------------------------------------------------------

    X_test, y_test = load_test_data()

    # --------------------------------------------------------
    # Training Runs
    # --------------------------------------------------------

    runs = get_training_runs()

    print(
        f"\nFound {len(runs)} training runs."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results: list[dict[str, Any]] = []

    for run in runs:

        model_name = run.data.tags.get(
            "model_name",
            "unknown",
        )

        result = evaluate_model(
            run_id=run.info.run_id,
            model_name=model_name,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(result)

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)

    results_df = pd.DataFrame(
        [
            {
                "model": result["name"],
                "run_id": result["run_id"],
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

    # --------------------------------------------------------
    # Best Model
    # --------------------------------------------------------

    best_result = max(
        results,
        key=lambda result: result["f1_score"],
    )

    print("\n")
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Model     : "
        f"{best_result['name']}"
    )

    print(
        f"Run ID    : "
        f"{best_result['run_id']}"
    )

    print(
        f"Accuracy  : "
        f"{best_result['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{best_result['f1_score']:.4f}"
    )

    print("\n")
    print("=" * 70)
    print("        EVALUATION COMPLETED ✓")
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
        print("        EVALUATION FAILED ✗")
        print("=" * 70)

        print(
            f"\nError: {error}"
        )

        raise