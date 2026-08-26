from typing import Any

import mlflow
from mlflow import MlflowClient

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_ALIAS,
)


# ============================================================
# MLflow Client
# ============================================================

def get_mlflow_client() -> MlflowClient:
    """
    Create and return MLflow client.
    """

    return MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )


# ============================================================
# Find Best Model
# ============================================================

def find_best_model() -> dict[str, Any]:
    """
    Find the best evaluated model from MLflow.

    The best model is selected based on
    eval_f1_score.
    """

    print("\n" + "=" * 70)
    print("                  FINDING BEST MODEL")
    print("=" * 70)

    client = get_mlflow_client()

    # --------------------------------------------------------
    # Get Experiment
    # --------------------------------------------------------

    experiment = (
        client.get_experiment_by_name(
            MLFLOW_EXPERIMENT
        )
    )

    if experiment is None:

        raise RuntimeError(
            f"MLflow experiment not found: "
            f"{MLFLOW_EXPERIMENT}"
        )

    print(
        f"Experiment: {MLFLOW_EXPERIMENT}"
    )

    # --------------------------------------------------------
    # Search Evaluated Runs
    # --------------------------------------------------------

    runs = client.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        filter_string=(
            "attributes.status = 'FINISHED' "
            "AND tags.evaluation_status = 'completed'"
        ),
        order_by=[
            "metrics.eval_f1_score DESC"
        ],
        max_results=1,
    )

    if not runs:

        raise RuntimeError(
            "No successfully evaluated model "
            "found in MLflow."
        )

    best_run = runs[0]

    # --------------------------------------------------------
    # Extract Metrics
    # --------------------------------------------------------

    metrics = best_run.data.metrics
    tags = best_run.data.tags

    required_metrics = [
        "eval_accuracy",
        "eval_precision",
        "eval_recall",
        "eval_f1_score",
    ]

    for metric_name in required_metrics:

        if metric_name not in metrics:

            raise RuntimeError(
                f"Required metric '{metric_name}' "
                f"not found in run "
                f"{best_run.info.run_id}"
            )

    # --------------------------------------------------------
    # Model Name
    # --------------------------------------------------------

    model_name = tags.get(
        "model_name",
        "unknown",
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "run_id": best_run.info.run_id,
        "model_name": model_name,
        "accuracy": metrics[
            "eval_accuracy"
        ],
        "precision": metrics[
            "eval_precision"
        ],
        "recall": metrics[
            "eval_recall"
        ],
        "f1_score": metrics[
            "eval_f1_score"
        ],
        "artifact_uri": (
            f"runs:/{best_run.info.run_id}/model"
        ),
    }

    # --------------------------------------------------------
    # Print Best Model
    # --------------------------------------------------------

    print("\nBest Model:")
    print(
        f"Model     : {result['model_name']}"
    )

    print(
        f"Run ID    : {result['run_id']}"
    )

    print(
        f"Accuracy  : "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{result['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{result['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{result['f1_score']:.4f}"
    )

    print(
        f"Artifact  : "
        f"{result['artifact_uri']}"
    )

    return result


# ============================================================
# Create Registered Model
# ============================================================

def create_registered_model(
    client: MlflowClient,
) -> None:
    """
    Create registered model if it does not exist.
    """

    try:

        client.get_registered_model(
            MLFLOW_MODEL_NAME
        )

        print(
            f"\nRegistered model already exists: "
            f"{MLFLOW_MODEL_NAME}"
        )

    except Exception:

        print(
            f"\nCreating registered model: "
            f"{MLFLOW_MODEL_NAME}"
        )

        client.create_registered_model(
            name=MLFLOW_MODEL_NAME,
            description=(
                "Fraud detection model "
                "managed by automated "
                "MLOps pipeline."
            ),
        )

        print(
            "Registered model created ✓"
        )


# ============================================================
# Register Best Model
# ============================================================

def register_best_model(
    best_result: dict[str, Any],
) -> str:
    """
    Register the best evaluated model
    as a new MLflow model version.
    """

    print("\n" + "=" * 70)
    print("                  MODEL REGISTRATION")
    print("=" * 70)

    client = get_mlflow_client()

    # --------------------------------------------------------
    # Create Registered Model
    # --------------------------------------------------------

    create_registered_model(
        client
    )

    # --------------------------------------------------------
    # Create Model Version
    # --------------------------------------------------------

    print(
        "\nCreating model version..."
    )

    model_version = client.create_model_version(
        name=MLFLOW_MODEL_NAME,
        source=best_result["artifact_uri"],
        run_id=best_result["run_id"],
        description=(
            "Production candidate selected "
            "by automated MLOps pipeline. "
            f"Algorithm={best_result['model_name']}, "
            f"F1={best_result['f1_score']:.4f}"
        ),
    )

    version = str(
        model_version.version
    )

    print(
        f"Registered model version: {version}"
    )

    # --------------------------------------------------------
    # Add Model Version Tags
    # --------------------------------------------------------

    tags = {
        "algorithm": (
            best_result["model_name"]
        ),
        "accuracy": (
            f"{best_result['accuracy']:.6f}"
        ),
        "precision": (
            f"{best_result['precision']:.6f}"
        ),
        "recall": (
            f"{best_result['recall']:.6f}"
        ),
        "f1_score": (
            f"{best_result['f1_score']:.6f}"
        ),
        "selection_metric": "f1_score",
        "selection_reason": (
            "Best evaluated model"
        ),
    }

    for key, value in tags.items():

        client.set_model_version_tag(
            name=MLFLOW_MODEL_NAME,
            version=version,
            key=key,
            value=value,
        )

    print(
        "Model version tags added ✓"
    )

    # --------------------------------------------------------
    # Set Champion Alias
    # --------------------------------------------------------

    print(
        f"\nAssigning alias "
        f"'@{MLFLOW_MODEL_ALIAS}'..."
    )

    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias=MLFLOW_MODEL_ALIAS,
        version=version,
    )

    print(
        f"Model version {version} is now "
        f"@{MLFLOW_MODEL_ALIAS} ✓"
    )

    # --------------------------------------------------------
    # Production URI
    # --------------------------------------------------------

    production_uri = (
        f"models:/{MLFLOW_MODEL_NAME}"
        f"@{MLFLOW_MODEL_ALIAS}"
    )

    print(
        "\nProduction Model URI:"
    )

    print(
        production_uri
    )

    return version


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("\n")
    print("=" * 70)
    print("        FRAUD DETECTION MODEL REGISTRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    print(
        f"\nMLflow URI      : "
        f"{MLFLOW_TRACKING_URI}"
    )

    print(
        f"Experiment      : "
        f"{MLFLOW_EXPERIMENT}"
    )

    print(
        f"Registered Model: "
        f"{MLFLOW_MODEL_NAME}"
    )

    print(
        f"Model Alias     : "
        f"@{MLFLOW_MODEL_ALIAS}"
    )

    # --------------------------------------------------------
    # Find Best Model
    # --------------------------------------------------------

    best_result = find_best_model()

    # --------------------------------------------------------
    # Register Best Model
    # --------------------------------------------------------

    version = register_best_model(
        best_result
    )

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("        MODEL REGISTRATION COMPLETED ✓")
    print("=" * 70)

    print(
        f"\nModel      : {MLFLOW_MODEL_NAME}"
    )

    print(
        f"Version    : {version}"
    )

    print(
        f"Alias      : @{MLFLOW_MODEL_ALIAS}"
    )

    print(
        f"Production : "
        f"models:/{MLFLOW_MODEL_NAME}"
        f"@{MLFLOW_MODEL_ALIAS}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("        MODEL REGISTRATION FAILED ✗")
        print("=" * 70)

        print(
            f"\nError: {error}"
        )

        raise