from typing import Any

import mlflow
from mlflow import MlflowClient

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MIN_ACCURACY,
    MIN_PRECISION,
    MIN_RECALL,
    MIN_F1_SCORE,
)


# ============================================================
# Constants
# ============================================================

REQUIRED_METRICS = {
    "eval_accuracy",
    "eval_precision",
    "eval_recall",
    "eval_f1_score",
}


# ============================================================
# Quality Gate
# ============================================================

def quality_gate(
    result: dict[str, Any],
) -> bool:
    """
    Validate the best evaluated model against
    production quality thresholds.
    """

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

        status = (
            "PASS"
            if actual >= minimum
            else "FAIL"
        )

        print(
            f"{metric_name:<12} "
            f"actual={actual:.4f} "
            f"required={minimum:.4f} "
            f"[{status}]"
        )

        if actual < minimum:
            passed = False

    print("\n" + "-" * 70)

    if passed:
        print("QUALITY GATE: PASSED ✓")
    else:
        print("QUALITY GATE: FAILED ✗")

    print("-" * 70)

    return passed


# ============================================================
# Get Best Evaluated Run
# ============================================================

def get_best_evaluated_run() -> dict[str, Any]:
    """
    Find the best completed evaluated model.

    Selection criteria:
        1. Run must be FINISHED
        2. evaluation_status must be completed
        3. Required evaluation metrics must exist
        4. Highest F1 score wins
    """

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    # --------------------------------------------------------
    # Get Experiment
    # --------------------------------------------------------

    experiment = client.get_experiment_by_name(
        MLFLOW_EXPERIMENT
    )

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: "
            f"{MLFLOW_EXPERIMENT}"
        )

    print(
        f"\nExperiment ID: "
        f"{experiment.experiment_id}"
    )

    # --------------------------------------------------------
    # Get Evaluated Runs
    # --------------------------------------------------------

    runs = client.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        filter_string=(
            "attributes.status = 'FINISHED' "
            "and tags.evaluation_status = 'completed'"
        ),
        order_by=[
            "metrics.eval_f1_score DESC"
        ],
        max_results=100,
    )

    if not runs:

        raise RuntimeError(
            "No completed evaluated model found "
            "in MLflow. Run the evaluation step first."
        )

    print(
        f"Found {len(runs)} completed "
        f"evaluated runs."
    )

    # --------------------------------------------------------
    # Validate Runs
    # --------------------------------------------------------

    valid_runs = []

    for run in runs:

        run_id = run.info.run_id
        metrics = run.data.metrics

        missing_metrics = (
            REQUIRED_METRICS
            - set(metrics.keys())
        )

        if missing_metrics:

            print(
                f"\nSkipping invalid run: {run_id}"
            )

            print(
                "Missing metrics: "
                f"{sorted(missing_metrics)}"
            )

            continue

        # Make sure F1 is valid
        f1_score = metrics.get(
            "eval_f1_score"
        )

        if f1_score is None:

            print(
                f"\nSkipping run {run_id}: "
                "F1 score is missing."
            )

            continue

        valid_runs.append(run)

    # --------------------------------------------------------
    # No Valid Runs
    # --------------------------------------------------------

    if not valid_runs:

        raise RuntimeError(
            "No valid evaluated runs found. "
            "Required evaluation metrics are missing."
        )

    # --------------------------------------------------------
    # Sort By F1
    # --------------------------------------------------------

    valid_runs.sort(
        key=lambda run: run.data.metrics[
            "eval_f1_score"
        ],
        reverse=True,
    )

    best_run = valid_runs[0]

    metrics = best_run.data.metrics
    tags = best_run.data.tags

    # --------------------------------------------------------
    # Model Information
    # --------------------------------------------------------

    model_name = tags.get(
        "model_name",
        tags.get(
            "evaluated_model",
            "unknown",
        ),
    )

    run_id = best_run.info.run_id

    model_uri = (
        f"runs:/{run_id}/model"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "run_id": run_id,
        "model_name": model_name,
        "model_uri": model_uri,

        "accuracy": float(
            metrics["eval_accuracy"]
        ),

        "precision": float(
            metrics["eval_precision"]
        ),

        "recall": float(
            metrics["eval_recall"]
        ),

        "f1_score": float(
            metrics["eval_f1_score"]
        ),
    }

    return result


# ============================================================
# Update MLflow Quality Gate Tags
# ============================================================

def update_quality_gate_status(
    run_id: str,
    passed: bool,
) -> None:
    """
    Store quality gate result in MLflow.
    """

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    status = (
        "passed"
        if passed
        else "failed"
    )

    client.set_tag(
        run_id,
        "quality_gate",
        status,
    )

    client.set_tag(
        run_id,
        "pipeline_stage",
        "quality_gate",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("\n")

    print("=" * 70)
    print("        FRAUD DETECTION QUALITY GATE")
    print("=" * 70)

    # --------------------------------------------------------
    # Configure MLflow
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
    # Find Best Evaluated Model
    # --------------------------------------------------------

    print(
        "\nFinding best evaluated model..."
    )

    best_result = (
        get_best_evaluated_run()
    )

    # --------------------------------------------------------
    # Display Best Model
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)
    print("              BEST EVALUATED MODEL")
    print("=" * 70)

    print(
        f"Model     : "
        f"{best_result['model_name']}"
    )

    print(
        f"Run ID    : "
        f"{best_result['run_id']}"
    )

    print(
        f"Model URI : "
        f"{best_result['model_uri']}"
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

    # --------------------------------------------------------
    # Quality Gate
    # --------------------------------------------------------

    passed = quality_gate(
        best_result
    )

    # --------------------------------------------------------
    # Update MLflow
    # --------------------------------------------------------

    update_quality_gate_status(
        run_id=best_result["run_id"],
        passed=passed,
    )

    # --------------------------------------------------------
    # Pipeline Decision
    # --------------------------------------------------------

    if not passed:

        print("\n")

        print("=" * 70)
        print("                    MODEL REJECTED ✗")
        print("=" * 70)

        print(
            "\nModel did not meet the "
            "production quality thresholds."
        )

        print(
            "\nPipeline stopped."
        )

        raise RuntimeError(
            "Model failed production quality gate."
        )

    # --------------------------------------------------------
    # Approved
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)
    print("                    MODEL APPROVED ✓")
    print("=" * 70)

    print(
        f"\nApproved Model : "
        f"{best_result['model_name']}"
    )

    print(
        f"Run ID         : "
        f"{best_result['run_id']}"
    )

    print(
        f"Model URI      : "
        f"{best_result['model_uri']}"
    )

    print(
        "\nQuality gate passed. "
        "Model is eligible for the next pipeline stage."
    )

    print("\n")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\n")

        print("=" * 70)
        print("        QUALITY GATE FAILED ✗")
        print("=" * 70)

        print(
            f"\nError: {error}"
        )

        raise