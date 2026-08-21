import mlflow
import pandas as pd


# ==========================================
# MLflow Configuration
# ==========================================

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "fraud-detection"


# ==========================================
# Get Latest Run for Each Algorithm
# ==========================================

def get_latest_runs():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise ValueError(
            f"Experiment '{EXPERIMENT_NAME}' not found"
        )

    # Get all runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],

        order_by=[
            "start_time DESC"
        ]
    )

    if runs.empty:
        raise ValueError(
            "No MLflow runs found."
        )

    # ------------------------------------------
    # Keep latest run of each algorithm
    # ------------------------------------------

    latest_runs = (
        runs
        .sort_values("start_time", ascending=False)
        .drop_duplicates(
            subset=["params.algorithm"],
            keep="first"
        )
    )

    return latest_runs


# ==========================================
# Evaluate Models
# ==========================================

def evaluate_models():

    runs = get_latest_runs()

    results = []

    for _, run in runs.iterrows():

        algorithm = run.get(
            "params.algorithm"
        )

        accuracy = run.get(
            "metrics.accuracy"
        )

        precision = run.get(
            "metrics.precision"
        )

        recall = run.get(
            "metrics.recall"
        )

        f1_score = run.get(
            "metrics.f1_score"
        )

        run_id = run["run_id"]

        results.append({

            "algorithm": algorithm,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1_score": f1_score,

            "run_id": run_id
        })

    results_df = pd.DataFrame(
        results
    )

    # ==========================================
    # Sort by F1 Score
    # ==========================================

    results_df = results_df.sort_values(
        by="f1_score",
        ascending=False
    )

    return results_df


# ==========================================
# Main
# ==========================================

def main():

    print("\n")
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    results = evaluate_models()

    print("\nModel Comparison:\n")

    print(
        results.to_string(
            index=False
        )
    )

    # ==========================================
    # Best Model
    # ==========================================

    best_model = results.iloc[0]

    print("\n")
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Algorithm : {best_model['algorithm']}"
    )

    print(
        f"Accuracy  : {best_model['accuracy']:.4f}"
    )

    print(
        f"Precision : {best_model['precision']:.4f}"
    )

    print(
        f"Recall    : {best_model['recall']:.4f}"
    )

    print(
        f"F1 Score  : {best_model['f1_score']:.4f}"
    )

    print(
        f"Run ID    : {best_model['run_id']}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()