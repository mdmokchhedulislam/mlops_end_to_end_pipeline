import mlflow
from mlflow import MlflowClient

from src.evaluation.evaluate import evaluate_models
# ==========================================
# Configuration
# ==========================================

MLFLOW_TRACKING_URI = "http://localhost:5000"

MODEL_NAME = "fraud-detection-model"

MIN_F1_SCORE = 0.80
MIN_RECALL = 0.75


# ==========================================
# Register Model
# ==========================================

def register_model():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    # ======================================
    # Get latest experiment results
    # ======================================

    results = evaluate_models()

    best_model = results.iloc[0]

    algorithm = best_model["algorithm"]

    f1_score = best_model["f1_score"]

    recall = best_model["recall"]

    run_id = best_model["run_id"]

    print("\n")
    print("=" * 70)
    print("MODEL REGISTRATION")
    print("=" * 70)

    print(f"\nAlgorithm : {algorithm}")
    print(f"F1 Score  : {f1_score:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Run ID    : {run_id}")

    # ======================================
    # Quality Gate
    # ======================================

    if f1_score < MIN_F1_SCORE:

        raise RuntimeError(
            f"Model rejected: "
            f"F1 Score {f1_score:.4f} "
            f"< {MIN_F1_SCORE}"
        )

    if recall < MIN_RECALL:

        raise RuntimeError(
            f"Model rejected: "
            f"Recall {recall:.4f} "
            f"< {MIN_RECALL}"
        )

    print("\nQuality Gate: PASS")

    # ======================================
    # Model URI
    # ======================================

    model_uri = (
        f"runs:/{run_id}/model"
    )

    print(
        f"\nModel URI: {model_uri}"
    )

    # ======================================
    # Register Model
    # ======================================

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("\n")
    print("=" * 70)
    print("MODEL REGISTERED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Model Name : {registered_model.name}"
    )

    print(
        f"Version    : {registered_model.version}"
    )

    print(
        f"Run ID     : {run_id}"
    )

    print("=" * 70)


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    register_model()