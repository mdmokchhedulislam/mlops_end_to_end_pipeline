import sys

from evaluate import evaluate_models


# ==========================================
# Quality Gate Configuration
# ==========================================

MIN_F1_SCORE = 0.80
MIN_RECALL = 0.75


# ==========================================
# Quality Gate
# ==========================================

def quality_gate():

    print("\n")
    print("=" * 70)
    print("MODEL QUALITY GATE")
    print("=" * 70)

    # Get evaluated models
    results = evaluate_models()

    # Best model
    best_model = results.iloc[0]

    algorithm = best_model["algorithm"]
    f1_score = best_model["f1_score"]
    recall = best_model["recall"]
    run_id = best_model["run_id"]

    print(f"\nModel      : {algorithm}")
    print(f"F1 Score   : {f1_score:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"Run ID     : {run_id}")

    print("\nQuality Requirements:")
    print(f"Minimum F1     : {MIN_F1_SCORE}")
    print(f"Minimum Recall : {MIN_RECALL}")

    # ==========================================
    # Check Quality Gate
    # ==========================================

    f1_pass = f1_score >= MIN_F1_SCORE

    recall_pass = recall >= MIN_RECALL

    print("\nGate Results:")

    print(
        f"F1 Score   : {'PASS' if f1_pass else 'FAIL'}"
    )

    print(
        f"Recall     : {'PASS' if recall_pass else 'FAIL'}"
    )

    # ==========================================
    # Final Decision
    # ==========================================

    if f1_pass and recall_pass:

        print("\n" + "=" * 70)
        print("QUALITY GATE: PASS")
        print("=" * 70)

        print(
            f"\nModel '{algorithm}' is approved "
            "for model registration."
        )

        return True

    else:

        print("\n" + "=" * 70)
        print("QUALITY GATE: FAIL")
        print("=" * 70)

        print(
            f"\nModel '{algorithm}' "
            "does not meet production requirements."
        )

        return False


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    passed = quality_gate()

    if not passed:

        sys.exit(1)