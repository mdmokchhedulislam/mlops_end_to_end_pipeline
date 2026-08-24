import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


DATA_PATH = "data/raw/transactions.csv"
MODEL_DIR = "data/processed"


def get_models():

    models = {

        "logistic_regression": LogisticRegression(
            max_iter=1000
        ),

        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),

        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),

        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )
    }

    return models


def train_model(name, model, X_train, X_test, y_train, y_test):

    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    with mlflow.start_run(
        run_name=name
    ):

        # -------------------------
        # Train
        # -------------------------

        model.fit(
            X_train,
            y_train
        )

        # -------------------------
        # Prediction
        # -------------------------

        predictions = model.predict(
            X_test
        )

        # -------------------------
        # Metrics
        # -------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        # -------------------------
        # MLflow Parameters
        # -------------------------

        mlflow.log_param(
            "algorithm",
            name
        )

        # -------------------------
        # MLflow Metrics
        # -------------------------

        mlflow.log_metrics({

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1_score": f1
        })

        # -------------------------
        # Save Model
        # -------------------------

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        model_path = os.path.join(
            MODEL_DIR,
            f"{name}.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        # -------------------------
        # MLflow Model
        # -------------------------

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]
        )
        
        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        print(
            f"Model     : {model_path}"
        )


def main():

    # ==========================
    # Load Dataset
    # ==========================

    df = pd.read_csv(
        DATA_PATH
    )

    X = df.drop(
        "fraud",
        axis=1
    )

    y = df["fraud"]

    # ==========================
    # Train/Test Split
    # ==========================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

    # ==========================
    # MLflow
    # ==========================

    mlflow.set_tracking_uri(
        "http://localhost:5000"
    )

    mlflow.set_experiment(
        "fraud-detection"
    )

    # ==========================
    # Models
    # ==========================

    models = get_models()

    # ==========================
    # Run Experiments
    # ==========================

    for name, model in models.items():

        train_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )


if __name__ == "__main__":
    main()