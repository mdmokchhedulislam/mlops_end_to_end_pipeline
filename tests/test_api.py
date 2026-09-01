from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "fraud-detection-api"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_predict_validation_error():
    # Missing required fields should fail schema validation, not 500.
    response = client.post("/predict", json={"amount": -5})
    assert response.status_code == 422


















# import mlflow
# import pandas as pd

# from unittest.mock import MagicMock

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# from fraud_detector.models.train import get_models, train_model


# def test_get_models():

#     models = get_models()

#     expected_models = {
#         "logistic_regression",
#         "decision_tree",
#         "random_forest",
#         "gradient_boosting",
#         "xgboost",
#     }

#     assert set(models.keys()) == expected_models
#     assert len(models) == 5


# def test_train_model(tmp_path, monkeypatch):

#     # -------------------------
#     # Create test dataset
#     # -------------------------

#     X, y = make_classification(
#         n_samples=100,
#         n_features=4,
#         n_informative=3,
#         n_redundant=0,
#         random_state=42,
#     )

#     X = pd.DataFrame(
#         X,
#         columns=[
#             "feature_1",
#             "feature_2",
#             "feature_3",
#             "feature_4",
#         ],
#     )

#     y = pd.Series(y)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y,
#     )

#     # -------------------------
#     # Use temporary model directory
#     # -------------------------

#     model_dir = tmp_path / "models"

#     model_dir.mkdir()

#     monkeypatch.setattr(
#         "src.training.train.MODEL_DIR",
#         str(model_dir),
#     )

#     # -------------------------
#     # Mock MLflow
#     # -------------------------

#     mock_run = MagicMock()

#     mock_run.__enter__.return_value = mock_run
#     mock_run.__exit__.return_value = None

#     # Mock MLflow start_run
#     monkeypatch.setattr(
#         mlflow,
#         "start_run",
#         lambda *args, **kwargs: mock_run,
#     )

#     # Mock MLflow parameter logging
#     monkeypatch.setattr(
#         mlflow,
#         "log_param",
#         lambda *args, **kwargs: None,
#     )

#     # Mock MLflow metric logging
#     monkeypatch.setattr(
#         mlflow,
#         "log_metrics",
#         lambda *args, **kwargs: None,
#     )

#     # Mock MLflow model logging
#     monkeypatch.setattr(
#         mlflow.sklearn,
#         "log_model",
#         lambda *args, **kwargs: None,
#     )

#     # -------------------------
#     # Create model
#     # -------------------------

#     model = LogisticRegression(
#         max_iter=1000
#     )

#     # -------------------------
#     # Train model
#     # -------------------------

#     train_model(
#         "test_logistic_regression",
#         model,
#         X_train,
#         X_test,
#         y_train,
#         y_test,
#     )

#     # -------------------------
#     # Check model file
#     # -------------------------

#     model_path = (
#         model_dir / "test_logistic_regression.pkl"
#     )

#     assert model_path.exists()

#     assert model_path.stat().st_size > 0