"""
API-level tests using FastAPI's TestClient (starlette TestClient under the
hood, driven by httpx). Uses the `lifespan` context so the real startup
logic runs — but points the model holder at a tiny pipeline we train on
the synthetic fixture, the same trick test_predict.py uses, so these
tests don't depend on models/model.pkl already existing.
"""

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocess import split_features_target
from src.features.build_features import build_preprocessor

VALID_PAYLOAD = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


@pytest.fixture
def client(tmp_path, sample_df, monkeypatch):
    X, y = split_features_target(sample_df)
    pipeline = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X, y)

    model_path = tmp_path / "model.pkl"
    joblib.dump(pipeline, model_path)

    # point the API module at our throwaway model before it starts up
    import src.api.main as api_main
    monkeypatch.setattr(api_main, "DEFAULT_MODEL_PATH", model_path)

    with TestClient(api_main.app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "predict" in response.json()


def test_health_endpoint_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_payload_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("disease", "no_disease")
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0


def test_predict_missing_field_returns_422(client):
    bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_invalid_thal_value_returns_422(client):
    bad_payload = {**VALID_PAYLOAD, "thal": 99}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_out_of_range_age_returns_422(client):
    bad_payload = {**VALID_PAYLOAD, "age": -5}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_response_has_request_id_header(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert "X-Request-ID" in response.headers


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.post("/predict", json=VALID_PAYLOAD)  # generate at least one data point
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert "model_predictions_total" in response.text
