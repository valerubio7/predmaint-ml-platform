"""Tests for src/api/main.py — FastAPI inference endpoints.

Uses TestClient (no network I/O) and monkeypatches the model loading so that
tests run without a trained model.pkl present on disk.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_model(prediction: int = 0, probability: float = 0.12) -> MagicMock:
    """Return a mock sklearn Pipeline that mimics predict / predict_proba."""
    mock = MagicMock()
    mock.predict.return_value = np.array([prediction])
    mock.predict_proba.return_value = np.array([[1 - probability, probability]])
    return mock


VALID_PAYLOAD = {
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0,
    "type_h": 0,
    "type_l": 1,
    "type_m": 0,
}


@pytest.fixture
def client_no_model(tmp_path, monkeypatch):
    """TestClient pointing to a path where no model.pkl exists."""
    import api.main as main_module

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nonexistent.pkl"))
    # Reset lazy-loaded model so the new path is respected
    main_module._model = None
    main_module.MODEL_PATH = tmp_path / "nonexistent.pkl"

    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client_with_model(monkeypatch):
    """TestClient with a mocked loaded model (no file I/O)."""
    import api.main as main_module

    mock_model = _make_mock_model(prediction=0, probability=0.05)
    main_module._model = mock_model  # inject directly, bypassing joblib.load

    from api.main import app

    yield TestClient(app)

    # Teardown: reset global state
    main_module._model = None


@pytest.fixture
def client_with_failure_model(monkeypatch):
    """TestClient whose model always predicts failure (1)."""
    import api.main as main_module

    mock_model = _make_mock_model(prediction=1, probability=0.93)
    main_module._model = mock_model

    from api.main import app

    yield TestClient(app)

    main_module._model = None


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_returns_200():
    """GET /health must return HTTP 200."""
    from api.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_body():
    """GET /health response body must be {'status': 'ok'}."""
    from api.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


def test_health_content_type_is_json():
    """GET /health must respond with application/json content-type."""
    from api.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /predict — model not loaded (503)
# ---------------------------------------------------------------------------


def test_predict_returns_503_when_model_missing(client_no_model):
    """POST /predict must return 503 when model file is not present."""
    response = client_no_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_predict_503_detail_mentions_model(client_no_model):
    """503 response detail must mention the model path or training instruction."""
    response = client_no_model.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    assert "detail" in body
    assert len(body["detail"]) > 0


# ---------------------------------------------------------------------------
# POST /predict — happy path (no failure predicted)
# ---------------------------------------------------------------------------


def test_predict_returns_200_with_valid_payload(client_with_model):
    """POST /predict with valid payload must return HTTP 200."""
    response = client_with_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_has_required_fields(client_with_model):
    """Response must include 'failure_predicted' and 'failure_probability'."""
    response = client_with_model.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    assert "failure_predicted" in body
    assert "failure_probability" in body


def test_predict_failure_predicted_is_bool(client_with_model):
    """'failure_predicted' field must be a boolean."""
    response = client_with_model.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    assert isinstance(body["failure_predicted"], bool)


def test_predict_failure_probability_is_float(client_with_model):
    """'failure_probability' must be a float."""
    response = client_with_model.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    assert isinstance(body["failure_probability"], float)


def test_predict_probability_in_valid_range(client_with_model):
    """'failure_probability' must be between 0.0 and 1.0 (inclusive)."""
    response = client_with_model.post("/predict", json=VALID_PAYLOAD)
    prob = response.json()["failure_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_no_failure_when_model_predicts_zero(client_with_model):
    """failure_predicted must be False when model predicts class 0."""
    response = client_with_model.post("/predict", json=VALID_PAYLOAD)
    assert response.json()["failure_predicted"] is False


# ---------------------------------------------------------------------------
# POST /predict — failure predicted (class 1)
# ---------------------------------------------------------------------------


def test_predict_failure_when_model_predicts_one(client_with_failure_model):
    """failure_predicted must be True when model predicts class 1."""
    response = client_with_failure_model.post("/predict", json=VALID_PAYLOAD)
    assert response.json()["failure_predicted"] is True


def test_predict_high_probability_on_failure(client_with_failure_model):
    """Failure probability must be >= 0.5 when model predicts class 1."""
    response = client_with_failure_model.post("/predict", json=VALID_PAYLOAD)
    assert response.json()["failure_probability"] >= 0.5


# ---------------------------------------------------------------------------
# POST /predict — invalid payloads (422 Unprocessable Entity)
# ---------------------------------------------------------------------------


def test_predict_422_missing_required_field(client_with_model):
    """Omitting a required field must return 422."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "torque"}
    response = client_with_model.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_422_wrong_type_for_field(client_with_model):
    """Passing a string for a numeric field must return 422."""
    payload = {**VALID_PAYLOAD, "air_temperature": "not_a_number"}
    response = client_with_model.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_422_empty_body(client_with_model):
    """Sending an empty body must return 422."""
    response = client_with_model.post("/predict", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict — model raises an unexpected exception (500)
# ---------------------------------------------------------------------------


def test_predict_returns_500_on_unexpected_model_error():
    """POST /predict must return 500 when the model raises an unexpected error."""
    import api.main as main_module

    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("Unexpected GPU OOM")
    main_module._model = mock_model

    from api.main import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 500

    main_module._model = None


def test_predict_500_detail_contains_error_message():
    """500 response detail must propagate the underlying exception message."""
    import api.main as main_module

    mock_model = MagicMock()
    mock_model.predict.side_effect = ValueError("bad input shape")
    main_module._model = mock_model

    from api.main import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert "bad input shape" in response.json()["detail"]

    main_module._model = None
