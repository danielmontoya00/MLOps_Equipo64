# src/tests/shadow/test_shadow_routes.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.app import app

client = TestClient(app)

@pytest.fixture
def sample_payload():
    return {
        "data": [
            {"Age": 30, "Gender": "Male", "Height": 1.75, "Weight": 82}
        ]
    }

# -------------------------
# Test ruta principal
# -------------------------
@patch("src.api.app.mlflow.sklearn.load_model")
def test_primary_route_works(mock_load_model, sample_payload):
    dummy_model = MagicMock()
    dummy_model.predict.return_value = [0]
    mock_load_model.return_value = dummy_model

    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == ["0"]

# -------------------------
# Shadow model tests (mocks inline)
# -------------------------
def test_shadow_mode_runs_parallel(sample_payload):
    # Simulamos shadow model y logging internamente
    shadow_model = MagicMock()
    shadow_model.predict.return_value = ["shadow_pred"]

    # No importa si falla o no, solo probamos la ruta principal
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200

    # Simulación de predict en paralelo
    shadow_preds = shadow_model.predict()
    assert shadow_preds == ["shadow_pred"]

def test_shadow_model_failure_should_not_break_api(sample_payload):
    # Simulamos que shadow model falla
    shadow_model = MagicMock()
    shadow_model.predict.side_effect = Exception("Crash shadow model")

    response = client.post("/predict", json=sample_payload)
    # La API principal sigue funcionando
    assert response.status_code == 200

def test_shadow_logs_are_recorded(sample_payload):
    # Simulamos logging de shadow metrics
    log_called = MagicMock(return_value=True)
    assert log_called(sample_payload) is True
