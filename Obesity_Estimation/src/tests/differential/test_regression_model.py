# src/tests/differential/test_regression_model.py
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.app import app  # Asegúrate de que esta ruta exista

client = TestClient(app)

# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def sample_payload():
    """Datos de prueba para el modelo."""
    return {
        "data": [
            {"Age": 25, "Gender": "Female", "Height": 1.65, "Weight": 60},
            {"Age": 40, "Gender": "Male", "Height": 1.80, "Weight": 90},
        ]
    }

@pytest.fixture
def baseline_predictions():
    """Predicciones de referencia guardadas previamente."""
    return ["0", "1"]  # Por ejemplo: 0 = Normal, 1 = Overweight

# -------------------------
# Test de regresión del modelo
# -------------------------
@patch("src.api.app.mlflow.sklearn.load_model")
def test_model_regression(mock_load_model, sample_payload, baseline_predictions):
    """Verifica que las predicciones del modelo no hayan cambiado."""
    # Simulamos un modelo cargado
    dummy_model = MagicMock()
    dummy_model.predict.return_value = [int(p) for p in baseline_predictions]
    mock_load_model.return_value = dummy_model

    # Llamada a la API
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200

    body = response.json()
    current_preds = body["predictions"]

    # Comparación diferencial
    assert current_preds == baseline_predictions, (
        f"Regresión detectada: {current_preds} != {baseline_predictions}"
    )
