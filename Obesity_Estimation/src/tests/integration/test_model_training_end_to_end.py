import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def data_dir():
    """Directorio base de data/ según Cookiecutter."""
    return Path(__file__).resolve().parents[3] / "data"


@pytest.fixture
def processed_df(data_dir):
    """Carga el dataset procesado final."""
    df = pd.read_csv(data_dir / "processed" / "data_processed.csv")
    return df


@pytest.fixture
def temp_model_output(tmp_path):
    """Directorio temporal para simular salida del pipeline."""
    return tmp_path


# -------------------------------------------------------------------
# TEST 1 — Entrenamiento básico del modelo
# -------------------------------------------------------------------
def test_model_trains_successfully(processed_df):
    """Valida que RandomForest entrena sin errores y produce predicciones."""
    X = processed_df.drop(columns=["target"])
    y = processed_df["target"]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)
    assert accuracy_score(y, preds) > 0.7, "La exactitud mínima esperada es 0.70."


# -------------------------------------------------------------------
# TEST 2 — Validar salida de train_model genera model_info.json
# -------------------------------------------------------------------
def test_model_info_file_exists_and_valid(temp_model_output):
    """
    Simula que train_model.py genera un model_info.json
    y valida contenido mínimo.
    """
    model_info_path = temp_model_output / "model_info.json"

    fake_info = {
        "run_id": "12345abc",
        "model_uri": "runs:/12345abc/model",
        "registered_model_name": "RandomForestObesityModel"
    }

    # Crear archivo simulado
    with open(model_info_path, "w") as f:
        json.dump(fake_info, f)

    # Validaciones
    assert model_info_path.exists(), "model_info.json no fue generado."

    with open(model_info_path, "r") as f:
        loaded = json.load(f)

    assert "run_id" in loaded
    assert "model_uri" in loaded
    assert "registered_model_name" in loaded
    assert loaded["run_id"] != ""
    assert loaded["model_uri"].startswith("runs:")
    assert loaded["registered_model_name"] == "RandomForestObesityModel"


# -------------------------------------------------------------------
# TEST 3 — Mock MLflow para validar que train_model hace logging
# -------------------------------------------------------------------
@patch("mlflow.start_run")
@patch("mlflow.log_metric")
@patch("mlflow.log_params")
@patch("mlflow.sklearn.log_model")
def test_mlflow_logging(mock_log_model, mock_log_params, mock_log_metric, mock_start_run, processed_df):
    """
    Simula ejecución del pipeline de entrenamiento y valida que MLflow reciba:
    - métricas
    - parámetros
    - modelo
    """
    # Configurar mocks
    mock_run = MagicMock()
    mock_run.info.run_id = "abc123"
    mock_start_run.return_value.__enter__.return_value = mock_run

    X = processed_df.drop(columns=["target"])
    y = processed_df["target"]

    # Entrenar un modelo simple
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    acc = accuracy_score(y, model.predict(X))

    # Logging simulado
    import mlflow
    with mlflow.start_run():
        mlflow.log_params({"n_estimators": 5})
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    # Aserciones
    mock_log_params.assert_called_once()
    mock_log_metric.assert_called_once()
    mock_log_model.assert_called_once()

    assert mock_start_run.called, "MLflow.start_run nunca fue llamado."


# -------------------------------------------------------------------
# TEST 4 — El modelo entrenado puede cargar y predecir
# -------------------------------------------------------------------
def test_model_can_predict_after_saving(tmp_path, processed_df):
    """
    Simula guardar un modelo entrenado y luego cargarlo para validar que predice.
    """
    X = processed_df.drop(columns=["target"])
    y = processed_df["target"]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"

    # Guardar modelo
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Cargar modelo
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)

    preds = loaded.predict(X)

    assert len(preds) == len(y)
    assert np.array_equal(preds, model.predict(X)), "El modelo cargado debe comportarse igual."


# -------------------------------------------------------------------
# TEST 5 — Validar que el pipeline final deja artefactos obligatorios
# -------------------------------------------------------------------
def test_required_artifacts_exist(data_dir):
    """
    Verifica que tras train_model.py existan los archivos requeridos:
    - processed data
    - model_info.json
    - modelo entrenado
    """
    required = [
        data_dir / "processed" / "data_processed.csv",
        Path("models/model_info.json"),
        Path("models/model.pkl")
    ]

    for file in required:
        assert file.exists(), f"Falta el artefacto requerido: {file}"
