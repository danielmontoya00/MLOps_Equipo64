# src/tests/unit/test_train_model.py
import json
import pickle
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Agrega la carpeta src al path para poder importar módulos

# Agrega la carpeta src al path
root_path = Path(__file__).resolve().parents[3]  # src/tests/unit -> subir 3 niveles -> Obesity_Estimation
sys.path.insert(0, str(root_path))

#sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from models.train_model import ModelTrainer, PathConfig
from features.create_feature import add_bmi_features
from api.app import app

def test_train_model_with_mlflow_mock(tmp_path):
    """Prueba entrenamiento y guardado usando MLflow (mock)."""

    # --- Preparar directorios temporales ---
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    # --- Crear CSVs falsos ---
    X_train = pd.DataFrame({"Age": [20, 30], "Weight": [70, 80]})
    y_train = pd.Series([0, 1])

    X_train.to_csv(processed / "X_train.csv", index=False)
    y_train.to_csv(processed / "y_train.csv", index=False)

    # --- Config personalizado ---
    path_cfg = PathConfig(
        data_dir=processed,
        models_dir=models_dir
    )

    trainer = ModelTrainer(path_config=path_cfg)

    # --- Mock de MLflow ---
    with patch("mlflow.start_run") as mock_start_run, \
         patch("mlflow.log_param"), \
         patch("mlflow.sklearn.log_model") as mock_log_model, \
         patch("mlflow.active_run") as mock_active_run:

        # Configurar context manager de start_run
        mock_context = MagicMock()
        mock_context.__enter__.return_value.info.run_id = "TEST_RUN_123"
        mock_start_run.return_value = mock_context

        # Mock para active_run
        mock_active_run.return_value = mock_context.__enter__.return_value

        # --- Ejecutar pipeline ---
        run_id = trainer.run_training_pipeline()

    # --- Validaciones ---
    assert run_id == "TEST_RUN_123"

    # Verificar que se guardó un modelo
    pkl_files = list(models_dir.glob("*.pkl"))
    assert len(pkl_files) == 1

    # Verificar que se creó model_info.json
    info_file = models_dir / "model_info.json"
    assert info_file.exists()

    with open(info_file, "r") as f:
        data = json.load(f)

    assert data["run_id"] == "TEST_RUN_123"
    assert data["model_uri"] == "runs:/TEST_RUN_123/obesity_model"
    assert data["registered_model_name"] == "RandomForestObesityModel"

    # Validar que el modelo realmente funciona
    with open(pkl_files[0], "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_train)
    assert len(preds) == len(X_train)
