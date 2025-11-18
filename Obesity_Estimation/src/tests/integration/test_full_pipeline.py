import json
import pickle
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Importar tu entrenamiento
from Obesity_Estimation.models.train_model import ModelTrainer, PathConfig


# ============================================================
#  HELPERS
# ============================================================

def create_raw_dataset(tmp_path):
    """Crea dataset crudo mínimo y realista para pruebas de integración."""
    df = pd.DataFrame({
        "Age": [21, 45, 30, 18],
        "Gender": ["Male", "Female", "Female", "Male"],
        "Height": [1.70, 1.60, 1.75, 1.80],
        "Weight": [70, 65, 85, 90],
        "NObeyesdad": ["Normal_Weight", "Obesity_Type_I", "Overweight", "Obesity_Type_II"]
    })

    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)

    raw_file = raw_dir / "obesity_estimation_final.csv"
    df.to_csv(raw_file, index=False)

    return raw_file


def run_make_dataset(tmp_path):
    """Simula notebooks/make_dataset.py pero como función de test."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    raw_file = tmp_path / "data/raw/obesity_estimation_final.csv"
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_file)

    # Codificación
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Separar
    X = df.drop("NObeyesdad", axis=1)
    y = df["NObeyesdad"]

    # Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Guardar procesados
    X_train.to_csv(processed / "X_train.csv", index=False)
    y_train.to_csv(processed / "y_train.csv", index=False)
    X_test.to_csv(processed / "X_test.csv", index=False)
    y_test.to_csv(processed / "y_test.csv", index=False)

    return processed


# ============================================================
#  FULL PIPELINE TEST
# ============================================================

def test_full_pipeline(tmp_path):
    """
    PRUEBA DE INTEGRACIÓN COMPLETA DEL PIPELINE:
    1) Crear dataset crudo
    2) Ejecutar transformación (make_dataset)
    3) Ejecutar model training (con MLflow mock)
    4) Validar que los artefactos finales existen y funcionan
    """

    # ----------------------------------------------------------------------
    # 1. Crear dataset crudo
    # ----------------------------------------------------------------------
    raw_file = create_raw_dataset(tmp_path)
    assert raw_file.exists(), "El archivo crudo no fue creado."

    # ----------------------------------------------------------------------
    # 2. Ejecutar procesamiento
    # ----------------------------------------------------------------------
    processed = run_make_dataset(tmp_path)
    assert (processed / "X_train.csv").exists()
    assert (processed / "y_train.csv").exists()

    # ----------------------------------------------------------------------
    # 3. Ejecutar entrenamiento usando MLflow mock
    # ----------------------------------------------------------------------
    path_cfg = PathConfig(
        data_dir=processed,
        models_dir=tmp_path / "models"
    )

    trainer = ModelTrainer(path_config=path_cfg)

    with patch("mlflow.start_run") as mock_run, \
         patch("mlflow.log_param"), \
         patch("mlflow.sklearn.log_model"):

        # Mock MLflow run context
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value.info.run_id = "INTEGRATION_RUN_999"
        mock_run.return_value = mock_ctx

        run_id = trainer.run_training_pipeline()

    assert run_id == "INTEGRATION_RUN_999"

    # ----------------------------------------------------------------------
    # 4. Validación final de artefactos
    # ----------------------------------------------------------------------

    # Modelo pkl
    model_files = list((tmp_path / "models").glob("*.pkl"))
    assert len(model_files) == 1, "No se guardó el modelo .pkl"

    # JSON
    model_info_path = tmp_path / "models/model_info.json"
    assert model_info_path.exists(), "model_info.json no se creó"

    with open(model_info_path, "r") as f:
        info = json.load(f)

    assert info["run_id"] == "INTEGRATION_RUN_999"
    assert info["model_uri"] == "runs:/INTEGRATION_RUN_999/obesity_model"

    # Modelo carga co
