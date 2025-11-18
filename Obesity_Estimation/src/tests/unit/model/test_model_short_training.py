import sys
from pathlib import Path
import pandas as pd
import pickle

# Aseguramos que la carpeta raíz del proyecto esté en sys.path
sys.path.append(str(Path(__file__).resolve().parents[4]))

from models.train_model import ModelTrainer, PathConfig


def test_short_training(tmp_path):
    """
    Entrena el modelo con un dataset mínimo para validar que el pipeline corre rápido.
    """
    df = pd.DataFrame({
        "Age": [20, 30, 40],
        "Gender": [1, 0, 1],
        "Height": [1.70, 1.60, 1.75],
        "Weight": [65, 80, 90],
        "NObeyesdad": [0, 1, 2],
    })

    # Crear carpeta data
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Guardar X_train y y_train por separado
    df.drop(columns=["NObeyesdad"]).to_csv(data_dir / "X_train.csv", index=False)
    df["NObeyesdad"].to_csv(data_dir / "y_train.csv", index=False)

    # Config
    cfg = PathConfig(data_dir=data_dir, models_dir=tmp_path / "models")
    trainer = ModelTrainer(path_config=cfg)

    # Ejecutar pipeline
    run_id = trainer.run_training_pipeline()

    # Validaciones
    assert run_id is not None
    model_files = list((tmp_path / "models").glob("*.pkl"))
    assert len(model_files) == 1, "No se generó modelo en entrenamiento corto"


def test_model_short_predict(tmp_path):
    """
    Carga el modelo entrenado en short training y prueba una predicción mínima.
    """
    df = pd.DataFrame({
        "Age": [20, 30, 40],
        "Gender": [1, 0, 1],
        "Height": [1.70, 1.60, 1.75],
        "Weight": [65, 80, 90],
        "NObeyesdad": [0, 1, 2],
    })

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Guardar X_train y y_train
    df.drop(columns=["NObeyesdad"]).to_csv(data_dir / "X_train.csv", index=False)
    df["NObeyesdad"].to_csv(data_dir / "y_train.csv", index=False)

    cfg = PathConfig(data_dir=data_dir, models_dir=tmp_path / "models")
    trainer = ModelTrainer(path_config=cfg)
    trainer.run_training_pipeline()

    # Cargar modelo
    model_files = list((tmp_path / "models").glob("*.pkl"))
    with open(model_files[0], "rb") as f:
        model = pickle.load(f)

    X_test = df.drop("NObeyesdad", axis=1)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)