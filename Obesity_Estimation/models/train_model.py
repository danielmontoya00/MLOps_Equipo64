"""
Model training script for obesity classification using Random Forest.
Includes MLflow tracking, local model persistence, and writes models/model_info.json
so downstream services (e.g., FastAPI) can load the exact model version.
"""

import os
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Carpeta relativa para MLflow artifacts y tracking
mlruns_dir = Path("mlruns").resolve()
mlruns_dir.mkdir(parents=True, exist_ok=True)
# Backend de tracking seguro y cross-platform
mlflow.set_tracking_uri(f"sqlite:///{Path('mlflow.db').resolve()}")


# -------------------- CONFIG DATACLASSES --------------------
@dataclass
class ModelConfig:
    """Configuration for the Random Forest model."""
    n_estimators: int = 50
    random_state: int = 42
    n_jobs: int = -1
    model_type: str = "RandomForestClassifier"


@dataclass
class PathConfig:
    """Configuration for file paths."""
    data_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    x_train_file: str = "X_train.csv"
    y_train_file: str = "y_train.csv"
    run_id_file: str = "current_run_id.txt"
    model_info_json: str = "model_info.json"   # <-- NUEVO


# -------------------- TRAINER --------------------
class ModelTrainer:
    """Handles model training, saving, and MLflow tracking."""

    def __init__(
        self,
        model_config: ModelConfig = ModelConfig(),
        path_config: PathConfig = PathConfig(),
        experiment_name: str = "Clasificación de Obesidad - RF",
        registered_model_name: str | None = None,
    ):
        """
        Args:
            registered_model_name: nombre en el MLflow Model Registry.
                Si no se pasa, se toma de la variable de entorno
                MLFLOW_REGISTERED_MODEL_NAME o 'RandomForestObesityModel' por defecto.
        """
        self.model_config = model_config
        self.path_config = path_config
        self.experiment_name = experiment_name
        self.registered_model_name = registered_model_name or os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME", "RandomForestObesityModel"
        )
        self.model = None

    # ------------- DATA -------------
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load training data from CSV files."""
        x_train_path = self.path_config.data_dir / self.path_config.x_train_file
        y_train_path = self.path_config.data_dir / self.path_config.y_train_file

        if not x_train_path.exists():
            raise FileNotFoundError(f"Training data not found: {x_train_path}")
        if not y_train_path.exists():
            raise FileNotFoundError(f"Training labels not found: {y_train_path}")

        X_train = pd.read_csv(x_train_path)
        y_train = pd.read_csv(y_train_path).values.ravel()

        print(f"Loaded training data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
        return X_train, y_train

    # ------------- MODEL -------------
    def create_model(self) -> RandomForestClassifier:
        """Create a Random Forest classifier with configured parameters."""
        return RandomForestClassifier(
            n_estimators=self.model_config.n_estimators,
            random_state=self.model_config.random_state,
            n_jobs=self.model_config.n_jobs,
        )

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the Random Forest model."""
        print("Entrenando el modelo RandomForestClassifier...")
        self.model = self.create_model()
        self.model.fit(X_train, y_train)
        print("Entrenamiento completado.")

    # ------------- PERSISTENCE -------------
    def save_model_locally(self, run_id: str) -> Path:
        """Save the trained model to local filesystem."""
        self.path_config.models_dir.mkdir(parents=True, exist_ok=True)
        model_filepath = self.path_config.models_dir / f"obesity_classifier_{run_id}.pkl"
        with open(model_filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Modelo local guardado: {model_filepath}")
        return model_filepath

    def save_run_id(self, run_id: str) -> None:
        """Save the current run ID to a text file for downstream scripts."""
        self.path_config.models_dir.mkdir(parents=True, exist_ok=True)
        out = self.path_config.models_dir / self.path_config.run_id_file
        with open(out, "w", encoding="utf-8") as f:
            f.write(run_id)
        print(f"Run ID guardado: {out}")

    def write_model_info_json(self, run_id: str, model_uri: str) -> Path:
        """Write models/model_info.json with run_id and model_uri for serving."""
        self.path_config.models_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "run_id": run_id,
            "model_uri": model_uri,  # e.g., runs:/<run_id>/obesity_model
            "registered_model_name": self.registered_model_name,
        }
        out_path = self.path_config.models_dir / self.path_config.model_info_json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"Archivo JSON creado: {out_path}")
        return out_path

    # ------------- MLflow -------------
    def log_to_mlflow(self) -> str:
        """
        Log model and params to MLflow.
        Returns:
            model_uri (runs:/<run_id>/obesity_model)
        """
        # Log hyperparameters
        mlflow.log_param("n_estimators", self.model_config.n_estimators)
        mlflow.log_param("random_state", self.model_config.random_state)
        mlflow.log_param("model_type", self.model_config.model_type)
        mlflow.log_param("n_jobs", self.model_config.n_jobs)

        # Log the model (artifact_path define el subdirectorio del run)
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="obesity_model",
            registered_model_name=self.registered_model_name,
        )
        # El run activo lo maneja el with mlflow.start_run()
        run_id = mlflow.active_run().info.run_id  # type: ignore
        # La URI para cargar el modelo luego:
        model_uri = f"runs:/{run_id}/obesity_model"
        print("Modelo y parámetros registrados en MLflow.")
        return model_uri

    # ------------- PIPELINE -------------
    def run_training_pipeline(self) -> str:
        """Execute the complete training pipeline with MLflow tracking."""
        print("Ejecutando script de entrenamiento...")

        # Set up MLflow experiment
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"MLflow Run ID: {run_id}")

            # Load data
            X_train, y_train = self.load_training_data()

            # Train model
            self.train_model(X_train, y_train)

            # Log to MLflow and build model_uri
            model_uri = self.log_to_mlflow()

            # Save locally and write pointers for downstream
            self.save_model_locally(run_id)
            self.save_run_id(run_id)
            self.write_model_info_json(run_id, model_uri)

            print("Script de entrenamiento finalizado y datos registrados en MLflow.")
            return run_id


# -------------------- ENTRYPOINT --------------------
def main():
    trainer = ModelTrainer()
    run_id = trainer.run_training_pipeline()
    print(f"Training completed successfully. Run ID: {run_id}")


if __name__ == "__main__":
    main()
