"""
Model training script for obesity classification using Random Forest.
Includes MLflow tracking and local model persistence.
"""
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


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


class ModelTrainer:
    """Handles model training, saving, and MLflow tracking."""
    
    def __init__(
        self, 
        model_config: ModelConfig = ModelConfig(),
        path_config: PathConfig = PathConfig(),
        experiment_name: str = "Clasificación de Obesidad - RF"
    ):
        self.model_config = model_config
        self.path_config = path_config
        self.experiment_name = experiment_name
        self.model = None
        
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load training data from CSV files.
        
        Returns:
            Tuple of (X_train, y_train)
            
        Raises:
            FileNotFoundError: If training files don't exist
        """
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
    
    def create_model(self) -> RandomForestClassifier:
        """Create a Random Forest classifier with configured parameters.
        
        Returns:
            Initialized RandomForestClassifier
        """
        return RandomForestClassifier(
            n_estimators=self.model_config.n_estimators,
            random_state=self.model_config.random_state,
            n_jobs=self.model_config.n_jobs
        )
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Entrenando el modelo RandomForestClassifier...")
        self.model = self.create_model()
        self.model.fit(X_train, y_train)
        print("Entrenamiento completado.")
    
    def save_model_locally(self, run_id: str) -> Path:
        """Save the trained model to local filesystem.
        
        Args:
            run_id: MLflow run ID to include in filename
            
        Returns:
            Path to saved model file
        """
        self.path_config.models_dir.mkdir(parents=True, exist_ok=True)
        model_filepath = self.path_config.models_dir / f'obesity_classifier_{run_id}.pkl'
        
        with open(model_filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Modelo local guardado: {model_filepath}")
        return model_filepath
    
    def save_run_id(self, run_id: str) -> None:
        """Save the current run ID to a text file for downstream scripts.
        
        Args:
            run_id: MLflow run ID
        """
        with open(self.path_config.run_id_file, 'w') as f:
            f.write(run_id)
        print(f"Run ID guardado: {run_id}")
    
    def log_to_mlflow(self, run_id: str) -> None:
        """Log model parameters and artifacts to MLflow.
        
        Args:
            run_id: MLflow run ID
        """
        # Log hyperparameters
        mlflow.log_param("n_estimators", self.model_config.n_estimators)
        mlflow.log_param("random_state", self.model_config.random_state)
        mlflow.log_param("model_type", self.model_config.model_type)
        mlflow.log_param("n_jobs", self.model_config.n_jobs)
        
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="obesity_model",
            registered_model_name="RandomForestObesityModel"
        )
        print("Modelo y parámetros registrados en MLflow.")
    
    def run_training_pipeline(self) -> str:
        """Execute the complete training pipeline with MLflow tracking.
        
        Returns:
            MLflow run ID
        """
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
            
            # Log to MLflow
            self.log_to_mlflow(run_id)
            
            # Save locally
            self.save_model_locally(run_id)
            self.save_run_id(run_id)
            
            print("Script de entrenamiento finalizado y datos registrados en MLflow.")
            return run_id


def main():
    """Main entry point for the training script."""
    trainer = ModelTrainer()
    run_id = trainer.run_training_pipeline()
    print(f"Training completed successfully. Run ID: {run_id}")


if __name__ == "__main__":
    main()