"""
Model evaluation script for obesity classification.
Loads trained model, evaluates on test data, and logs metrics to MLflow.
"""
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import mlflow
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    run_id_file: str = "current_run_id.txt"
    data_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    x_test_file: str = "X_test.csv"
    y_test_file: str = "y_test.csv"
    metrics_file: str = "metrics.txt"
    nested_run: bool = True
    log_per_class_metrics: bool = True


class ModelEvaluator:
    """Handles model evaluation and metrics logging."""
    
    def __init__(self, config: EvaluationConfig = EvaluationConfig()):
        self.config = config
        self.model: Optional[RandomForestClassifier] = None
        self.run_id: Optional[str] = None
        
    def load_run_id(self) -> str:
        """Load the MLflow run ID from file.
        
        Returns:
            MLflow run ID
            
        Raises:
            FileNotFoundError: If run ID file doesn't exist
            ValueError: If run ID file is empty
        """
        run_id_path = Path(self.config.run_id_file)
        
        if not run_id_path.exists():
            raise FileNotFoundError(
                f"Run ID file not found: {run_id_path}. "
                "Please run train.py first."
            )
        
        run_id = run_id_path.read_text().strip()
        
        if not run_id:
            raise ValueError(f"Run ID file is empty: {run_id_path}")
        
        print(f"Adjuntando evaluación al MLflow Run ID: {run_id}")
        return run_id
    
    def load_model(self, run_id: str) -> RandomForestClassifier:
        """Load the trained model from local filesystem.
        
        Args:
            run_id: MLflow run ID used in model filename
            
        Returns:
            Loaded scikit-learn model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        model_path = self.config.models_dir / f'obesity_classifier_{run_id}.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Expected model for run ID: {run_id}"
            )
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Modelo cargado desde: {model_path}")
        return model
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load test data from CSV files.
        
        Returns:
            Tuple of (X_test, y_test)
            
        Raises:
            FileNotFoundError: If test files don't exist
        """
        x_test_path = self.config.data_dir / self.config.x_test_file
        y_test_path = self.config.data_dir / self.config.y_test_file
        
        if not x_test_path.exists():
            raise FileNotFoundError(f"Test data not found: {x_test_path}")
        if not y_test_path.exists():
            raise FileNotFoundError(f"Test labels not found: {y_test_path}")
        
        X_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).values.ravel()
        
        print(f"Loaded test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
        return X_test, y_test
    
    def evaluate_model(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("Evaluando el modelo...")
        predictions = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report_dict = classification_report(y_test, predictions, output_dict=True)
        report_text = classification_report(y_test, predictions)
        
        print(f"Accuracy del modelo: {accuracy:.4f}")
        
        return {
            'predictions': predictions,
            'accuracy': accuracy,
            'report_dict': report_dict,
            'report_text': report_text
        }
    
    def log_metrics_to_mlflow(self, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics to MLflow.
        
        Args:
            metrics: Dictionary containing evaluation metrics
        """
        # Log overall accuracy
        mlflow.log_metric("accuracy", metrics['accuracy'])
        
        # Log per-class metrics if configured
        if self.config.log_per_class_metrics:
            report_dict = metrics['report_dict']
            
            for class_name, class_metrics in report_dict.items():
                # Skip aggregate metrics (accuracy, macro avg, weighted avg)
                if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                
                # Skip if not a dictionary (happens with accuracy key)
                if not isinstance(class_metrics, dict):
                    continue
                
                # Log precision, recall, f1-score for each class
                safe_class_name = str(class_name).replace(' ', '_').lower()
                
                if 'precision' in class_metrics:
                    mlflow.log_metric(
                        f"precision_{safe_class_name}", 
                        class_metrics['precision']
                    )
                if 'recall' in class_metrics:
                    mlflow.log_metric(
                        f"recall_{safe_class_name}", 
                        class_metrics['recall']
                    )
                if 'f1-score' in class_metrics:
                    mlflow.log_metric(
                        f"f1_{safe_class_name}", 
                        class_metrics['f1-score']
                    )
            
            # Log aggregate metrics
            if 'weighted avg' in report_dict:
                weighted_avg = report_dict['weighted avg']
                mlflow.log_metric("weighted_avg_precision", weighted_avg['precision'])
                mlflow.log_metric("weighted_avg_recall", weighted_avg['recall'])
                mlflow.log_metric("weighted_avg_f1", weighted_avg['f1-score'])
        
        print("Métricas registradas en MLflow.")
    
    def save_metrics_report(self, metrics: Dict[str, Any]) -> Path:
        """Save metrics report to local filesystem.
        
        Args:
            metrics: Dictionary containing evaluation metrics
            
        Returns:
            Path to saved metrics file
        """
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.config.reports_dir / self.config.metrics_file
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write("\n--- Reporte de Clasificación ---\n")
            f.write(metrics['report_text'])
        
        print(f"Reporte guardado en: {metrics_path}")
        return metrics_path
    
    def log_artifacts_to_mlflow(self, metrics_path: Path) -> None:
        """Log metrics report as MLflow artifact.
        
        Args:
            metrics_path: Path to metrics report file
        """
        mlflow.log_artifact(str(metrics_path))
        print("Reporte registrado como artefacto en MLflow.")
    
    def run_evaluation_pipeline(self) -> Dict[str, Any]:
        """Execute the complete evaluation pipeline with MLflow tracking.
        
        Returns:
            Dictionary containing evaluation results
            
        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If run ID is invalid
        """
        print("Ejecutando script de evaluación...")
        
        # Load run ID
        self.run_id = self.load_run_id()
        
        # Resume the existing MLflow run
        with mlflow.start_run(run_id=self.run_id, nested=self.config.nested_run):
            # Load model and test data
            self.model = self.load_model(self.run_id)
            X_test, y_test = self.load_test_data()
            
            # Evaluate model
            metrics = self.evaluate_model(X_test, y_test)
            
            # Log to MLflow
            self.log_metrics_to_mlflow(metrics)
            
            # Save and log report
            metrics_path = self.save_metrics_report(metrics)
            self.log_artifacts_to_mlflow(metrics_path)
            
            print("Script de evaluación finalizado.")
            return metrics


def main():
    """Main entry point for the evaluation script."""
    try:
        evaluator = ModelEvaluator()
        metrics = evaluator.run_evaluation_pipeline()
        print(f"Evaluation completed successfully. Accuracy: {metrics['accuracy']:.4f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()