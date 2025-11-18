"""
Model Performance Monitor

Evaluates model performance on drifted data, compares with baseline,
and logs results to MLflow with alerts and recommended actions.
"""

import pickle
import json
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from .drift_detector import DriftDetector, DriftReport, DriftThresholds


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceComparison:
    """Comparison between baseline and current performance."""
    baseline_metrics: PerformanceMetrics
    current_metrics: PerformanceMetrics
    accuracy_drop: float
    f1_drop: float
    degradation_score: float  # 0-1, higher = worse degradation
    is_degraded: bool
    
    def summary(self) -> str:
        """Generate text summary."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                 PERFORMANCE COMPARISON REPORT                ║
╚══════════════════════════════════════════════════════════════╝

BASELINE PERFORMANCE:
  Accuracy:  {self.baseline_metrics.accuracy:.4f}
  Precision: {self.baseline_metrics.precision_weighted:.4f}
  Recall:    {self.baseline_metrics.recall_weighted:.4f}
  F1-Score:  {self.baseline_metrics.f1_weighted:.4f}

CURRENT PERFORMANCE:
  Accuracy:  {self.current_metrics.accuracy:.4f}
  Precision: {self.current_metrics.precision_weighted:.4f}
  Recall:    {self.current_metrics.recall_weighted:.4f}
  F1-Score:  {self.current_metrics.f1_weighted:.4f}

PERFORMANCE CHANGES:
  Accuracy Drop:  {self.accuracy_drop:.4f} ({self.accuracy_drop*100:.2f}%)
  F1-Score Drop:  {self.f1_drop:.4f} ({self.f1_drop*100:.2f}%)
  Degradation Score: {self.degradation_score:.3f}
  Status: {'⚠️  DEGRADED' if self.is_degraded else '✅ STABLE'}
"""


@dataclass
class MonitoringAlert:
    """Alert configuration for monitoring."""
    drift_detected: bool
    performance_degraded: bool
    alert_level: str  # NONE, LOW, MEDIUM, HIGH, CRITICAL
    recommended_actions: list
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceMonitor:
    """Monitor model performance and detect degradation."""
    
    def __init__(
        self,
        models_dir: Path = Path("models"),
        reports_dir: Path = Path("reports/monitoring"),
        accuracy_threshold: float = 0.05,  # Alert if accuracy drops > 5%
        f1_threshold: float = 0.05,  # Alert if F1 drops > 5%
    ):
        """
        Initialize performance monitor.
        
        Args:
            models_dir: Directory with saved models
            reports_dir: Directory to save monitoring reports
            accuracy_threshold: Max acceptable accuracy drop
            f1_threshold: Max acceptable F1 drop
        """
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.accuracy_threshold = accuracy_threshold
        self.f1_threshold = f1_threshold
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self, run_id: Optional[str] = None) -> Any:
        """
        Load trained model.
        
        Args:
            run_id: MLflow run ID. If None, loads from model_info.json
            
        Returns:
            Loaded model
        """
        if run_id is None:
            # Load from model_info.json
            info_path = self.models_dir / "model_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                run_id = info['run_id']
            else:
                # Try current_run_id.txt
                run_id_path = self.models_dir / "current_run_id.txt"
                run_id = run_id_path.read_text().strip()
        
        model_path = self.models_dir / f"obesity_classifier_{run_id}.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded from {model_path}")
        return model
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            PerformanceMetrics object
        """
        return PerformanceMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision_weighted=precision_score(y_true, y_pred, average='weighted', zero_division=0),
            recall_weighted=recall_score(y_true, y_pred, average='weighted', zero_division=0),
            f1_weighted=f1_score(y_true, y_pred, average='weighted', zero_division=0)
        )
    
    def evaluate_on_dataset(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Tuple[PerformanceMetrics, np.ndarray]:
        """
        Evaluate model on a dataset.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            
        Returns:
            Tuple of (metrics, predictions)
        """
        predictions = model.predict(X)
        metrics = self.calculate_metrics(y, predictions)
        return metrics, predictions
    
    def compare_performance(
        self,
        baseline_metrics: PerformanceMetrics,
        current_metrics: PerformanceMetrics
    ) -> PerformanceComparison:
        """
        Compare baseline and current performance.
        
        Args:
            baseline_metrics: Baseline performance metrics
            current_metrics: Current performance metrics
            
        Returns:
            PerformanceComparison object
        """
        accuracy_drop = baseline_metrics.accuracy - current_metrics.accuracy
        f1_drop = baseline_metrics.f1_weighted - current_metrics.f1_weighted
        
        # Calculate degradation score (0-1)
        degradation_score = max(0, (accuracy_drop + f1_drop) / 2)
        
        # Check if degraded
        is_degraded = (
            accuracy_drop > self.accuracy_threshold or
            f1_drop > self.f1_threshold
        )
        
        return PerformanceComparison(
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            accuracy_drop=accuracy_drop,
            f1_drop=f1_drop,
            degradation_score=degradation_score,
            is_degraded=is_degraded
        )
    
    def generate_alert(
        self,
        drift_report: DriftReport,
        performance_comparison: PerformanceComparison
    ) -> MonitoringAlert:
        """
        Generate monitoring alert based on drift and performance.
        
        Args:
            drift_report: Data drift detection report
            performance_comparison: Performance comparison results
            
        Returns:
            MonitoringAlert with recommended actions
        """
        drift_detected = drift_report.drift_score > 0.1
        performance_degraded = performance_comparison.is_degraded
        
        # Determine alert level
        if drift_report.alert_level == "CRITICAL" or performance_comparison.degradation_score > 0.2:
            alert_level = "CRITICAL"
        elif drift_report.alert_level == "HIGH" or performance_comparison.degradation_score > 0.1:
            alert_level = "HIGH"
        elif drift_report.alert_level == "MEDIUM" or performance_comparison.degradation_score > 0.05:
            alert_level = "MEDIUM"
        elif drift_detected or performance_degraded:
            alert_level = "LOW"
        else:
            alert_level = "NONE"
        
        # Generate recommendations
        recommendations = []
        
        if alert_level == "CRITICAL":
            recommendations.append("🔴 URGENT: Immediate model retraining required")
            recommendations.append("🔴 Consider rolling back to previous model version")
            recommendations.append("🔴 Investigate root cause of distribution shift")
            
        elif alert_level == "HIGH":
            recommendations.append("🟠 Schedule model retraining within 24-48 hours")
            recommendations.append("🟠 Review feature engineering pipeline")
            recommendations.append("🟠 Analyze drifted features for data quality issues")
            
        elif alert_level == "MEDIUM":
            recommendations.append("🟡 Plan model retraining in next sprint")
            recommendations.append("🟡 Monitor trend - check if drift is increasing")
            recommendations.append("🟡 Review data collection process")
            
        elif alert_level == "LOW":
            recommendations.append("🟢 Continue monitoring")
            recommendations.append("🟢 Document drift patterns for future reference")
            
        else:
            recommendations.append("✅ No action required - system stable")
        
        # Add specific recommendations based on findings
        if drift_detected:
            drifted_features = drift_report.get_drifted_features()
            if len(drifted_features) > 0:
                recommendations.append(
                    f"📊 Investigate drift in features: {', '.join(drifted_features[:5])}"
                )
        
        if performance_degraded:
            recommendations.append(
                f"📉 Performance degraded by {performance_comparison.degradation_score*100:.1f}%"
            )
        
        return MonitoringAlert(
            drift_detected=drift_detected,
            performance_degraded=performance_degraded,
            alert_level=alert_level,
            recommended_actions=recommendations
        )
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.reports_dir / f"confusion_matrix_{title.replace(' ', '_').lower()}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_metric_comparison(
        self,
        baseline_metrics: PerformanceMetrics,
        current_metrics: PerformanceMetrics,
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot comparison of baseline vs current metrics.
        
        Args:
            baseline_metrics: Baseline metrics
            current_metrics: Current metrics
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        baseline_values = [
            baseline_metrics.accuracy,
            baseline_metrics.precision_weighted,
            baseline_metrics.recall_weighted,
            baseline_metrics.f1_weighted
        ]
        current_values = [
            current_metrics.accuracy,
            current_metrics.precision_weighted,
            current_metrics.recall_weighted,
            current_metrics.f1_weighted
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        bars2 = ax.bar(x + width/2, current_values, width, label='Current', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance: Baseline vs Current')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.reports_dir / "metrics_comparison.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_drift_heatmap(
        self,
        drift_report: DriftReport,
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot heatmap of drift magnitude per feature.
        
        Args:
            drift_report: Drift detection report
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        # Extract feature names and drift magnitudes
        features = []
        magnitudes = []
        psi_scores = []
        
        for name, result in drift_report.feature_results.items():
            features.append(name)
            magnitudes.append(result.drift_magnitude)
            psi_scores.append(result.psi_score if result.psi_score else 0)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Feature': features,
            'Drift Magnitude': magnitudes,
            'PSI Score': psi_scores
        })
        df = df.sort_values('Drift Magnitude', ascending=False)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(features) * 0.3)))
        
        # Drift magnitude
        ax1.barh(df['Feature'], df['Drift Magnitude'], color='steelblue')
        ax1.set_xlabel('Drift Magnitude')
        ax1.set_title('Feature Drift Magnitude')
        ax1.axvline(x=0.5, color='r', linestyle='--', label='Threshold', alpha=0.5)
        ax1.legend()
        
        # PSI scores
        colors = ['red' if x > 0.2 else 'orange' if x > 0.1 else 'green' 
                 for x in df['PSI Score']]
        ax2.barh(df['Feature'], df['PSI Score'], color=colors)
        ax2.set_xlabel('PSI Score')
        ax2.set_title('Population Stability Index (PSI)')
        ax2.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Small change')
        ax2.axvline(x=0.2, color='red', linestyle='--', alpha=0.5, label='Significant drift')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.reports_dir / "drift_heatmap.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def log_to_mlflow(
        self,
        drift_report: DriftReport,
        performance_comparison: PerformanceComparison,
        alert: MonitoringAlert,
        artifacts: Dict[str, Path],
        experiment_name: str = "Model Monitoring",
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Log monitoring results to MLflow.
        
        Args:
            drift_report: Drift detection report
            performance_comparison: Performance comparison
            alert: Monitoring alert
            artifacts: Dictionary of artifact paths
            experiment_name: MLflow experiment name
            tags: Additional tags
            
        Returns:
            MLflow run ID
        """
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            
            # Log drift metrics
            mlflow.log_metric("drift_score", drift_report.drift_score)
            mlflow.log_metric("n_drifted_features", drift_report.n_drifted_features)
            mlflow.log_metric("drift_ratio", drift_report.n_drifted_features / drift_report.n_features)
            
            # Log performance metrics
            mlflow.log_metric("baseline_accuracy", performance_comparison.baseline_metrics.accuracy)
            mlflow.log_metric("current_accuracy", performance_comparison.current_metrics.accuracy)
            mlflow.log_metric("accuracy_drop", performance_comparison.accuracy_drop)
            mlflow.log_metric("baseline_f1", performance_comparison.baseline_metrics.f1_weighted)
            mlflow.log_metric("current_f1", performance_comparison.current_metrics.f1_weighted)
            mlflow.log_metric("f1_drop", performance_comparison.f1_drop)
            mlflow.log_metric("degradation_score", performance_comparison.degradation_score)
            
            # Log alert information
            mlflow.log_param("alert_level", alert.alert_level)
            mlflow.log_param("drift_detected", alert.drift_detected)
            mlflow.log_param("performance_degraded", alert.performance_degraded)
            
            # Log tags
            mlflow.set_tag("monitoring_type", "drift_and_performance")
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Log artifacts
            for name, path in artifacts.items():
                if path.exists():
                    mlflow.log_artifact(str(path))
            
            # Log alert as JSON
            alert_path = self.reports_dir / "alert.json"
            with open(alert_path, 'w') as f:
                json.dump(alert.to_dict(), f, indent=2)
            mlflow.log_artifact(str(alert_path))
            
            # Log drift report
            drift_summary_path = self.reports_dir / "drift_report.txt"
            with open(drift_summary_path, 'w') as f:
                f.write(drift_report.summary())
            mlflow.log_artifact(str(drift_summary_path))
            
            # Log performance comparison
            perf_summary_path = self.reports_dir / "performance_comparison.txt"
            with open(perf_summary_path, 'w') as f:
                f.write(performance_comparison.summary())
            mlflow.log_artifact(str(perf_summary_path))
            
            print(f"\n✅ Monitoring results logged to MLflow (Run ID: {run_id})")
            
        return run_id


def main():
    """Demo: Full monitoring pipeline."""
    print("=== Model Performance Monitor Demo ===\n")
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Load model
    model = monitor.load_model()
    
    # Load baseline data
    baseline_path = Path("data/processed")
    X_baseline = pd.read_csv(baseline_path / "X_test.csv")
    y_baseline = pd.read_csv(baseline_path / "y_test.csv").values.ravel()
    
    # Evaluate on baseline
    print("Evaluating on baseline data...")
    baseline_metrics, _ = monitor.evaluate_on_dataset(model, X_baseline, y_baseline)
    print(f"Baseline accuracy: {baseline_metrics.accuracy:.4f}")
    
    # Load drifted data
    drift_path = Path("data/monitoring")
    if not (drift_path / "X_mean_shift.csv").exists():
        print("⚠️  Drifted data not found. Run drift_simulator.py first.")
        return
    
    X_drifted = pd.read_csv(drift_path / "X_mean_shift.csv")
    y_drifted = pd.read_csv(drift_path / "y_mean_shift.csv").values.ravel()
    
    # Evaluate on drifted data
    print("Evaluating on drifted data...")
    current_metrics, current_pred = monitor.evaluate_on_dataset(model, X_drifted, y_drifted)
    print(f"Current accuracy: {current_metrics.accuracy:.4f}")
    
    # Detect drift
    print("\nDetecting drift...")
    detector = DriftDetector()
    drift_report = detector.detect_dataset_drift(X_baseline, X_drifted)
    
    # Compare performance
    print("\nComparing performance...")
    comparison = monitor.compare_performance(baseline_metrics, current_metrics)
    
    # Generate alert
    alert = monitor.generate_alert(drift_report, comparison)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    artifacts = {
        'confusion_matrix': monitor.plot_confusion_matrix(
            y_drifted, current_pred, "Current Data"
        ),
        'metrics_comparison': monitor.plot_metric_comparison(
            baseline_metrics, current_metrics
        ),
        'drift_heatmap': monitor.plot_drift_heatmap(drift_report)
    }
    
    # Print reports
    print(drift_report.summary())
    print(comparison.summary())
    
    print("\n" + "="*64)
    print(f"ALERT LEVEL: {alert.alert_level}")
    print("="*64)
    print("RECOMMENDED ACTIONS:")
    for action in alert.recommended_actions:
        print(f"  {action}")
    print("="*64)
    
    # Log to MLflow
    print("\nLogging to MLflow...")
    monitor.log_to_mlflow(
        drift_report,
        comparison,
        alert,
        artifacts,
        tags={'drift_scenario': 'mean_shift'}
    )
    
    print("\n✅ Monitoring complete!")


if __name__ == "__main__":
    main()
