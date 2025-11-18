"""
Data Drift Monitoring Pipeline

Complete end-to-end pipeline for:
1. Simulating data drift scenarios
2. Detecting drift using statistical tests
3. Evaluating model performance degradation
4. Generating alerts and recommendations
5. Logging everything to MLflow
"""

import argparse
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import json

from src.monitoring.drift_simulator import DriftSimulator, DriftConfig, DriftType
from src.monitoring.drift_detector import DriftDetector, DriftThresholds
from src.monitoring.performance_monitor import PerformanceMonitor


def run_monitoring_pipeline(
    drift_scenario: str = "mean_shift",
    drift_intensity: float = 0.3,
    output_dir: Path = Path("data/monitoring"),
    generate_new_data: bool = True
) -> Dict[str, Any]:
    """
    Run complete monitoring pipeline.
    
    Args:
        drift_scenario: Type of drift to simulate
        drift_intensity: Intensity of drift (0-1)
        output_dir: Directory for monitoring data
        generate_new_data: Whether to generate new drifted data
        
    Returns:
        Dictionary with monitoring results
    """
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          DATA DRIFT MONITORING PIPELINE                     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    results = {}
    
    # ========== STEP 1: Generate Drifted Data ==========
    if generate_new_data:
        print("\n" + "="*64)
        print("STEP 1: Generating Drifted Dataset")
        print("="*64)
        
        # Map scenario name to DriftType
        drift_type_map = {
            "mean_shift": DriftType.MEAN_SHIFT,
            "variance_shift": DriftType.VARIANCE_SHIFT,
            "seasonal": DriftType.SEASONAL,
            "feature_missing": DriftType.FEATURE_MISSING,
            "label_drift": DriftType.LABEL_DRIFT,
            "combined": DriftType.COMBINED,
        }
        
        drift_type = drift_type_map.get(drift_scenario, DriftType.MEAN_SHIFT)
        
        config = DriftConfig(
            drift_type=drift_type,
            intensity=drift_intensity,
            missing_probability=0.2,
            seasonal_amplitude=0.5,
            random_state=42
        )
        
        simulator = DriftSimulator(config)
        drift_info = simulator.generate_drifted_dataset(
            output_dir=output_dir,
            suffix=drift_scenario
        )
        
        results['drift_generation'] = drift_info
        print(f"✅ Generated {drift_scenario} drift with intensity {drift_intensity}")
    else:
        print(f"\n⏭️  Skipping data generation, using existing data")
    
    # ========== STEP 2: Load Data ==========
    print("\n" + "="*64)
    print("STEP 2: Loading Data")
    print("="*64)
    
    # Load baseline data
    baseline_path = Path("data/processed")
    X_baseline = pd.read_csv(baseline_path / "X_test.csv")
    y_baseline = pd.read_csv(baseline_path / "y_test.csv").values.ravel()
    print(f"✅ Loaded baseline: {X_baseline.shape}")
    
    # Load monitoring/drifted data
    X_monitoring = pd.read_csv(output_dir / f"X_{drift_scenario}.csv")
    y_monitoring = pd.read_csv(output_dir / f"y_{drift_scenario}.csv").values.ravel()
    print(f"✅ Loaded monitoring: {X_monitoring.shape}")
    
    # ========== STEP 3: Detect Drift ==========
    print("\n" + "="*64)
    print("STEP 3: Detecting Data Drift")
    print("="*64)
    
    detector = DriftDetector(
        thresholds=DriftThresholds(
            ks_pvalue=0.05,
            psi_threshold=0.2,
            chi2_pvalue=0.05,
            js_divergence=0.1
        )
    )
    
    drift_report = detector.detect_dataset_drift(X_baseline, X_monitoring)
    
    print("\n" + drift_report.summary())
    results['drift_report'] = {
        'drift_score': drift_report.drift_score,
        'n_drifted_features': drift_report.n_drifted_features,
        'alert_level': drift_report.alert_level,
        'drifted_features': drift_report.get_drifted_features()
    }
    
    # ========== STEP 4: Evaluate Performance ==========
    print("\n" + "="*64)
    print("STEP 4: Evaluating Model Performance")
    print("="*64)
    
    monitor = PerformanceMonitor(
        accuracy_threshold=0.05,
        f1_threshold=0.05
    )
    
    # Load model
    model = monitor.load_model()
    
    # Evaluate on baseline
    print("\n📊 Evaluating on baseline data...")
    baseline_metrics, baseline_pred = monitor.evaluate_on_dataset(
        model, X_baseline, y_baseline
    )
    print(f"   Baseline Accuracy: {baseline_metrics.accuracy:.4f}")
    print(f"   Baseline F1: {baseline_metrics.f1_weighted:.4f}")
    
    # Evaluate on monitoring data
    print("\n📊 Evaluating on monitoring data...")
    current_metrics, current_pred = monitor.evaluate_on_dataset(
        model, X_monitoring, y_monitoring
    )
    print(f"   Current Accuracy: {current_metrics.accuracy:.4f}")
    print(f"   Current F1: {current_metrics.f1_weighted:.4f}")
    
    # Compare performance
    comparison = monitor.compare_performance(baseline_metrics, current_metrics)
    
    print("\n" + comparison.summary())
    results['performance'] = {
        'baseline_accuracy': baseline_metrics.accuracy,
        'current_accuracy': current_metrics.accuracy,
        'accuracy_drop': comparison.accuracy_drop,
        'degradation_score': comparison.degradation_score,
        'is_degraded': comparison.is_degraded
    }
    
    # ========== STEP 5: Generate Alert ==========
    print("\n" + "="*64)
    print("STEP 5: Generating Alert and Recommendations")
    print("="*64)
    
    alert = monitor.generate_alert(drift_report, comparison)
    
    print(f"\n🚨 ALERT LEVEL: {alert.alert_level}")
    print(f"   Drift Detected: {alert.drift_detected}")
    print(f"   Performance Degraded: {alert.performance_degraded}")
    print(f"\n📋 RECOMMENDED ACTIONS:")
    for action in alert.recommended_actions:
        print(f"   {action}")
    
    results['alert'] = alert.to_dict()
    
    # ========== STEP 6: Generate Visualizations ==========
    print("\n" + "="*64)
    print("STEP 6: Generating Visualizations")
    print("="*64)
    
    artifacts = {}
    
    # Confusion matrices
    print("  • Generating confusion matrices...")
    artifacts['cm_baseline'] = monitor.plot_confusion_matrix(
        y_baseline, baseline_pred, 
        title="Baseline Data",
        save_path=monitor.reports_dir / f"cm_baseline_{drift_scenario}.png"
    )
    artifacts['cm_current'] = monitor.plot_confusion_matrix(
        y_monitoring, current_pred,
        title=f"Monitoring Data ({drift_scenario})",
        save_path=monitor.reports_dir / f"cm_current_{drift_scenario}.png"
    )
    
    # Metrics comparison
    print("  • Generating metrics comparison...")
    artifacts['metrics_comparison'] = monitor.plot_metric_comparison(
        baseline_metrics, current_metrics,
        save_path=monitor.reports_dir / f"metrics_{drift_scenario}.png"
    )
    
    # Drift heatmap
    print("  • Generating drift heatmap...")
    artifacts['drift_heatmap'] = monitor.plot_drift_heatmap(
        drift_report,
        save_path=monitor.reports_dir / f"drift_heatmap_{drift_scenario}.png"
    )
    
    print(f"✅ Generated {len(artifacts)} visualizations")
    results['artifacts'] = {k: str(v) for k, v in artifacts.items()}
    
    # ========== STEP 7: Log to MLflow ==========
    print("\n" + "="*64)
    print("STEP 7: Logging to MLflow")
    print("="*64)
    
    run_id = monitor.log_to_mlflow(
        drift_report=drift_report,
        performance_comparison=comparison,
        alert=alert,
        artifacts=artifacts,
        experiment_name="Model Monitoring - Data Drift",
        tags={
            'drift_scenario': drift_scenario,
            'drift_intensity': str(drift_intensity),
            'drift_type': drift_scenario
        }
    )
    
    results['mlflow_run_id'] = run_id
    print(f"✅ Results logged to MLflow (Run ID: {run_id})")
    
    # ========== STEP 8: Save Summary ==========
    print("\n" + "="*64)
    print("STEP 8: Saving Summary Report")
    print("="*64)
    
    summary_path = monitor.reports_dir / f"monitoring_summary_{drift_scenario}.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Summary saved to {summary_path}")
    
    # ========== Final Summary ==========
    print("\n" + "╔" + "="*62 + "╗")
    print("║" + " "*20 + "MONITORING COMPLETE" + " "*23 + "║")
    print("╚" + "="*62 + "╝")
    print(f"\n📊 Drift Score: {drift_report.drift_score:.3f} ({drift_report.alert_level})")
    print(f"📉 Performance Drop: {comparison.degradation_score:.3f} ({'DEGRADED' if comparison.is_degraded else 'STABLE'})")
    print(f"🚨 Alert Level: {alert.alert_level}")
    print(f"📁 Reports saved in: {monitor.reports_dir}")
    print(f"🔬 MLflow Run: {run_id}\n")
    
    return results


def main():
    """Command-line interface for monitoring pipeline."""
    parser = argparse.ArgumentParser(
        description="Data Drift Monitoring Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with mean shift drift
  python -m models.monitor_drift --scenario mean_shift --intensity 0.3

  # Run with seasonal drift
  python -m models.monitor_drift --scenario seasonal --intensity 0.5

  # Use existing drifted data
  python -m models.monitor_drift --scenario mean_shift --no-generate

Available scenarios: mean_shift, variance_shift, seasonal, feature_missing, label_drift, combined
        """
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='mean_shift',
        choices=['mean_shift', 'variance_shift', 'seasonal', 'feature_missing', 'label_drift', 'combined'],
        help='Type of drift scenario to simulate'
    )
    
    parser.add_argument(
        '--intensity',
        type=float,
        default=0.3,
        help='Intensity of drift (0-1, default: 0.3)'
    )
    
    parser.add_argument(
        '--no-generate',
        action='store_true',
        help='Skip data generation, use existing drifted data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/monitoring',
        help='Directory for monitoring data (default: data/monitoring)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_monitoring_pipeline(
        drift_scenario=args.scenario,
        drift_intensity=args.intensity,
        output_dir=Path(args.output_dir),
        generate_new_data=not args.no_generate
    )
    
    return results


if __name__ == "__main__":
    main()
