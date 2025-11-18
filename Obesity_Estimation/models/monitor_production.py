"""
Script para Monitoreo de Drift con Datos Reales de Producción

Este script compara datos de entrenamiento (baseline) vs datos nuevos
de producción para detectar drift REAL (no simulado).
"""

import pandas as pd
from pathlib import Path
from src.monitoring.drift_detector import DriftDetector, DriftThresholds
from src.monitoring.performance_monitor import PerformanceMonitor


def monitor_production_drift(
    baseline_data_path: str = "data/processed/X_test.csv",
    production_data_path: str = "data/production/X_new.csv",  # TUS DATOS REALES
    production_labels_path: str = "data/production/y_new.csv"  # Si los tienes
):
    """
    Monitorea drift usando datos REALES de producción (no simulados).
    
    Args:
        baseline_data_path: Datos de referencia (training/test set)
        production_data_path: Datos nuevos de producción
        production_labels_path: Labels de producción (opcional)
    """
    print("=" * 70)
    print("MONITOREO DE DRIFT CON DATOS REALES DE PRODUCCIÓN")
    print("=" * 70)
    
    # 1. Cargar datos baseline
    print("\n1️⃣ Cargando datos baseline...")
    X_baseline = pd.read_csv(baseline_data_path)
    print(f"   ✅ Baseline: {X_baseline.shape[0]} muestras")
    
    # 2. Cargar datos de producción
    print("\n2️⃣ Cargando datos de producción...")
    if not Path(production_data_path).exists():
        print(f"   ❌ ERROR: No se encontraron datos de producción en {production_data_path}")
        print("\n   Para usar este script:")
        print("   1. Exporta datos nuevos de producción a CSV")
        print("   2. Guárdalos en data/production/X_new.csv")
        print("   3. Ejecuta este script nuevamente")
        return
    
    X_production = pd.read_csv(production_data_path)
    print(f"   ✅ Producción: {X_production.shape[0]} muestras")
    
    # 3. Detectar drift
    print("\n3️⃣ Detectando drift...")
    detector = DriftDetector(
        thresholds=DriftThresholds(
            ks_pvalue=0.05,
            psi_threshold=0.2,
            js_divergence=0.1
        )
    )
    
    drift_report = detector.detect_dataset_drift(X_baseline, X_production)
    
    # 4. Mostrar resultados
    print("\n" + "=" * 70)
    print("RESULTADOS DE DRIFT REAL")
    print("=" * 70)
    print(drift_report.summary())
    
    # 5. Interpretar resultados
    print("\n" + "=" * 70)
    print("INTERPRETACIÓN")
    print("=" * 70)
    
    if drift_report.drift_score < 0.1:
        print("✅ NO HAY DRIFT significativo")
        print("   Los datos de producción son similares al baseline")
        print("   El modelo debería funcionar bien")
    
    elif drift_report.drift_score < 0.3:
        print("🟢 DRIFT BAJO detectado")
        print("   Hay algunos cambios pero son manejables")
        print("   Continúa monitoreando")
    
    elif drift_report.drift_score < 0.5:
        print("🟡 DRIFT MEDIO detectado")
        print("   Los datos están cambiando significativamente")
        print("   Considera reentrenar el modelo pronto")
    
    elif drift_report.drift_score < 0.7:
        print("🟠 DRIFT ALTO detectado")
        print("   Cambios importantes en la distribución")
        print("   ACCIÓN REQUERIDA: Planifica reentrenamiento")
    
    else:
        print("🔴 DRIFT CRÍTICO detectado")
        print("   Los datos son muy diferentes al baseline")
        print("   URGENTE: Reentrenar modelo inmediatamente")
    
    # 6. Evaluar performance si tenemos labels
    if Path(production_labels_path).exists():
        print("\n4️⃣ Evaluando impacto en performance...")
        monitor = PerformanceMonitor()
        model = monitor.load_model()
        
        y_baseline = pd.read_csv("data/processed/y_test.csv").values.ravel()
        y_production = pd.read_csv(production_labels_path).values.ravel()
        
        baseline_metrics, _ = monitor.evaluate_on_dataset(model, X_baseline, y_baseline)
        production_metrics, _ = monitor.evaluate_on_dataset(model, X_production, y_production)
        
        comparison = monitor.compare_performance(baseline_metrics, production_metrics)
        print(comparison.summary())
    
    # 7. Guardar resultados
    output_dir = Path("reports/production_monitoring")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "drift_report_real.txt", "w") as f:
        f.write(drift_report.summary())
    
    print(f"\n📁 Reporte guardado en: {output_dir / 'drift_report_real.txt'}")


if __name__ == "__main__":
    monitor_production_drift()
