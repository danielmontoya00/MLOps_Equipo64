# Data Drift Monitoring System

Sistema completo de monitoreo y detección de data drift para el modelo de clasificación de obesidad.

## 📋 Descripción

Este sistema permite:
- **Simular** diferentes escenarios de drift en los datos
- **Detectar** cambios en la distribución usando pruebas estadísticas
- **Evaluar** el impacto en el desempeño del modelo
- **Generar** alertas automáticas y recomendaciones
- **Registrar** resultados en MLflow para seguimiento

## 🏗️ Arquitectura

```
src/monitoring/
├── __init__.py
├── drift_simulator.py      # Simulación de drift
├── drift_detector.py        # Detección estadística
└── performance_monitor.py   # Monitoreo de performance

models/
└── monitor_drift.py         # Pipeline completo
```

## 🚀 Uso Rápido

### 1. Ejecutar Pipeline Completo

```bash
# Desde el directorio Obesity_Estimation/
python -m models.monitor_drift --scenario mean_shift --intensity 0.3
```

### 2. Escenarios Disponibles

- `mean_shift`: Desplazamiento de medias (covariate drift)
- `variance_shift`: Cambio en varianzas
- `seasonal`: Patrones estacionales
- `feature_missing`: Features faltantes
- `label_drift`: Concept drift (cambio en etiquetas)
- `combined`: Múltiples tipos de drift

### 3. Ejemplos de Uso

```bash
# Drift estacional moderado
python -m models.monitor_drift --scenario seasonal --intensity 0.5

# Drift severo en features
python -m models.monitor_drift --scenario combined --intensity 0.7

# Usar datos previamente generados
python -m models.monitor_drift --scenario mean_shift --no-generate
```

## 📊 Métricas de Drift

### Population Stability Index (PSI)
- **PSI < 0.1**: Sin cambio significativo
- **0.1 ≤ PSI < 0.2**: Cambio pequeño
- **PSI ≥ 0.2**: Drift significativo detectado

### Kolmogorov-Smirnov Test
- **p-value < 0.05**: Distribuciones diferentes (drift detectado)
- **p-value ≥ 0.05**: No hay evidencia de drift

### Jensen-Shannon Divergence
- **JS < 0.1**: Distribuciones similares
- **JS ≥ 0.1**: Drift detectado

## 🚨 Niveles de Alerta

| Nivel | Drift Score | Acción Recomendada |
|-------|-------------|-------------------|
| **CRITICAL** | ≥ 0.7 | Reentrenamiento urgente, considerar rollback |
| **HIGH** | ≥ 0.5 | Reentrenar en 24-48 horas |
| **MEDIUM** | ≥ 0.3 | Planificar reentrenamiento próximo sprint |
| **LOW** | ≥ 0.1 | Continuar monitoreando |
| **NONE** | < 0.1 | Sistema estable |

## 📈 Visualizaciones Generadas

El sistema genera automáticamente:

1. **Confusion Matrices**: Baseline vs datos con drift
2. **Comparación de Métricas**: Barras comparativas de accuracy, precision, recall, F1
3. **Heatmap de Drift**: Magnitud de drift por feature
4. **Reportes en Texto**: Resúmenes detallados

Todas las visualizaciones se guardan en `reports/monitoring/`

## 🔬 Integración con MLflow

Cada ejecución registra en MLflow:

- **Métricas de Drift**: drift_score, n_drifted_features, etc.
- **Métricas de Performance**: accuracy, F1, precision, recall
- **Degradación**: accuracy_drop, f1_drop, degradation_score
- **Artefactos**: Gráficos, reportes, alertas
- **Tags**: drift_scenario, drift_intensity, drift_type

### Ver resultados en MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Abrir http://localhost:5000
# Buscar experimento "Model Monitoring - Data Drift"
```

## 🔧 Uso Programático

### Simular Drift

```python
from src.monitoring.drift_simulator import DriftSimulator, DriftConfig, DriftType

config = DriftConfig(
    drift_type=DriftType.MEAN_SHIFT,
    intensity=0.3,
    random_state=42
)

simulator = DriftSimulator(config)
drift_info = simulator.generate_drifted_dataset(suffix="custom_drift")
```

### Detectar Drift

```python
from src.monitoring.drift_detector import DriftDetector
import pandas as pd

detector = DriftDetector()

X_baseline = pd.read_csv("data/processed/X_test.csv")
X_current = pd.read_csv("data/monitoring/X_custom_drift.csv")

report = detector.detect_dataset_drift(X_baseline, X_current)
print(report.summary())
```

### Monitorear Performance

```python
from src.monitoring.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
model = monitor.load_model()

baseline_metrics, _ = monitor.evaluate_on_dataset(model, X_baseline, y_baseline)
current_metrics, _ = monitor.evaluate_on_dataset(model, X_current, y_current)

comparison = monitor.compare_performance(baseline_metrics, current_metrics)
print(comparison.summary())
```

## 📝 Configuración de Umbrales

Puedes ajustar los umbrales de detección:

```python
from src.monitoring.drift_detector import DriftThresholds

thresholds = DriftThresholds(
    ks_pvalue=0.05,        # Kolmogorov-Smirnov
    psi_threshold=0.2,     # Population Stability Index
    chi2_pvalue=0.05,      # Chi-cuadrado
    js_divergence=0.1      # Jensen-Shannon
)

detector = DriftDetector(thresholds=thresholds)
```

## 🎯 Casos de Uso

### 1. Monitoreo Continuo en Producción

```bash
# Ejecutar diariamente con datos nuevos
python -m models.monitor_drift \
  --scenario mean_shift \
  --output-dir data/monitoring/$(date +%Y%m%d)
```

### 2. Validación de Modelo Antes de Deploy

```bash
# Simular diferentes escenarios
for scenario in mean_shift seasonal combined; do
  python -m models.monitor_drift --scenario $scenario --intensity 0.4
done
```

### 3. Análisis de Degradación

```python
# Incrementar intensidad gradualmente
for intensity in [0.1, 0.3, 0.5, 0.7]:
    run_monitoring_pipeline(
        drift_scenario="mean_shift",
        drift_intensity=intensity
    )
```

## 🔍 Interpretación de Resultados

### Ejemplo de Output

```
╔══════════════════════════════════════════════════════════════╗
║                    DRIFT DETECTION REPORT                    ║
╚══════════════════════════════════════════════════════════════╝

Overall Drift Score: 0.425
Alert Level: MEDIUM

Features Analyzed: 16
Features with Drift: 6 (37.5%)

⚠️  Drifted Features:
  • Weight: p-value=0.0012, PSI=0.345, magnitude=0.567
  • Height: p-value=0.0089, PSI=0.256, magnitude=0.421
  ...
```

**Interpretación**:
- 37.5% de features muestran drift
- Alert Level MEDIUM → planificar reentrenamiento
- Weight y Height son las features más afectadas

## 🛠️ Troubleshooting

### Error: "Run ID file not found"

```bash
# Entrenar modelo primero
cd Obesity_Estimation
python models/train_model.py
```

### Error: "Drifted data not found"

```bash
# Generar datos primero
python -m models.monitor_drift --scenario mean_shift
```

### Visualizaciones no se guardan

```bash
# Crear directorio manualmente
mkdir -p reports/monitoring
```

## 📚 Referencias

- [Population Stability Index](https://www.listendata.com/2015/05/population-stability-index.html)
- [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- [Jensen-Shannon Divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)

## 🤝 Contribuciones

Para agregar nuevos tipos de drift:

1. Añadir tipo en `DriftType` enum
2. Implementar método `simulate_<tipo>_drift()` en `DriftSimulator`
3. Agregar caso en método `apply_drift()`

## 📄 Licencia

Parte del proyecto MLOps_Equipo64
