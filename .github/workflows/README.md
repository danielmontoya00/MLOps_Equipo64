# GitHub Actions - Data Drift Monitoring

Este directorio contiene workflows de GitHub Actions para automatizar el monitoreo de drift en el modelo de clasificación de obesidad.

## 📋 Workflows Disponibles

### 1. `drift-monitoring.yml` - Monitoreo Programado

**Propósito:** Ejecutar monitoreo de drift de forma automática y periódica.

**Triggers:**
- ⏰ **Schedule:** Cada domingo a medianoche UTC (configurable)
- 🖱️ **Manual:** Dispatch workflow desde GitHub UI

**Características:**
- Ejecuta múltiples escenarios de drift
- Genera reportes detallados con visualizaciones
- Sube artefactos (reportes, gráficos, datos)
- Detecta drift crítico y falla si es necesario
- Genera resumen en Markdown

**Inputs Manuales:**
```yaml
drift_scenario: mean_shift | variance_shift | seasonal | feature_missing | label_drift | combined
drift_intensity: 0.0 - 1.0 (default: 0.3)
run_all_scenarios: true | false
```

**Outputs:**
- `drift-monitoring-reports-{run_number}/`: Reportes completos
- `drift-visualizations-{run_number}/`: Gráficos PNG
- `monitoring_summary.md`: Resumen ejecutivo

### 2. `pr-drift-check.yml` - Validación en Pull Requests

**Propósito:** Validar que los cambios en PRs no introduzcan drift severo.

**Triggers:**
- Pull requests a `main` o `develop`
- Cambios en archivos críticos: models/, src/, data/, requirements.txt

**Características:**
- Ejecuta test rápido de drift (intensidad baja)
- Comenta en el PR con resultados
- Falla si detecta drift CRITICAL
- Genera reporte resumido

**Outputs:**
- Comentario automático en PR
- `pr-drift-check-{pr_number}/`: Artefactos del análisis

## 🚀 Uso

### Ejecución Manual del Workflow

1. Ve a la pestaña **Actions** en GitHub
2. Selecciona **"Model Drift Monitoring"**
3. Click en **"Run workflow"**
4. Configura parámetros:
   - Scenario: `mean_shift`
   - Intensity: `0.3`
   - Run all scenarios: ✓ (opcional)
5. Click **"Run workflow"**

### Ver Resultados

Los artefactos se generan automáticamente:

```bash
# Descargar artefactos
gh run download <run-id>

# O desde GitHub UI:
Actions → Workflow Run → Artifacts section
```

### Configurar Schedule

Edita `.github/workflows/drift-monitoring.yml`:

```yaml
on:
  schedule:
    # Diario a las 2 AM UTC
    - cron: '0 2 * * *'
    
    # Cada lunes y jueves a medianoche
    - cron: '0 0 * * 1,4'
    
    # Primer día de cada mes
    - cron: '0 0 1 * *'
```

## 📊 Interpretación de Resultados

### Alert Levels

| Nivel | Acción en Workflow | Significado |
|-------|-------------------|-------------|
| **NONE** | ✅ Pass | Sistema estable |
| **LOW** | ✅ Pass | Drift mínimo, continuar monitoreando |
| **MEDIUM** | ⚠️ Pass con warning | Planificar revisión |
| **HIGH** | ⚠️ Pass con warning | Acción requerida pronto |
| **CRITICAL** | ❌ Fail | Acción inmediata requerida |

### Estructura de Artefactos

```
drift-monitoring-reports-123/
├── reports/monitoring/
│   ├── alert.json                      # Alertas y recomendaciones
│   ├── drift_report.txt                # Reporte textual de drift
│   ├── performance_comparison.txt      # Comparación de métricas
│   ├── monitoring_summary_*.json       # Resúmenes por escenario
│   ├── cm_baseline_*.png              # Confusion matrices
│   ├── cm_current_*.png
│   ├── metrics_*.png                  # Comparación de métricas
│   └── drift_heatmap_*.png            # Magnitud de drift por feature
├── data/monitoring/
│   ├── X_*.csv                        # Datos con drift simulado
│   └── y_*.csv
└── monitoring_summary.md               # Resumen ejecutivo
```

## 🔧 Configuración Avanzada

### Notificaciones

Para habilitar notificaciones (email, Slack, Teams):

1. **Slack:**
```yaml
- name: 📢 Send Slack notification
  if: steps.drift_check.outputs.critical_drift == 'true'
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "🚨 Critical drift detected in ${{ github.repository }}",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "*Alert Level:* ${{ steps.drift_check.outputs.alert_level }}\n*Run:* <${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|View Details>"
            }
          }
        ]
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

2. **Email:**
```yaml
- name: 📧 Send email notification
  if: steps.drift_check.outputs.critical_drift == 'true'
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    server_port: 465
    username: ${{ secrets.EMAIL_USERNAME }}
    password: ${{ secrets.EMAIL_PASSWORD }}
    subject: "🚨 Critical Drift Detected"
    body: |
      Critical drift detected in model monitoring.
      
      Alert Level: ${{ steps.drift_check.outputs.alert_level }}
      Run: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
    to: team@example.com
    from: ML Monitoring System
```

### Integración con MLflow

Para subir métricas a MLflow remoto:

```yaml
- name: 📊 Upload to MLflow
  env:
    MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
    MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_USERNAME }}
    MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_PASSWORD }}
  run: |
    python -m models.monitor_drift --scenario $SCENARIO
```

### Almacenar Reportes en S3/GCS

```yaml
- name: 📦 Upload to S3
  uses: aws-actions/configure-aws-credentials@v2
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: us-east-1

- name: 🚀 Sync to S3
  run: |
    aws s3 sync reports/monitoring/ \
      s3://ml-monitoring-reports/drift-$(date +%Y%m%d)/
```

## 🎯 Casos de Uso

### 1. Monitoreo Continuo en Producción

```yaml
# .github/workflows/drift-monitoring.yml
on:
  schedule:
    - cron: '0 */6 * * *'  # Cada 6 horas
```

Útil para:
- Detectar drift en datos de producción
- Alertas tempranas de degradación
- Auditoría de calidad del modelo

### 2. Validación Pre-Deploy

```yaml
# En .github/workflows/pr-drift-check.yml
on:
  pull_request:
    branches: [main]
```

Útil para:
- Validar cambios antes de merge
- Prevenir introducción de drift
- Code review automatizado

### 3. Análisis de Tendencias

Ejecutar manualmente con diferentes intensidades:

```bash
# Via GitHub CLI
gh workflow run drift-monitoring.yml \
  -f drift_scenario=mean_shift \
  -f drift_intensity=0.1

gh workflow run drift-monitoring.yml \
  -f drift_scenario=mean_shift \
  -f drift_intensity=0.5

gh workflow run drift-monitoring.yml \
  -f drift_scenario=mean_shift \
  -f drift_intensity=0.9
```

Útil para:
- Estudiar sensibilidad del modelo
- Determinar umbrales óptimos
- Planificación de reentrenamiento

## 🐛 Troubleshooting

### Error: "No model found"

**Solución:** El workflow entrenará automáticamente un modelo si no existe.

### Error: "Module not found"

**Solución:** Verifica que `requirements.txt` esté actualizado:
```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements"
```

### Artefactos muy grandes

**Solución:** Reduce `retention-days` o excluye archivos pesados:
```yaml
- uses: actions/upload-artifact@v3
  with:
    retention-days: 7  # En vez de 30
    path: |
      reports/monitoring/*.png
      !reports/monitoring/data/
```

### Workflow muy lento

**Solución:** 
- Reduce número de escenarios en ejecuciones automáticas
- Usa intensidades más bajas para checks rápidos
- Ejecuta todos los escenarios solo en runs manuales

## 📚 Referencias

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Artifact Upload/Download](https://github.com/actions/upload-artifact)
- [Scheduling Workflows](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule)
- [Manual Workflow Dispatch](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#workflow_dispatch)

## 🤝 Contribuciones

Para agregar nuevos workflows:

1. Crear archivo en `.github/workflows/`
2. Seguir naming convention: `<feature>-<purpose>.yml`
3. Documentar en este README
4. Probar con `act` localmente (opcional)

```bash
# Instalar act
brew install act  # macOS
# o descargar desde https://github.com/nektos/act

# Probar workflow localmente
act -j drift-monitoring
```
