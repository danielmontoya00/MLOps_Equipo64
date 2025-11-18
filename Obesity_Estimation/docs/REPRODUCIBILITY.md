# Reproducibilidad y Semillas Aleatorias

Este proyecto está configurado para garantizar la reproducibilidad de todos los experimentos y resultados.

## Configuración de Semillas Aleatorias

### Módulo Central: `src/utils/seed.py`

El proyecto incluye un módulo centralizado para configurar todas las semillas aleatorias:

```python
from src.utils.seed import set_seed

# Establecer semilla global
set_seed(42)
```

Esta función configura:
- **Python's random**: `random.seed()`
- **NumPy**: `np.random.seed()`
- **PYTHONHASHSEED**: Variable de entorno para hash randomization
- **Scikit-learn**: A través de NumPy y parámetros `random_state`

### Scripts con Semillas Configuradas

Los siguientes scripts ya tienen semillas aleatorias configuradas:

1. **notebooks/make_dataset.py**
   - Semilla global al inicio del script
   - `train_test_split(..., random_state=42)`

2. **models/train_model.py**
   - Semilla global al inicio del script
   - `RandomForestClassifier(..., random_state=42)`

3. **models/evaluate_model.py**
   - Carga modelos entrenados con semillas fijas

4. **notebooks/eda.py**
   - Semilla global para visualizaciones consistentes

5. **src/monitoring/drift_simulator.py**
   - Semilla configurable vía `DriftConfig.random_state`
   - Semilla global al inicializar el simulador

6. **Module/run_pipeline.py**
   - Semilla global al inicio del pipeline

## Variable de Entorno

Puedes configurar la semilla globalmente usando la variable de entorno `RANDOM_SEED`:

```bash
export RANDOM_SEED=42
```

O en tu archivo `.env`:

```env
RANDOM_SEED=42
```

El valor por defecto es `42` si no se especifica ninguna variable de entorno.

## Garantías de Reproducibilidad

### Entrenamiento de Modelos
- Cada ejecución de `train_model.py` producirá exactamente el mismo modelo
- Los splits de datos serán idénticos
- Las inicializaciones del Random Forest serán determinísticas

### Simulación de Drift
- Los datos sintéticos generados serán idénticos entre ejecuciones
- Configurar `DriftConfig.random_state` para controlar la aleatoriedad

### Visualizaciones
- Las figuras generadas serán consistentes
- Los colores y disposiciones se mantendrán iguales

## Buenas Prácticas

1. **Siempre usar `set_seed()` al inicio** de scripts que involucren operaciones aleatorias

2. **Pasar `random_state`** a funciones de scikit-learn:
   ```python
   train_test_split(X, y, test_size=0.2, random_state=42)
   RandomForestClassifier(n_estimators=100, random_state=42)
   ```

3. **Documentar la semilla** usada en experimentos de MLflow

4. **No modificar la semilla** sin documentar el cambio y su razón

5. **Para experimentación**: Si necesitas variar resultados, usa diferentes valores de semilla de manera controlada

## Verificación

Para verificar que las semillas están funcionando correctamente:

```bash
# Ejecutar el pipeline dos veces
python Module/run_pipeline.py
python Module/run_pipeline.py

# Los modelos resultantes deben ser idénticos
# Comparar los archivos de métricas en reports/
```

## Notas Técnicas

- **PYTHONHASHSEED**: Necesario para reproducibilidad completa en diccionarios de Python < 3.7
- **NumPy legacy**: Usando `np.random.seed()` para compatibilidad. Para código nuevo, considerar usar `np.random.default_rng(seed)`
- **Multithreading**: `n_jobs=-1` en RandomForest puede introducir no-determinismo en algunas plataformas. Para máxima reproducibilidad, usar `n_jobs=1`

## Referencias

- [Scikit-learn: Controlling randomness](https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness)
- [NumPy Random Generator](https://numpy.org/doc/stable/reference/random/generator.html)
- [MLflow: Reproducible Runs](https://mlflow.org/docs/latest/tracking.html)
