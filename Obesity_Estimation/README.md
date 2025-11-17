# Proyecto de Estimación de Obesidad (MLOps)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Este repositorio contiene nuestro proyecto end-to-end para la materia de Operaciones de aprendizaje automático. 

El objetivo es predecir los niveles de obesidad basándose en factores de estilo de vida, pero el enfoque principal es demostrar un flujo de trabajo de MLOps robusto, reproducible y listo para el despliegue.

Este proyecto demuestra los siguientes conceptos clave de MLOps:
* **Seguimiento de Experimentos:** Uso de **MLflow** para registrar ejecuciones de entrenamiento, métricas y parámetros.
* **Versionado de Artefactos:** Almacenamiento y versionado de modelos usando el **Registro de Modelos de MLflow**.
* **Versionado de Datos:** Uso de **DVC** para gestionar datasets sin sobrecargar el repositorio de Git.
* **Despliegue como Servicio:** Exposición del modelo entrenado a través de una API RESTful usando **FastAPI**.
* **Reproducibilidad:** Garantizar que el entorno (con `requirements.txt`) y los scripts (con `random_state`) produzcan resultados consistentes.

---

## 🛠️ Tech Stack (Tecnologías Utilizadas)

* **Python 3.10+**
* **Scikit-learn:** Para el entrenamiento del modelo (Random Forest).
* **MLflow:** Para el seguimiento de experimentos y registro de modelos.
* **DVC:** Para el versionado de datos.
* **FastAPI:** Para crear el endpoint de la API.
* **Uvicorn:** Como servidor ASGI para FastAPI.
* **Pydantic:** Para la validación de datos en la API.
* **Pathlib:** Para un manejo robusto de las rutas de archivos.

--- 

## 📁 Estructura del Proyecto

La estructura del proyecto sigue una lógica modular para separar las responsabilidades.

```
MLOps_Equipo64/
└── Obesity_Estimation/
    ├── data/
    │   ├── external        <- Data from third party sources.
    │   ├── interim         <- Intermediate data that has been transformed.
    │   ├── processed       <- The final, canonical data sets for modeling.
    │   └── raw             <- The original, immutable data dump.
    ├── docs                <- A default mkdocs project; see www.mkdocs.org for details
    ├── mlruns              <- Created by MLflow, ignored by Git
    ├── models              <- Trained and serialized models, model predictions, or model summaries  
    │   ├── evaluate_model.py   
    │   ├── train_model.py
    │   ├── current_run_id.txt      <- Temporal file (Ignored by Git)
    │   ├── model_info.json         <- Model API pointer from MLFLOW
    │   └── run_pipeline.py
    ├── Module/             <- Source code for use in this project.
    │   └── run_pipeline.py
    ├── notebooks/
    │   ├── 1.0-eda-analisis_exploratorio.ipynb
    │   ├── eda.py
    │   └── make_dataset.py
    ├── references          <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
    ├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
                               generated with `pip freeze > requirements.txt`
    ├── src/
    │   ├── api/
    │   │   └── app.py # API Script (FastAPI) 
    │   ├── data/
    │   │   └── load_data.py
    │   ├── features/
    │   │   ├── clean_data.py
    │   │   └── create_feature.py
    │   └── visualization/
    │       └── visualize.py
    └── tests/
        └── test_data.py
```
---

## 🚀 Cómo Empezar

Sigue estos pasos para configurar y ejecutar el proyecto en un entorno local o un Codespace.

Todos los comandos deben ejecutarse desde el directorio raíz del proyecto (`MLOPS_Equipo64/Obesity_Estimation/`).

### 1. Prerrequisitos

* Git
* Python 3.10 o superior
* (Opcional) GitHub Codespaces para un entorno limpio instantáneo.

### 2. Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone <https://github.com/danielmontoya00/MLOps_Equipo64.git>
    cd MLOPS_Equipo64
    ```

2.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Descarga los datos versionados:**
    *(Este paso asume que `data/processed` está configurado con DVC)*
    ```bash
    dvc pull
    ```

---
