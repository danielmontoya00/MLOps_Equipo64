import os
import pickle
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

print("Ejecutando script de entrenamiento...")

# --- 1. Construir rutas absolutas dinámicamente ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Subir un nivel desde /models/
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Verificar rutas
print(f"📂 Leyendo datos desde: {DATA_DIR}")

# --- Configuración de MLflow --- 
# # Define el nombre del experimento. Si no existe, se crea. mlflow.set_experiment("Clasificación de Obesidad - RF")
# Define los parámetros del modelo para rastreo
N_ESTIMATORS = 100
RANDOM_STATE = 42

# Configurar el Mlflow
mlflow.set_experiment("Clasificación de Obesidad - RandomForest")

# Inicia un nuevo MLflow run (experimento)
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"🔍 MLflow Run ID: {run_id}")

    # --- 1. Cargar Datos de Entrenamiento ---
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()

    # --- 2. Registrar parámetros en MLflow ---
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("features", list(X_train.columns))

    # --- 3. Entrenar el modelo ---
    print("Entrenando el modelo RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    # --- 4. Registrar modelo en MLflow ---
    # Esto guarda el modelo en el artifacts store de MLflow 
    ## y lo hace accesible desde la GUI.
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="obesity_model",
        registered_model_name="RandomForestObesityModel"
    )

    # --- 5. Guardar el modelo localmente (Para ser usado por evaluate.py) --- 
    # Aunque MLflow lo guarda, lo mantenemos local para el pipeline actual de evaluación
    os.makedirs(MODELS_DIR, exist_ok=True)
    local_model_path = os.path.join(MODELS_DIR, f"obesity_classifier_{run_id}.pkl")

    with open(local_model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modelo local guardado en: {local_model_path}")

    # --- 6. Guardar el Run ID para evaluación posterior ---
    run_id_path = os.path.join(BASE_DIR, "current_run_id.txt")
    with open(run_id_path, "w") as f:
        f.write(run_id)

    print(f"Run ID guardado en: {run_id_path}")
    print("Script de entrenamiento finalizado y datos registrados en MLflow.")