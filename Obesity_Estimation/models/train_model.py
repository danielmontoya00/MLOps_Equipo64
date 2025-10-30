import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import mlflow
import mlflow.sklearn
from datetime import datetime

# --- Configuración de MLflow ---
# Define el nombre del experimento. Si no existe, se crea.
mlflow.set_experiment("Clasificación de Obesidad - RF")

# Define los parámetros del modelo para rastreo
N_ESTIMATORS = 50
RANDOM_STATE = 42

print("Ejecutando script de entrenamiento...")

# Inicia un nuevo MLflow run (experimento)
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    
    # --- 1. Cargar Datos de Entrenamiento ---
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()

    # --- 2. Registrar Hiperparámetros en MLflow ---
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("model_type", "RandomForestClassifier")
    
    # --- 3. Definir y Entrenar el Modelo ---
    print("Entrenando el modelo RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, 
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    # --- 4. Registrar el Modelo en MLflow ---
    # Esto guarda el modelo en el artifacts store de MLflow
    # y lo hace accesible desde la GUI.
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="obesity_model",
        registered_model_name="RandomForestObesityModel"
    )
    
    # --- 5. Guardar el modelo localmente (Para ser usado por evaluate.py) ---
    # Aunque MLflow lo guarda, lo mantenemos local para el pipeline actual de evaluación
    models_path = 'models'
    os.makedirs(models_path, exist_ok=True)
    model_filepath = os.path.join(models_path, f'obesity_classifier_{run_id}.pkl')
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"Modelo local guardado con Run ID en el nombre: {model_filepath}")
    
    # Guardamos el Run ID para que evaluate.py sepa dónde logear las métricas
    with open('current_run_id.txt', 'w') as f:
        f.write(run_id)

print("Script de entrenamiento finalizado y datos registrados en MLflow.")
