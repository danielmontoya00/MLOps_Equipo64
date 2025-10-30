import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score
import os
import mlflow

print("Ejecutando script de evaluación...")

# --- 1. Obtener el MLflow Run ID del entrenamiento anterior ---
try:
    with open('current_run_id.txt', 'r') as f:
        run_id = f.read().strip()
    print(f"Adjuntando evaluación al MLflow Run ID: {run_id}")
except FileNotFoundError:
    print("Error: No se encontró 'current_run_id.txt'. Debe ejecutar train.py primero.")
    exit()

# Reanuda el run existente para agregar las métricas
with mlflow.start_run(run_id=run_id, nested=True): 
    
    # --- 2. Cargar Modelo y Datos de Prueba ---
    # Busca el modelo guardado localmente con el Run ID
    model_filepath = f'models/obesity_classifier_{run_id}.pkl'
    try:
        with open(model_filepath, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el modelo en {model_filepath}")
        exit()

    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

    # --- 3. Realizar Predicciones y Evaluar Rendimiento ---
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    report_text = classification_report(y_test, predictions)

    print(f"Accuracy del modelo: {accuracy:.4f}")
    
    # --- 4. Registrar Métricas en MLflow ---
    mlflow.log_metric("accuracy", accuracy)
    
    # Registrar métricas detalladas (Ejemplo: F1-score de una clase específica)
    # Reemplaza 'nombre_clase' por una de tus clases reales si usas 'output_dict=True'
    # Ejemplo:
    # if 'Overweight' in report:
    #     mlflow.log_metric("f1_overweight", report['Overweight']['f1-score'])
    
    # --- 5. Guardar el Reporte Completo como Artifact de MLflow ---
    reports_path = 'reports'
    os.makedirs(reports_path, exist_ok=True)
    metrics_filepath = os.path.join(reports_path, 'metrics.txt')
    
    with open(metrics_filepath, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\n--- Reporte de Clasificación ---\n")
        f.write(report_text)

    # Loguea el archivo de reporte como un artefacto
    mlflow.log_artifact(metrics_filepath)

    print(f"Métricas y reporte guardados en MLflow y en {metrics_filepath}")

print("Script de evaluación finalizado.")
