import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score
import os

print("Ejecutando script de evaluación...")

# --- 1. Cargar Modelo y Datos de Prueba ---
model_filepath = 'models/obesity_classifier_v1.pkl'
with open(model_filepath, 'rb') as f:
    model = pickle.load(f)

X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# --- 2. Realizar Predicciones ---
predictions = model.predict(X_test)

# --- 3. Evaluar Rendimiento ---
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Accuracy del modelo: {accuracy:.4f}")
print("\n--- Reporte de Clasificación ---")
print(report)

# --- 4. Guardar las Métricas ---
reports_path = 'reports'
os.makedirs(reports_path, exist_ok=True) # Crea la carpeta si no existe

metrics_filepath = os.path.join(reports_path, 'metrics.txt')
with open(metrics_filepath, 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write("\n--- Reporte de Clasificación ---\n")
    f.write(report)

print(f"Métricas guardadas en: {metrics_filepath}")