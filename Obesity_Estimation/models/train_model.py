import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

print("Ejecutando script de entrenamiento...")

# --- 1. Cargar Datos de Entrenamiento ---
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel() # .ravel() es importante para el formato

# --- 2. Definir y Entrenar el Modelo ---
print("Entrenando el modelo RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Entrenamiento completado.")

# --- 3. Guardar el Modelo Entrenado ---
models_path = 'models'
os.makedirs(models_path, exist_ok=True) # Crea la carpeta si no existe

model_filepath = os.path.join(models_path, 'obesity_classifier_v1.pkl')
with open(model_filepath, 'wb') as f:
    pickle.dump(model, f)

print(f"Modelo guardado exitosamente en: {model_filepath}")