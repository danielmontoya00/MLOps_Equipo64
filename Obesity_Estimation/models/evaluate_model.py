import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


print("Ejecutando script de evaluación...")

# --- Rutas ---
# --- Definir rutas dinámicas ---
CURRENT_DIR = os.path.dirname(__file__)  # /models/
BASE_DIR = os.path.dirname(CURRENT_DIR)  # /Obesity_Estimation/

DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'obesity_classifier_v1.pkl')
REPORTS_PATH = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS_PATH, exist_ok=True)

# --- 1. Cargar Modelo y Datos de Prueba ---
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

X_test = pd.read_csv(os.path.join(DATA_PATH, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(DATA_PATH, 'y_test.csv')).values.ravel()

# --- 2. Realizar Predicciones ---

print("Evaluando modelo base (RandomForest entrenado)...")
predictions = model.predict(X_test)

# --- 3. Evaluar Rendimiento ---
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Accuracy del modelo: {accuracy:.4f}")
print("\n--- Reporte de Clasificación ---")
print(report)

# --- 4. Guardar las Métricas ---
os.makedirs(REPORTS_PATH, exist_ok=True) # Crea la carpeta si no existe

metrics_filepath = os.path.join(REPORTS_PATH, 'metrics.txt')
with open(metrics_filepath, 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write("\n--- Reporte de Clasificación ---\n")
    f.write(report)

print(f"Métricas guardadas en: {metrics_filepath}")

# ========== ADICIONES COMENZANDO AQUÍ ==========

# --- 5. Cargar datos de entrenamiento y validación ---
X_train = pd.read_csv(os.path.join(DATA_PATH, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(DATA_PATH, 'y_train.csv')).values.ravel()

X_val = pd.read_csv(os.path.join(DATA_PATH, 'X_val.csv'))
y_val = pd.read_csv(os.path.join(DATA_PATH, 'y_val.csv')).values.ravel()

# --- 6. Escalamiento de variables numéricas ---
print(" Escalando variables numéricas...")
scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=np.number).columns

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# --- 7. PCA para visualización 2D ---
# Codificar la variable objetivo
print("Generando visualización PCA...")
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train_encoded, cmap='tab10', alpha=0.6)
plt.title("Reducción de dimensionalidad con PCA")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.colorbar(scatter, label='Clase (Obesidad)')

# Ajustar límites del eje X para mejor visualización
x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
x_range = x_max - x_min
plt.xlim(x_min - 0.1*x_range, x_max + 0.1*x_range)

plt.tight_layout()
plt.show()

# --- 8. MODELO 1: Random Forest ---

print("Entrenando y evaluando modelos adicionales...")

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

for name, clf in models.items():
    print(f"\nEntrenando modelo: {name}")
    clf.fit(X_train, y_train)

    # Validación
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    # Prueba
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"{name} - Validación Accuracy: {val_acc:.4f}")
    print(f"{name} - Test Accuracy: {test_acc:.4f}")

    # Guardar métricas individuales
    metrics_path = os.path.join(REPORTS_PATH, f'metrics_{name.replace(" ", "_").lower()}.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"{name} - Validación Accuracy: {val_acc:.4f}\n")
        f.write(f"{name} - Test Accuracy: {test_acc:.4f}\n\n")
        f.write("--- Reporte de Clasificación ---\n")
        f.write(classification_report(y_test, y_test_pred))


# --- 10. Matrices de Confusión ---
print("Generando matrices de confusión...")
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test, models["Random Forest"].predict(X_test),
    display_labels=le.classes_,
    cmap='Blues',
    ax=axes[0],
    xticks_rotation=45
)
axes[0].set_title("Matriz de Confusión - Random Forest")

ConfusionMatrixDisplay.from_predictions(
    y_test, models["SVM"].predict(X_test),
    display_labels=le.classes_,
    cmap='Greens',
    ax=axes[1],
    xticks_rotation=45
)
axes[1].set_title("Matriz de Confusión - SVM")

plt.tight_layout()
plt.show()
