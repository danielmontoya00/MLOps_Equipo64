import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


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

# ========== ADICIONES COMENZANDO AQUÍ ==========

# --- 5. Cargar datos de entrenamiento y validación ---
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()

X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()

# --- 6. Escalamiento de variables numéricas ---
scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=np.number).columns

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# --- 7. PCA para visualización 2D ---
# Codificar la variable objetivo
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
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Validación
y_val_pred_rf = rf.predict(X_val)
print("Random Forest - Validación Accuracy:", accuracy_score(y_val, y_val_pred_rf))

# Prueba
y_test_pred_rf = rf.predict(X_test)
print("Random Forest - Test Accuracy:", accuracy_score(y_test, y_test_pred_rf))

# --- 9. MODELO 2: SVM ---
svm = SVC(random_state=42)
svm.fit(X_train, y_train)

# Validación
y_val_pred_svm = svm.predict(X_val)
print("SVM - Validación Accuracy:", accuracy_score(y_val, y_val_pred_svm))

# Prueba
y_test_pred_svm = svm.predict(X_test)
print("SVM - Test Accuracy:", accuracy_score(y_test, y_test_pred_svm))

# --- 10. Matrices de Confusión ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 fila, 2 columnas

ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred_rf,
    display_labels=le.classes_,
    cmap='Blues',
    ax=axes[0],
    xticks_rotation=45
)
axes[0].set_title("Matriz de Confusión - Random Forest")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred_svm,
    display_labels=le.classes_,
    cmap='Greens',
    ax=axes[1],
    xticks_rotation=45
)
axes[1].set_title("Matriz de Confusión - SVM")

plt.tight_layout()
plt.show()

# --- 11. Guardar métricas de ambos modelos ---
metrics_rf_path = os.path.join(reports_path, 'metrics_rf.txt')
with open(metrics_rf_path, 'w') as f:
    f.write(f"Random Forest - Test Accuracy: {accuracy_score(y_test, y_test_pred_rf):.4f}\n")
    f.write("\n--- Reporte de Clasificación ---\n")
    f.write(classification_report(y_test, y_test_pred_rf))

metrics_svm_path = os.path.join(reports_path, 'metrics_svm.txt')
with open(metrics_svm_path, 'w') as f:
    f.write(f"SVM - Test Accuracy: {accuracy_score(y_test, y_test_pred_svm):.4f}\n")
    f.write("\n--- Reporte de Clasificación ---\n")
    f.write(classification_report(y_test, y_test_pred_svm))

print("✅ Métricas adicionales guardadas en carpeta 'reports'")