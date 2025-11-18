import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score


def test_differential_performance():
    """
    Compara el modelo nuevo vs un baseline guardado.
    Requiere:
    - models/baseline_model.pkl  (modelo referencia)
    - models/model_info.json     (modelo actual)
    """

    # Cargar baseline
    baseline = joblib.load("models/baseline_model.pkl")

    # Cargar modelo nuevo desde su ruta MLflow / artefactos
    with open("models/model_info.json", "r") as f:
        info = json.load(f)

    model_path = f"models/{info['registered_model_name']}.pkl"
    new_model = joblib.load(model_path)

    # Cargar dataset test
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # Predicciones
    acc_baseline = accuracy_score(y_test, baseline.predict(X_test))
    acc_new = accuracy_score(y_test, new_model.predict(X_test))

    # Validación: el modelo nuevo NO debe degradarse > 2%
    assert acc_new >= acc_baseline - 0.02, (
        f"Regresión: acc old={acc_baseline:.3f} acc new={acc_new:.3f}"
    )
