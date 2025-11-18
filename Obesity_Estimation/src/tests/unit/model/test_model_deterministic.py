import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Apunta directamente a tu archivo .pkl
MODEL_PATH = Path(__file__).resolve().parents[5] / "Obesity_Estimation" / "models" / "obesity_classifier_v1.pkl"

def load_model():
    # Verifica que el archivo exista
    assert MODEL_PATH.exists(), f"No se encontró el modelo en {MODEL_PATH}"

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def test_model_is_deterministic():
    """
    Con el mismo input, el modelo debe producir siempre el mismo output.
    """

    model = load_model()

    # Input sintético estable
    X = pd.DataFrame({
    "Age": [25, 40, 30],
    "Gender": [1, 0, 1],
    "Height": [1.70, 1.80, 1.65],
    "Weight": [70, 85, 60],
    "FCVC": [2, 1, 3],
    "NCP": [2, 2, 1],
    "CAEC": [0, 1, 0],
    "SMOKE": [0, 1, 0],
    "CH2O": [2, 3, 1],
    "SCC": [0, 1, 0],
    "FAF": [1, 0, 2],
    "TUE": [0, 1, 0],
    "CALC": [1, 0, 1],
    "MTRANS": [1, 2, 0],
    "FAVC": [0, 1, 1],
    "family_history_with_overweight": [1, 0, 1]  # <-- obligatorio
    })
    
    # Reordenar columnas según como el modelo fue entrenado
    X = X[model.feature_names_in_]

    preds1 = model.predict(X)
    preds2 = model.predict(X)

    # Debug print opcional
    # print(preds1, preds2)

    assert np.array_equal(preds1, preds2), (
        "El modelo NO es determinístico: produce diferentes predicciones."
    )
