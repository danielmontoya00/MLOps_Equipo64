# src/test/iunit/test_model_utils.py
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def test_model_training_and_prediction(tmp_path):
    # Datos falsos
    X = pd.DataFrame({
        "age": [20, 30, 25, 40],
        "weight": [70, 80, 65, 90]
    })
    y = [0, 1, 0, 1]

    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Prueba de predicción
    preds = model.predict(X)
    assert len(preds) == len(y)

    # Guardar modelo en carpeta temporal
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    assert model_path.exists(), "El archivo del modelo no fue guardado"

    # Recargar modelo
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)

    preds2 = loaded.predict(X)
    assert (preds == preds2).all()
