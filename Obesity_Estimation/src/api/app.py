from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os, json
import pandas as pd
import mlflow.sklearn

app = FastAPI(title="Obesity Classifier API", version="1.0.0")

# ---------- Configuración ----------
# Prioridad:
# 1) MLFLOW_MODEL_URI (ej. "models:/obesity_classifier/Production" o "runs:/<run_id>/model")
# 2) models/model_info.json -> "model_uri"
# Si nada existe, la API arranca pero /predict devolverá error claro.
MLFLOW_MODEL_URI_ENV = os.getenv("MLFLOW_MODEL_URI")
MODEL_URI_CACHE: str | None = None
MODEL = None  # se inicializa on-demand


class Payload(BaseModel):
    data: list[dict] = Field(..., description="Lista de filas con las mismas columnas que X_test.csv (Modo rápido)")


def _read_model_info_json() -> str | None:
    path = "models/model_info.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
        return info.get("model_uri")
    return None


def _resolve_model_uri() -> str | None:
    # 1) ENV
    if MLFLOW_MODEL_URI_ENV:
        return MLFLOW_MODEL_URI_ENV
    # 2) JSON
    return _read_model_info_json()


def _ensure_model_loaded():
    """Carga el modelo una sola vez, justo antes de predecir."""
    global MODEL, MODEL_URI_CACHE
    if MODEL is not None:
        return
    uri = _resolve_model_uri()
    if not uri:
        raise FileNotFoundError(
            "No hay MODEL_URI disponible. Define MLFLOW_MODEL_URI o ejecuta primero 'python train.py' "
            "para generar models/model_info.json."
        )
    MODEL = mlflow.sklearn.load_model(uri)
    MODEL_URI_CACHE = uri


@app.get("/")
def health():
    """Muestra estado y la fuente del modelo si está disponible."""
    status = "ready" if MODEL is not None or _resolve_model_uri() else "model_missing"
    return {
        "status": status,
        "model_uri_env": MLFLOW_MODEL_URI_ENV,
        "model_uri_json": _read_model_info_json(),
        "active_model_uri": MODEL_URI_CACHE,
    }


@app.post("/predict")
def predict(payload: Payload):
    try:
        _ensure_model_loaded()
        X = pd.DataFrame(payload.data)
        preds = MODEL.predict(X)
        return {"n": len(preds), "predictions": [str(p) for p in preds], "model_uri": MODEL_URI_CACHE}
    except FileNotFoundError as e:
        # Falta el modelo: respuesta clara al cliente
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
