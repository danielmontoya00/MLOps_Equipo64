from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os, json
import pandas as pd
import pickle
import mlflow.sklearn

app = FastAPI(
    title="Obesity Classifier API",
    version="1.0.0",
    description="""
    API para predicción de niveles de obesidad basada en características demográficas y de estilo de vida.
    
    ## Características del modelo
    
    El modelo predice el nivel de obesidad basándose en:
    - Datos demográficos (género, edad, altura, peso)
    - Hábitos alimenticios (consumo de vegetales, comidas principales, snacks)
    - Estilo de vida (actividad física, consumo de agua, uso de tecnología)
    - Historial familiar de sobrepeso
    
    ## Endpoints disponibles
    
    - **GET /** - Health check y estado del modelo
    - **POST /predict** - Realizar predicciones de obesidad
    """,
    contact={
        "name": "MLOps Equipo 64",
        "url": "https://github.com/yourusername/MLOps_Equipo64",
    }
)

# ---------- Configuración ----------
# Prioridad:
# 1) MLFLOW_MODEL_URI (ej. "models:/obesity_classifier/Production" o "runs:/<run_id>/model")
# 2) Obesity_Estimation/models/model_info.json -> "model_uri"
# 3) Cargar directamente el último .pkl disponible
MLFLOW_MODEL_URI_ENV = os.getenv("MLFLOW_MODEL_URI")
MODEL_URI_CACHE: str | None = None
MODEL = None  # se inicializa on-demand

# Mapeo de índices a nombres de clases de obesidad
# Basado en el orden alfabético típico de LabelEncoder
OBESITY_CLASSES = {
    0: "Insufficient_Weight",
    1: "Normal_Weight", 
    2: "Obesity_Type_I",
    3: "Obesity_Type_II",
    4: "Obesity_Type_III",
    5: "Overweight_Level_I",
    6: "Overweight_Level_II"
}


class InputFeatures(BaseModel):
    """Features de entrada para la predicción de obesidad."""
    Gender: int = Field(..., ge=0, le=1, description="Género: 0=Femenino, 1=Masculino", example=0)
    Age: float = Field(..., gt=0, le=120, description="Edad en años", example=21.0)
    Height: float = Field(..., gt=0, le=3, description="Altura en metros", example=1.70)
    Weight: float = Field(..., gt=0, le=300, description="Peso en kilogramos", example=50.0)
    family_history_with_overweight: int = Field(..., ge=0, le=1, description="Historial familiar de sobrepeso: 0=No, 1=Sí", example=0)
    FAVC: int = Field(..., ge=0, le=1, description="Consumo frecuente de alimentos altos en calorías: 0=No, 1=Sí", example=0)
    FCVC: float = Field(..., ge=0, le=3, description="Frecuencia de consumo de vegetales (0-3)", example=3.0)
    NCP: float = Field(..., ge=0, le=4, description="Número de comidas principales al día", example=3.0)
    CAEC: int = Field(..., ge=0, le=3, description="Consumo de alimentos entre comidas: 0=No, 1=A veces, 2=Frecuentemente, 3=Siempre", example=0)
    SMOKE: int = Field(..., ge=0, le=1, description="Fuma: 0=No, 1=Sí", example=0)
    CH2O: float = Field(..., ge=0, le=3, description="Consumo diario de agua en litros", example=2.5)
    SCC: int = Field(..., ge=0, le=1, description="Monitorea el consumo de calorías: 0=No, 1=Sí", example=0)
    FAF: float = Field(..., ge=0, le=3, description="Frecuencia de actividad física (0=Nunca, 1=1-2 días, 2=2-4 días, 3=4-5 días)", example=2.0)
    TUE: float = Field(..., ge=0, le=24, description="Tiempo de uso de dispositivos tecnológicos (horas/día)", example=1.0)
    CALC: int = Field(..., ge=0, le=3, description="Consumo de alcohol: 0=No, 1=A veces, 2=Frecuentemente, 3=Siempre", example=0)
    MTRANS: int = Field(..., ge=0, le=4, description="Medio de transporte: 0=Auto, 1=Moto, 2=Bicicleta, 3=Transporte público, 4=Caminando", example=4)
    
    class Config:
        schema_extra = {
            "examples": {
                "insufficient_weight": {
                    "summary": "Peso Insuficiente",
                    "description": "Ejemplo de persona con peso insuficiente (bajo peso)",
                    "value": {
                        "Gender": 0,
                        "Age": 21.0,
                        "Height": 1.70,
                        "Weight": 48.0,
                        "family_history_with_overweight": 0,
                        "FAVC": 0,
                        "FCVC": 3.0,
                        "NCP": 3.0,
                        "CAEC": 0,
                        "SMOKE": 0,
                        "CH2O": 2.5,
                        "SCC": 0,
                        "FAF": 2.5,
                        "TUE": 1.0,
                        "CALC": 0,
                        "MTRANS": 4
                    }
                },
                "normal_weight": {
                    "summary": "Peso Normal",
                    "description": "Ejemplo de persona con peso normal",
                    "value": {
                        "Gender": 1,
                        "Age": 25.0,
                        "Height": 1.75,
                        "Weight": 70.0,
                        "family_history_with_overweight": 0,
                        "FAVC": 0,
                        "FCVC": 2.5,
                        "NCP": 3.0,
                        "CAEC": 1,
                        "SMOKE": 0,
                        "CH2O": 2.0,
                        "SCC": 0,
                        "FAF": 2.0,
                        "TUE": 1.5,
                        "CALC": 1,
                        "MTRANS": 3
                    }
                },
                "overweight": {
                    "summary": "Sobrepeso",
                    "description": "Ejemplo de persona con sobrepeso",
                    "value": {
                        "Gender": 1,
                        "Age": 30.0,
                        "Height": 1.75,
                        "Weight": 85.0,
                        "family_history_with_overweight": 1,
                        "FAVC": 1,
                        "FCVC": 2.0,
                        "NCP": 3.0,
                        "CAEC": 2,
                        "SMOKE": 0,
                        "CH2O": 1.5,
                        "SCC": 0,
                        "FAF": 1.0,
                        "TUE": 2.0,
                        "CALC": 2,
                        "MTRANS": 0
                    }
                },
                "obesity": {
                    "summary": "Obesidad",
                    "description": "Ejemplo de persona con obesidad",
                    "value": {
                        "Gender": 1,
                        "Age": 35.0,
                        "Height": 1.75,
                        "Weight": 110.0,
                        "family_history_with_overweight": 1,
                        "FAVC": 1,
                        "FCVC": 1.0,
                        "NCP": 4.0,
                        "CAEC": 3,
                        "SMOKE": 0,
                        "CH2O": 1.0,
                        "SCC": 0,
                        "FAF": 0.0,
                        "TUE": 3.0,
                        "CALC": 2,
                        "MTRANS": 0
                    }
                }
            }
        }


class Payload(BaseModel):
    """Payload de entrada conteniendo una lista de registros a predecir."""
    data: List[InputFeatures] = Field(
        ..., 
        description="Lista de registros con features para predicción",
        min_items=1
    )
    
    class Config:
        schema_extra = {
            "examples": {
                "insufficient_weight": {
                    "summary": "Ejemplo: Peso Insuficiente",
                    "value": {
                        "data": [{
                            "Gender": 0,
                            "Age": 21.0,
                            "Height": 1.70,
                            "Weight": 48.0,
                            "family_history_with_overweight": 0,
                            "FAVC": 0,
                            "FCVC": 3.0,
                            "NCP": 3.0,
                            "CAEC": 0,
                            "SMOKE": 0,
                            "CH2O": 2.5,
                            "SCC": 0,
                            "FAF": 2.5,
                            "TUE": 1.0,
                            "CALC": 0,
                            "MTRANS": 4
                        }]
                    }
                },
                "normal_weight": {
                    "summary": "Ejemplo: Peso Normal",
                    "value": {
                        "data": [{
                            "Gender": 1,
                            "Age": 25.0,
                            "Height": 1.75,
                            "Weight": 70.0,
                            "family_history_with_overweight": 0,
                            "FAVC": 0,
                            "FCVC": 2.5,
                            "NCP": 3.0,
                            "CAEC": 1,
                            "SMOKE": 0,
                            "CH2O": 2.0,
                            "SCC": 0,
                            "FAF": 2.0,
                            "TUE": 1.5,
                            "CALC": 1,
                            "MTRANS": 3
                        }]
                    }
                },
                "overweight": {
                    "summary": "Ejemplo: Sobrepeso",
                    "value": {
                        "data": [{
                            "Gender": 1,
                            "Age": 30.0,
                            "Height": 1.75,
                            "Weight": 85.0,
                            "family_history_with_overweight": 1,
                            "FAVC": 1,
                            "FCVC": 2.0,
                            "NCP": 3.0,
                            "CAEC": 2,
                            "SMOKE": 0,
                            "CH2O": 1.5,
                            "SCC": 0,
                            "FAF": 1.0,
                            "TUE": 2.0,
                            "CALC": 2,
                            "MTRANS": 0
                        }]
                    }
                },
                "obesity": {
                    "summary": "Ejemplo: Obesidad",
                    "value": {
                        "data": [{
                            "Gender": 1,
                            "Age": 35.0,
                            "Height": 1.75,
                            "Weight": 110.0,
                            "family_history_with_overweight": 1,
                            "FAVC": 1,
                            "FCVC": 1.0,
                            "NCP": 4.0,
                            "CAEC": 3,
                            "SMOKE": 0,
                            "CH2O": 1.0,
                            "SCC": 0,
                            "FAF": 0.0,
                            "TUE": 3.0,
                            "CALC": 2,
                            "MTRANS": 0
                        }]
                    }
                },
                "multiple_predictions": {
                    "summary": "Ejemplo: Múltiples Predicciones",
                    "value": {
                        "data": [
                            {
                                "Gender": 0,
                                "Age": 21.0,
                                "Height": 1.70,
                                "Weight": 48.0,
                                "family_history_with_overweight": 0,
                                "FAVC": 0,
                                "FCVC": 3.0,
                                "NCP": 3.0,
                                "CAEC": 0,
                                "SMOKE": 0,
                                "CH2O": 2.5,
                                "SCC": 0,
                                "FAF": 2.5,
                                "TUE": 1.0,
                                "CALC": 0,
                                "MTRANS": 4
                            },
                            {
                                "Gender": 1,
                                "Age": 25.0,
                                "Height": 1.75,
                                "Weight": 70.0,
                                "family_history_with_overweight": 0,
                                "FAVC": 0,
                                "FCVC": 2.5,
                                "NCP": 3.0,
                                "CAEC": 1,
                                "SMOKE": 0,
                                "CH2O": 2.0,
                                "SCC": 0,
                                "FAF": 2.0,
                                "TUE": 1.5,
                                "CALC": 1,
                                "MTRANS": 3
                            }
                        ]
                    }
                }
            }
        }


class PredictionResponse(BaseModel):
    """Respuesta con las predicciones del modelo."""
    n: int = Field(..., description="Número de predicciones realizadas")
    predictions: List[str] = Field(..., description="Lista de predicciones de nivel de obesidad")
    model_uri: str = Field(..., description="URI del modelo utilizado para la predicción")
    
    class Config:
        schema_extra = {
            "example": {
                "n": 1,
                "predictions": ["Normal_Weight"],
                "model_uri": "file:///app/Obesity_Estimation/models/obesity_classifier_latest.pkl"
            }
        }


class HealthResponse(BaseModel):
    """Respuesta del endpoint de health check."""
    status: str = Field(..., description="Estado del servicio: 'ready' o 'model_missing'")
    model_uri_env: str | None = Field(None, description="URI del modelo desde variable de entorno")
    model_uri_json: str | None = Field(None, description="URI del modelo desde model_info.json")
    model_uri_pkl: str | None = Field(None, description="Ruta al archivo .pkl encontrado")
    active_model_uri: str | None = Field(None, description="URI del modelo actualmente cargado")
    resolved_uri: str | None = Field(None, description="URI resuelto que se usará para cargar el modelo")


class Payload(BaseModel):
    data: list[dict] = Field(..., description="Lista de filas con las mismas columnas que X_test.csv (Modo rápido)")


def _read_model_info_json() -> str | None:
    # Buscar en múltiples rutas posibles
    paths = [
        "models/model_info.json",
        "Obesity_Estimation/models/model_info.json",
        "/app/Obesity_Estimation/models/model_info.json"
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)
            return info.get("model_uri")
    return None


def _find_latest_pkl_model() -> str | None:
    """Busca el último modelo .pkl disponible en el directorio de modelos."""
    model_dirs = [
        "models",
        "Obesity_Estimation/models",
        "/app/Obesity_Estimation/models"
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            if pkl_files:
                # Usar el último modificado
                pkl_files_with_path = [os.path.join(model_dir, f) for f in pkl_files]
                latest_pkl = max(pkl_files_with_path, key=os.path.getmtime)
                return latest_pkl
    return None


def _resolve_model_uri() -> str | None:
    # 1) ENV - solo si está configurado explícitamente
    if MLFLOW_MODEL_URI_ENV:
        return MLFLOW_MODEL_URI_ENV
    
    # 2) Buscar .pkl directamente (prioridad alta para Docker)
    pkl_path = _find_latest_pkl_model()
    if pkl_path:
        return f"file://{pkl_path}"
    
    # 3) JSON (solo si no hay .pkl disponible)
    json_uri = _read_model_info_json()
    if json_uri:
        return json_uri
    
    return None


def _ensure_model_loaded():
    """Carga el modelo una sola vez, justo antes de predecir."""
    global MODEL, MODEL_URI_CACHE
    if MODEL is not None:
        return
    
    uri = _resolve_model_uri()
    if not uri:
        raise FileNotFoundError(
            "No hay MODEL_URI disponible. Define MLFLOW_MODEL_URI, coloca model_info.json, "
            "o asegúrate de que hay archivos .pkl en Obesity_Estimation/models/"
        )
    
    MODEL_URI_CACHE = uri
    
    # Cargar modelo según el tipo de URI
    if uri.startswith("file://"):
        # Cargar directamente desde pickle
        pkl_path = uri.replace("file://", "")
        with open(pkl_path, "rb") as f:
            MODEL = pickle.load(f)
    else:
        # Cargar desde MLflow
        MODEL = mlflow.sklearn.load_model(uri)


@app.get("/", response_model=HealthResponse, tags=["Health"])
def health():
    """
    Health check del servicio.
    
    Retorna el estado del servicio y la información sobre el modelo cargado o disponible.
    - **status**: 'ready' si hay un modelo disponible, 'model_missing' si no
    - **model_uri_***: Diferentes fuentes de configuración del modelo
    """
    model_uri = _resolve_model_uri()
    status = "ready" if MODEL is not None or model_uri else "model_missing"
    return {
        "status": status,
        "model_uri_env": MLFLOW_MODEL_URI_ENV,
        "model_uri_json": _read_model_info_json(),
        "model_uri_pkl": _find_latest_pkl_model(),
        "active_model_uri": MODEL_URI_CACHE,
        "resolved_uri": model_uri
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(payload: Payload):
    """
    Realizar predicción de nivel de obesidad.
    
    ## Descripción
    Este endpoint recibe una lista de registros con características demográficas y de estilo de vida,
    y retorna la predicción del nivel de obesidad para cada registro.
    
    ## Niveles de Obesidad Posibles
    - **Insufficient_Weight**: Peso insuficiente (IMC < 18.5)
    - **Normal_Weight**: Peso normal (IMC 18.5-24.9)
    - **Overweight_Level_I**: Sobrepeso nivel I (IMC 25-27.4)
    - **Overweight_Level_II**: Sobrepeso nivel II (IMC 27.5-29.9)
    - **Obesity_Type_I**: Obesidad tipo I (IMC 30-34.9)
    - **Obesity_Type_II**: Obesidad tipo II (IMC 35-39.9)
    - **Obesity_Type_III**: Obesidad tipo III (IMC ≥ 40)
    
    ## Ejemplo de uso
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "data": [{
                "Gender": 1,
                "Age": 21.0,
                "Height": 1.75,
                "Weight": 70.5,
                "family_history_with_overweight": 1,
                "FAVC": 1,
                "FCVC": 2.5,
                "NCP": 3.0,
                "CAEC": 2,
                "SMOKE": 0,
                "CH2O": 2.0,
                "SCC": 0,
                "FAF": 1.5,
                "TUE": 2.0,
                "CALC": 1,
                "MTRANS": 3
            }]
        }
    )
    print(response.json())
    ```
    
    ## Respuestas
    - **200**: Predicción exitosa
    - **400**: Error en los datos de entrada
    - **503**: Modelo no disponible
    """
    try:
        _ensure_model_loaded()
        # Convertir lista de InputFeatures a lista de dicts
        data_dicts = [item.dict() if hasattr(item, 'dict') else item for item in payload.data]
        X = pd.DataFrame(data_dicts)
        preds = MODEL.predict(X)
        
        # Convertir índices numéricos a nombres de clases
        prediction_names = []
        for pred in preds:
            pred_int = int(pred)
            pred_name = OBESITY_CLASSES.get(pred_int, f"Unknown_Class_{pred_int}")
            prediction_names.append(pred_name)
        
        return {
            "n": len(preds),
            "predictions": prediction_names,
            "model_uri": MODEL_URI_CACHE
        }
    except FileNotFoundError as e:
        # Falta el modelo: respuesta clara al cliente
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
