import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def data_paths():
    """
    Fixture con rutas del flujo de datos según Cookiecutter Data Science.
    """
    base = Path(__file__).resolve().parents[3] / "data"
    return {
        "raw": base / "raw" / "obesity_estimation_modified.csv",
        "interim": base / "interim" / "data_interim.csv",
        "processed": base / "processed" / "data_processed.csv",
    }


@pytest.fixture
def raw_df(data_paths):
    return pd.read_csv(data_paths["raw"])


@pytest.fixture
def interim_df(data_paths):
    return pd.read_csv(data_paths["interim"])


@pytest.fixture
def processed_df(data_paths):
    return pd.read_csv(data_paths["processed"])


# ------------------------------------------------------
# TEST 1 — Validar archivo raw
# ------------------------------------------------------
def test_raw_data_structure(raw_df):
    """
    El archivo RAW debe contener columnas mínimas esperadas.
    """
    expected_cols = {
        "Gender", "Age", "Height", "Weight",
        "family_history_with_overweight", "NObeyesdad"
    }

    assert expected_cols.issubset(
        set(raw_df.columns)
    ), f"Faltan columnas obligatorias en RAW: {expected_cols - set(raw_df.columns)}"

    assert len(raw_df) > 0, "El archivo RAW está vacío."


# ------------------------------------------------------
# TEST 2 — Flujo raw → interim mantiene filas
# ------------------------------------------------------
def test_raw_to_interim_row_count(raw_df, interim_df):
    """
    El preprocesamiento no debe eliminar más del 10% de las filas.
    """
    ratio = len(interim_df) / len(raw_df)
    assert ratio > 0.9, f"Se eliminaron demasiadas filas: quedaron {ratio:.2%}"


# ------------------------------------------------------
# TEST 3 — Columnas esperadas en INTERIM
# ------------------------------------------------------
def test_interim_columns(interim_df):
    """
    Validar que el dataset INTERIM tenga variables limpias y codificadas mínimas.
    """
    expected_cols = {
        "age_scaled",
        "height_m",
        "weight_kg",
        "bmi",
        "family_history_bool",
        "target"  # codificación de NObeyesdad
    }

    assert expected_cols.issubset(
        set(interim_df.columns)
    ), f"INTERIM no contiene columnas procesadas esperadas."


# ------------------------------------------------------
# TEST 4 — Validar valores numéricos (rangos razonables)
# ------------------------------------------------------
def test_numeric_ranges(interim_df):
    """
    Asegura rangos razonables después de preprocesamiento.
    """
    assert interim_df["height_m"].between(1.0, 2.2).all(), "Altura fuera de rango."
    assert interim_df["weight_kg"].between(30, 250).all(), "Peso fuera de rango."
    assert interim_df["bmi"].between(10, 80).all(), "BMI fuera de rango."


# ------------------------------------------------------
# TEST 5 — Validar duplicados en processed
# ------------------------------------------------------
def test_no_duplicates_processed(processed_df):
    """
    El dataset PROCESSED no debe contener duplicados.
    """
    duplicates = processed_df.duplicated().sum()
    assert duplicates == 0, f"PROCESSED contiene {duplicates} duplicados."


# ------------------------------------------------------
# TEST 6 — processed debe contener X_final y y_final
# ------------------------------------------------------
def test_processed_dataset_structure(processed_df):
    """
    Validar que PROCESSED es la matriz final lista para el modelado.
    """
    assert "target" in processed_df.columns, "PROCESSED debe contener 'target'."
    assert processed_df.drop(columns=["target"]).shape[1] > 3, \
        "PROCESSED debe tener al menos 3 variables predictoras."


# ------------------------------------------------------
# TEST 7 — Tipos de datos correctos
# ------------------------------------------------------
def test_datatypes(processed_df):
    """
    Validar que los tipos de datos sean numéricos para modelado.
    """
    numeric_cols = processed_df.drop(columns=["target"]).select_dtypes(include=["int64", "float64"]).columns

    assert len(numeric_cols) == (
        processed_df.shape[1] - 1
    ), "Hay columnas no numéricas en PROCESSED."


# ------------------------------------------------------
# TEST 8 — Sin valores nulos en processed
# ------------------------------------------------------
def test_no_nulls(processed_df):
    """
    Asegura que después del procesamiento final no existan valores nulos.
    """
    nulls = processed_df.isna().sum().sum()
    assert nulls == 0, f"PROCESSED contiene {nulls} valores nulos."
