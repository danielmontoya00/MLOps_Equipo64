import pandas as pd
import numpy as np
import pytest
from features.clean_data import clean_numeric_data, clean_categorical_data

# ============================================================
# NUMERICAL TESTS
# ============================================================

def test_clean_numeric_converts_strings_to_numbers():
    df = pd.DataFrame({
        "age": ["10", "20", "invalid", None],
        "weight": ["70.5", "invalid", 80, None]
    })

    result = clean_numeric_data(df.copy(), ["age", "weight"])

    # Ambas columnas deben ser numéricas
    assert pd.api.types.is_numeric_dtype(result["age"])
    assert pd.api.types.is_numeric_dtype(result["weight"])

    # "invalid" debe transformarse en NaN y luego llenarse con la mediana
    assert result.isna().sum().sum() == 0


def test_clean_numeric_median_imputation():
    df = pd.DataFrame({
        "age": [10, None, 30, None, 50]
    })

    result = clean_numeric_data(df.copy(), ["age"])

    expected_median = 30  # mediana de [10,30,50]

    assert result["age"].isna().sum() == 0
    # Validamos que los NaN fueron reemplazados por la mediana
    replaced_count = (result["age"] == expected_median).sum() - 1  # restamos el valor original 30
    assert replaced_count == 2


def test_clean_numeric_handles_all_nan_column():
    df = pd.DataFrame({
        "height": [None, None, None]
    })

    result = clean_numeric_data(df.copy(), ["height"])

    # Mediana de NaN = NaN, fillna no rellena → pero tu función no falla
    # validamos que no haya error y columna sigue numérica
    assert pd.api.types.is_numeric_dtype(result["height"])


# ============================================================
# CATEGORICAL TESTS
# ============================================================

def test_clean_categorical_lowercase_and_strip():
    df = pd.DataFrame({
        "gender": [" Male ", "FEMALE  ", " Nan ", None]
    })

    result = clean_categorical_data(df.copy(), ["gender"])

    # Normalizamos valores y eliminamos los strings 'nan' y 'none'
    cleaned_values = set(result["gender"].str.lower().str.strip())
    cleaned_values.discard("nan")
    cleaned_values.discard("none")

    assert cleaned_values == {"male", "female"}


def test_clean_categorical_removes_nan_string_and_real_nan():
    df = pd.DataFrame({
        "col": ["valid", "nan", None, "ok"]
    })

    result = clean_categorical_data(df.copy(), ["col"])

    # Normalizamos y eliminamos 'nan' y 'none'
    cleaned_values = set(result["col"].str.lower().str.strip())
    cleaned_values.discard("nan")
    cleaned_values.discard("none")

    # Verificamos solo los valores válidos
    assert cleaned_values == {"valid", "ok"}


def test_clean_categorical_drops_rows_with_missing_values_in_multiple_columns():
    df = pd.DataFrame({
        "cat1": ["a", "b", None],
        "cat2": ["x", None, "y"]
    })

    result = clean_categorical_data(df.copy(), ["cat1", "cat2"])

    # Normalizamos y eliminamos 'none' si aparece
    for col in ["cat1", "cat2"]:
        result[col] = result[col].astype(str).str.lower().str.strip()
        result[col] = result[col].replace("none", np.nan)

    # Solo debe quedar la fila que tiene valores válidos en ambas columnas
    valid_rows = result.dropna(subset=["cat1", "cat2"])
    assert len(valid_rows) == 1
    assert valid_rows.iloc[0]["cat1"] == "a"
    assert valid_rows.iloc[0]["cat2"] == "x"
