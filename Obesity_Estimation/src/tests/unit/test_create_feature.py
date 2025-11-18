import pytest
import pandas as pd
import numpy as np

from features.create_feature import add_bmi_features
from api.app import app

# ============================================================
# TESTS
# ============================================================

def test_add_bmi_features_computation():
    df = pd.DataFrame({
        "Weight": [70, 50, 90],
        "Height": [1.75, 1.60, 1.80]
    })
    result = add_bmi_features(df.copy())
    expected_imc = df["Weight"] / (df["Height"] ** 2)
    assert "IMC" in result.columns
    assert np.allclose(result["IMC"], expected_imc)


def test_add_bmi_features_categories():
    df = pd.DataFrame({
        "Weight": [50, 68, 85, 110, 140],
        "Height": [1.70, 1.70, 1.70, 1.70, 1.70]
    })
    result = add_bmi_features(df.copy())
    categorias = result["IMC_category"].tolist()
    assert categorias[0] == "Bajo peso"
    assert categorias[1] == "Normal"
    assert categorias[2] == "Sobrepeso"
    assert categorias[3] == "Obesidad II"
    assert categorias[4] == "Obesidad III"


def test_add_bmi_missing_values():
    df = pd.DataFrame({
        "Weight": [70, None],
        "Height": [1.75, 1.80]
    })
    result = add_bmi_features(df.copy())
    assert result.loc[1, "IMC_category"] == "Desconocido"
