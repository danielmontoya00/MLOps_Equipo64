import pandas as pd
from src.features.clean_data import (
    clean_numeric_data,
    clean_categorical_data
)

def test_clean_numeric_data():
    df = pd.DataFrame({
        "edad": [20, None, 40],
        "peso": [55.0, 60.0, None]
    })

    numeric_cols = ["edad", "peso"]
    df_clean = clean_numeric_data(df, numeric_cols)

    # La media de edad = (20 + 40) / 2 = 30
    assert df_clean["edad"].isna().sum() == 0
    assert df_clean.loc[1, "edad"] == 30

    # La media de peso = (55 + 60) / 2 = 57.5
    assert df_clean["peso"].isna().sum() == 0
    assert df_clean.loc[2, "peso"] == 57.5


def test_clean_categorical_data():
    df = pd.DataFrame({
        "genero": ["Hombre", None, "Mujer", "Hombre"],
    })

    cat_cols = ["genero"]
    df_clean = clean_categorical_data(df, cat_cols)

    # La fila con None se convierte en 'none' (str)
    assert df_clean["genero"].isna().sum() == 0
    assert df_clean.loc[1, "genero"] == "none"


