import pandas as pd
from src.features.clean_data import clean_numeric_data

def test_numeric_imputation():
    df = pd.DataFrame({
        "Age": [20, None, 30, 5000],
        "Weight": [60, 65, None, -999]
    })
    numeric_cols = ["Age", "Weight"]

    df_clean = clean_numeric_data(df.copy(), numeric_cols)

    # Ningún NaN después de la limpieza
    assert df_clean[numeric_cols].isnull().sum().sum() == 0

    # Los NaNs se reemplazan por la mediana de cada columna
    assert df_clean.loc[1, "Age"] == 30      # mediana de [20,30,5000] = 30
    assert df_clean.loc[2, "Weight"] == 60   # mediana de [60,65,70,-999] = 60