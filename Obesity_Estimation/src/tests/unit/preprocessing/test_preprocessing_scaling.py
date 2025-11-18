import pytest
import pandas as pd
import numpy as np

from src.features.clean_data import clean_numeric_data


def test_numeric_scaling_normalization():
    """
    Verifica que clean_numeric_data:
    - convierta numéricos correctamente
    - deje los valores en rangos razonables
    - no genere NaNs
    """
    df = pd.DataFrame({
        "Age": [10, 20, 30, 40],
        "Height": [1.5, 1.7, 1.8, 1.6],
        "Weight": [50, 60, 70, 80]
    })

    numeric_cols = ["Age", "Height", "Weight"]
    df_clean = clean_numeric_data(df.copy(), numeric_cols)

    # No NaNs nuevos
    assert df_clean[numeric_cols].isnull().sum().sum() == 0

    # Verificar tipos numéricos
    for col in numeric_cols:
        assert np.issubdtype(df_clean[col].dtype, np.number)

    # Rango razonable tras normalización (si aplica)
    for col in numeric_cols:
        assert df_clean[col].std() != 0
        assert df_clean[col].min() >= df[col].min() - 1e-5
        assert df_clean[col].max() <= df[col].max() + 1e-5
