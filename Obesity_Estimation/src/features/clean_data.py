import pandas as pd
import numpy as np

def clean_numeric_data(df, numeric_cols):
    """Convierte columnas numéricas, elimina NaN y homogeniza formatos."""
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def clean_categorical_data(df, categorical_cols):
    """Limpia texto y reemplaza valores nulos."""
    for col in categorical_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace('nan', np.nan)
        )
    df.dropna(subset=categorical_cols, inplace=True)
    return df