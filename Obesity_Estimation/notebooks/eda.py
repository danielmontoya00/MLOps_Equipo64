import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.load_data import load_raw_data
from src.features.clean_data import clean_numeric_data, clean_categorical_data
from src.features.create_feature import add_bmi_features
from src.visualization.visualize import (
    plot_numeric_distributions,
    plot_categorical_distributions,
    plot_correlation_matrix,
    plot_bmi_distribution
)

# Definición de columnas
numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
categorical_cols = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS','NObeyesdad']

# --- Cargar datos ---
df = load_raw_data()

# --- Limpieza ---
df = clean_numeric_data(df, numeric_cols)
df = clean_categorical_data(df, categorical_cols)

# --- Feature Engineering ---
df = add_bmi_features(df)

# --- Visualizaciones ---
plot_numeric_distributions(df, numeric_cols)
plot_categorical_distributions(df, categorical_cols)
plot_correlation_matrix(df, numeric_cols)
plot_bmi_distribution(df)

print("✅ EDA completado. Figuras guardadas en /reports/figures/")