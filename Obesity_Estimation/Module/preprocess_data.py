# Module/preprocess_data.py
import pandas as pd
from pathlib import Path

def load_raw_data(filename="obesity_estimation_modified.csv"):
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "raw" / filename
    df = pd.read_csv(file_path)
    return df

def create_interim_data(df):
    # Convertir columnas numéricas
    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Altura en metros y limpieza de outliers
    if "Height" in df.columns:
        df["height_m"] = df["Height"].apply(lambda x: x/100 if x>3 else x)
        df["height_m"] = df["height_m"].clip(lower=1.0, upper=2.2)
        df["height_m"].fillna(df["height_m"].median(), inplace=True)
    else:
        df["height_m"] = 1.7

    # Peso en kg y limpieza de outliers
    if "Weight" in df.columns:
        df["weight_kg"] = df["Weight"].clip(lower=30, upper=250)
        df["weight_kg"].fillna(df["weight_kg"].median(), inplace=True)
    else:
        df["weight_kg"] = 70.0

    # BMI
    df["bmi"] = df["weight_kg"] / df["height_m"]**2
    df["bmi"] = df["bmi"].clip(lower=10, upper=80)

    # family_history_bool
    if "family_history" in df.columns:
        df["family_history_bool"] = df["family_history"].map({"yes":1,"no":0}).fillna(0)
    else:
        df["family_history_bool"] = 0

    # age_scaled
    if "Age" in df.columns:
        df["age_scaled"] = (df["Age"] - df["Age"].mean()) / df["Age"].std()
    else:
        df["age_scaled"] = 0

    # Eliminar duplicados y NA
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))

    return df

def create_processed_data(df):
    df_proc = df.copy()
    categorical_cols = [col for col in ["Gender", "family_history"] if col in df_proc.columns]
    for col in categorical_cols:
        df_proc[col] = df_proc[col].astype("category").cat.codes

    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE",
                    "height_m", "weight_kg", "bmi", "family_history_bool", "age_scaled"]
    for col in numeric_cols:
        if col in df_proc.columns:
            df_proc[col] = pd.to_numeric(df_proc[col])

    df_proc = df_proc.dropna()
    df_proc = df_proc.drop_duplicates()
    return df_proc

if __name__ == "__main__":
    df_raw = load_raw_data()
    df_interim = create_interim_data(df_raw)
    df_processed = create_processed_data(df_interim)

    project_root = Path(__file__).resolve().parent.parent
    df_interim.to_csv(project_root / "data" / "interim" / "interim_data.csv", index=False)
    df_processed.to_csv(project_root / "data" / "processed" / "processed_data.csv", index=False)
    print("✅ Pipeline de preprocesamiento completado.")
