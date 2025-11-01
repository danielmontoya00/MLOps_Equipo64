from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    # === 1. Definir rutas absolutas ===
    project_root = Path(__file__).resolve().parents[1]  # sube desde /notebooks → /Obesity_Estimation
    raw_path = project_root / "data" / "raw" / "obesity_estimation_final.csv"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # === 2. Cargar datos crudos ===
    if not raw_path.exists():
        raise FileNotFoundError(f"❌ No se encontró el archivo en: {raw_path}")

    df = pd.read_csv(raw_path)
    print(f"Datos cargados desde: {raw_path}")
    print(f"Shape: {df.shape}")

    # === 3. Codificación de variables categóricas ===
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # === 4. Separar features (X) y target (y) ===
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']

    # === 5. División de datos ===
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # === 6. Guardar datasets procesados ===
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    X_val.to_csv(processed_dir / "X_val.csv", index=False)
    y_val.to_csv(processed_dir / "y_val.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    print("✅ Datos procesados y guardados en:")
    print(processed_dir)


if __name__ == "__main__":
    main()