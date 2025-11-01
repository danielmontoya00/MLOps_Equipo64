from pathlib import Path
import pandas as pd

def load_raw_data(filename="obesity_estimation_modified.csv"):
    """
    Carga los datos sin transformación desde un CSV ubicado en data/raw.
    Usa rutas absolutas basadas en la estructura del proyecto.
    """
    # Calcula la ruta absoluta del proyecto
    project_root = Path(__file__).resolve().parents[2]  # sube 2 niveles: src/data → Obesity_Estimation
    path = project_root / "data" / "raw" / filename

    # Verifica si el archivo existe
    if not path.exists():
        raise FileNotFoundError(f"❌ No se encontró el archivo en: {path}")

    # Carga el CSV
    df = pd.read_csv(path)
    print("✅ Datos cargados correctamente desde:", path)
    print(df.info())
    return df