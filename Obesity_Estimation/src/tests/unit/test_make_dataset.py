# src/tests/unit/test_make_dataset.py
import pytest
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def tmp_raw_file(tmp_path):
    """Crea un CSV temporal con datos de prueba."""
    df = pd.DataFrame({
        "Gender": ["Male", "Female", "Female", "Male"],
        "Height": [1.70, 1.60, 1.65, 1.80],
        "Weight": [70, 50, 60, 90],
        "NObeyesdad": ["Normal", "Obesity I", "Normal", "Obesity II"]
    })
    tmp_file = tmp_path / "raw.csv"
    df.to_csv(tmp_file, index=False)
    return tmp_file

@pytest.fixture
def tmp_processed_dir(tmp_path):
    """Directorio temporal donde se guardarán los CSVs procesados."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    return processed_dir

# ============================================================
# TEST
# ============================================================

def test_make_dataset(tmp_raw_file, tmp_processed_dir):
    """Prueba el pipeline de procesamiento del dataset."""

    # 1. Cargar archivo
    df = pd.read_csv(tmp_raw_file)

    # 2. Codificación categórica
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    assert "Gender" in encoders
    assert df["Gender"].dtype in ["int32", "int64"]

    # 3. Separar X e y
    X = df.drop("NObeyesdad", axis=1)
    y = df["NObeyesdad"]

    assert "NObeyesdad" not in X.columns

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # 5. Guardar
    (tmp_processed_dir / "X_train.csv").write_text(X_train.to_csv(index=False))
    (tmp_processed_dir / "X_test.csv").write_text(X_test.to_csv(index=False))
    (tmp_processed_dir / "y_train.csv").write_text(y_train.to_csv(index=False))
    (tmp_processed_dir / "y_test.csv").write_text(y_test.to_csv(index=False))

    # 6. Verificación de archivos
    assert (tmp_processed_dir / "X_train.csv").exists()
    assert (tmp_processed_dir / "X_test.csv").exists()
    assert (tmp_processed_dir / "y_train.csv").exists()
    assert (tmp_processed_dir / "y_test.csv").exists()
