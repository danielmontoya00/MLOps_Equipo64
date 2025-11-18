# src/test/iunit/conftest.py
import pandas as pd
import pytest
from pathlib import Path

@pytest.fixture
def sample_raw_df():
    """DataFrame simulado para pruebas de make_dataset."""
    return pd.DataFrame({
        "Age": [25, 30],
        "Gender": ["Male", "Female"],
        "Height": [1.75, 1.60],
        "Weight": [70, 60],
        "NObeyesdad": ["Normal_Weight", "Overweight"]
    })

@pytest.fixture
def tmp_raw_file(tmp_path, sample_raw_df):
    """Genera un CSV temporal para make_dataset."""
    file_path = tmp_path / "obesity_raw_test.csv"
    sample_raw_df.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def tmp_processed_dir(tmp_path):
    """Directorio temporal para outputs del dataset procesado."""
    out_dir = tmp_path / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
