import os
import sys
import pytest

# Agrega src/ al PYTHONPATH para que los imports funcionen
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, SRC_PATH)

print("Ejecutando tests desde:", os.getcwd())
print("PYTHONPATH configurado a:", SRC_PATH)

# Ejecutar pytest en toda la carpeta de tests
pytest_args = [
    "tests",  # dentro de src/
    "-v",
]

pytest.main(pytest_args)
