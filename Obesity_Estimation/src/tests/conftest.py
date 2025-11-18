import sys
from pathlib import Path

# Agrega la carpeta 'src' al path de Python
project_root = Path(__file__).resolve().parents[1]  # sube de tests/ → src/
sys.path.insert(0, str(project_root))