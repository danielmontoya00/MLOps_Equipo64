import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Definir la raíz del proyecto
project_root = Path(__file__).resolve().parent.parent

raw_file = project_root / "data/raw/obesity_estimation_final.csv"
processed_dir = project_root / "data/processed"
interim_dir = project_root / "data/interim"

processed_dir.mkdir(parents=True, exist_ok=True)
interim_dir.mkdir(parents=True, exist_ok=True)

# Cargar datos crudos
df = pd.read_csv(raw_file)

# Cargar datos crudos
#df = pd.read_csv('data/raw/obesity_estimation_final.csv')

# --- LIMPIEZA Y TRANSFORMACIÓN ---
# (Este dataset está bastante limpio, pero aquí irían los pasos)
# Ejemplo: df.dropna(inplace=True)

# Codificar variables categóricas a numéricas
# Convertimos todas las columnas de tipo 'object' a números
encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le # Guardamos el encoder por si lo necesitamos

# Separar características (X) y variable objetivo (y)
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar los datos procesados
X_train.to_csv(processed_dir/'X_train.csv', index=False)
y_train.to_csv(processed_dir/'y_train.csv', index=False)
X_test.to_csv(processed_dir/'X_test.csv', index=False)
y_test.to_csv(processed_dir/'y_test.csv', index=False)


print(f"Datos procesados y guardados en {processed_dir}/")