import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Cargar datos crudos
df = pd.read_csv('data/raw/obesity_estimation_final.csv')

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
X_train.to_csv('data/processed/X_train.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("Datos procesados y guardados en data/processed/")