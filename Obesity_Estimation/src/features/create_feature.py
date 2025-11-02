import pandas as pd

def add_bmi_features(df):
    """Crea columna de IMC y su categoría según la OMS."""
    df['IMC'] = df['Weight'] / (df['Height'] ** 2)

    def clasificar_imc(imc):
        if pd.isnull(imc): return 'Desconocido'
        elif imc < 18.5: return 'Bajo peso'
        elif imc < 25: return 'Normal'
        elif imc < 30: return 'Sobrepeso'
        elif imc < 35: return 'Obesidad I'
        elif imc < 40: return 'Obesidad II'
        else: return 'Obesidad III'

    df['IMC_category'] = df['IMC'].apply(clasificar_imc)
    return df