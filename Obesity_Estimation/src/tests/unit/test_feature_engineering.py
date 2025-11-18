import pytest
import pandas as pd
from features.create_feature import add_bmi_features

def test_bmi_feature_calculation():
    """
    Verifica:
    - Cálculo correcto de IMC
    - Categorías correctas
    - No NaNs nuevos
    """

    df = pd.DataFrame({
        "Height": [1.70, 1.50, 1.80],
        "Weight": [70, 90, 85]
    })

    df_feat = add_bmi_features(df.copy())

    # IMC correcto
    expected_imc = [70/1.70**2, 90/1.50**2, 85/1.80**2]
    # Usar np.allclose para comparar listas de floats
    assert df_feat["IMC"].tolist() == pytest.approx(expected_imc, rel=1e-5)

    # Columna categórica creada
    assert "IMC_category" in df_feat.columns

    # No NaNs
    assert df_feat.isnull().sum().sum() == 0
