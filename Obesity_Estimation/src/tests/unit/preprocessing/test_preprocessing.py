import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def test_label_encoding():
    df = pd.DataFrame({
        "Genero": ["Hombre", "Mujer", "Mujer"],
        "Edad": [23, 45, 31]
    })

    le = LabelEncoder()
    df["Genero_enc"] = le.fit_transform(df["Genero"])

    assert df["Genero_enc"].dtype == "int32" or "int64"
    assert df["Genero_enc"].nunique() == 2

def test_train_test_split_shapes():
    df = pd.DataFrame({
        "col1": range(100),
        "target": [0 if i < 50 else 1 for i in range(100)]
    })

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20