# src/tests/unit/test_metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def test_accuracy_computation():
    """
    Verifica el cálculo de la exactitud (accuracy).
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    acc = accuracy_score(y_true, y_pred)

    assert acc == 0.75, "Accuracy esperado de 0.75"

def test_classification_report_output():
    """
    Verifica que el reporte de clasificación contenga las claves correctas.
    """
    y_true = [0, 1, 1]
    y_pred = [0, 1, 0]
    report = classification_report(y_true, y_pred, output_dict=True)

    assert "0" in report
    assert "1" in report
    assert "accuracy" in report
