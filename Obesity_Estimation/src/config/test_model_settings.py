import json
import pytest
from pathlib import Path

# Ruta a tu archivo real
CONFIG_PATH = Path("src/config/model_config.json")


def test_model_config_file_exists():
    assert CONFIG_PATH.exists(), "model_config.json no existe en src/config/"


def test_model_config_valid_json():
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError:
        pytest.fail("model_config.json no está bien formado")

    assert isinstance(config, dict)


def test_required_hyperparameters_exist():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    required = ["model_type", "params"]

    for key in required:
        assert key in config, f"Falta la clave requerida '{key}'"


def test_hyperparameters_have_valid_types():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    params = config.get("params", {})

    # Ejemplo: RandomForest
    if config["model_type"] == "RandomForest":
        assert isinstance(params.get("n_estimators"), int)
        assert isinstance(params.get("max_depth"), (int, type(None)))
        assert isinstance(params.get("random_state"), int)


def test_hyperparameters_within_expected_range():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    params = config.get("params", {})

    if config["model_type"] == "RandomForest":
        assert params["n_estimators"] > 0
        assert params["max_depth"] is None or params["max_depth"] > 0
