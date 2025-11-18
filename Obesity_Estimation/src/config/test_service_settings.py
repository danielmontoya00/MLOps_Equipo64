import json
import pytest
from pathlib import Path

SERVICE_CONFIG_PATH = Path("src/config/service_config.json")


def test_service_config_exists():
    assert SERVICE_CONFIG_PATH.exists(), "service_config.json no encontrado"


def test_service_config_is_valid_json():
    try:
        with open(SERVICE_CONFIG_PATH) as f:
            config = json.load(f)
    except json.JSONDecodeError:
        pytest.fail("service_config.json está mal formado")

    assert isinstance(config, dict)


def test_service_config_required_fields():
    with open(SERVICE_CONFIG_PATH) as f:
        config = json.load(f)

    required_keys = ["host", "port", "debug", "api_prefix"]
    for k in required_keys:
        assert k in config, f"Falta la llave '{k}' en configuración del servicio"


def test_service_config_correct_types():
    with open(SERVICE_CONFIG_PATH) as f:
        config = json.load(f)

    assert isinstance(config["host"], str)
    assert isinstance(config["port"], int)
    assert isinstance(config["debug"], bool)
    assert isinstance(config["api_prefix"], str)


def test_service_port_in_valid_range():
    with open(SERVICE_CONFIG_PATH) as f:
        config = json.load(f)

    assert 1 < config["port"] < 65535, "Puerto fuera del rango permitido"
