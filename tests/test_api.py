import pytest
pytest.importorskip("fastapi"); pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from api.main import app
from src.config import MODEL_DIR

client = TestClient(app)
OK = {"temperature_2m_max": 38, "temperature_2m_min": 28, "relative_humidity_2m_mean": 65, "dew_point_2m_mean": 24}
has_model = (MODEL_DIR / "heatwave_pipeline.joblib").exists()


def test_health_200():
    assert client.get("/health").status_code == 200


def test_index_served():
    assert client.get("/").status_code == 200


def test_missing_field_422():
    assert client.post("/predict", json={"temperature_2m_max": 38}).status_code == 422


def test_wrong_type_422():
    assert client.post("/predict", json={**OK, "temperature_2m_max": "hot"}).status_code == 422


def test_invalid_range_422():
    assert client.post("/predict", json={**OK, "relative_humidity_2m_mean": 150}).status_code == 422
    assert client.post("/predict", json={**OK, "temperature_2m_max": 20, "temperature_2m_min": 30}).status_code == 422


@pytest.mark.skipif(not has_model, reason="model not trained")
def test_valid_prediction_200():
    r = client.post("/predict", json=OK)
    assert r.status_code == 200 and r.json()["prediction"] in (0, 1) and 0 <= r.json()["probability"] <= 1


@pytest.mark.skipif(not has_model, reason="model not trained")
def test_model_info():
    assert client.get("/model-info").status_code == 200


@pytest.mark.skipif(has_model, reason="only when model missing")
def test_model_unavailable_503():
    assert client.post("/predict", json=OK).status_code == 503
