import numpy as np, pandas as pd, pytest
from src.config import MODEL_DIR, ALL_FEATURES
from src.predict import predict_scenario, load_model

pytestmark = pytest.mark.skipif(not (MODEL_DIR / "heatwave_pipeline.joblib").exists(), reason="model not trained yet")
W = {"temperature_2m_max": 38, "temperature_2m_min": 28, "relative_humidity_2m_mean": 65, "dew_point_2m_mean": 24}


def test_pipeline_loads_and_has_all_features():
    pipe, meta = load_model()
    assert meta["features"] == ALL_FEATURES


def test_prediction_ranges():
    r = predict_scenario(W, date="2025-06-15")
    assert r["prediction"] in (0, 1) and 0 <= r["probability"] <= 1


def test_deterministic():
    assert predict_scenario(W, date="2025-06-15") == predict_scenario(W, date="2025-06-15")


def test_missing_optional_inputs_are_imputed():
    r = predict_scenario(W, date="2025-06-15")         # only 4 required fields given
    assert np.isfinite(r["probability"])


def test_hot_humid_scores_higher_than_cool_dry():
    hot = predict_scenario(W, date="2025-06-15")["probability"]
    cool = predict_scenario({"temperature_2m_max": 22, "temperature_2m_min": 12, "relative_humidity_2m_mean": 35, "dew_point_2m_mean": 5}, date="2025-01-15")["probability"]
    assert hot > cool
