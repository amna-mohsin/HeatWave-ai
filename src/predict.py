"""Reusable prediction engine used by the Streamlit dashboard."""
import json
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from .config import MODEL_DIR, BASE_FEATURES, ALL_FEATURES, HI_THRESHOLD_C
from .feature_engineering import build_features
from .heat import rh_from_dewpoint, heat_index_c

PIPELINE_FILE = MODEL_DIR / "heatwave_pipeline.joblib"
META_FILE = MODEL_DIR / "model_metadata.json"


class ModelNotAvailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_model():
    """Load a TRUSTED joblib file produced by src/train.py (joblib uses pickle: never load unknown files)."""
    if not PIPELINE_FILE.exists():
        raise ModelNotAvailable(f"{PIPELINE_FILE.name} not found. Run `python -m src.train` first.")
    meta = json.loads(META_FILE.read_text()) if META_FILE.exists() else {}
    return joblib.load(PIPELINE_FILE), meta


def risk_band(p: float, thr: float) -> str:
    return "Dangerous heat risk" if p >= thr else ("Elevated" if p >= thr * 0.5 else "Low")


def predict_features(feature_row: pd.DataFrame) -> dict:
    """feature_row: DataFrame with the ALL_FEATURES columns (one row)."""
    pipe, meta = load_model()
    X = feature_row[ALL_FEATURES]
    p = float(pipe.predict_proba(X)[:, 1][0]); thr = float(meta.get("decision_threshold", 0.5))
    return {"prediction": int(p >= thr), "probability": round(p, 4), "threshold": round(thr, 4),
            "label": "Dangerous Heat" if p >= thr else "No Dangerous Heat", "risk_band": risk_band(p, thr),
            "model": meta.get("model", "unknown")}


def predict_scenario(weather: dict, date=None, history_days: int = 14) -> dict:
    """User-supplied 'today' weather. Missing optional fields stay NaN (the pipeline imputes them).
    The recent past is assumed to equal today's conditions (steady-state scenario)."""
    date = pd.Timestamp(date or pd.Timestamp.now(tz="Asia/Karachi").date())
    row = {c: np.nan if weather.get(c) is None else weather.get(c, np.nan) for c in BASE_FEATURES}
    if row["temperature_2m_mean"] is None or pd.isna(row["temperature_2m_mean"]):
        row["temperature_2m_mean"] = (row["temperature_2m_max"] + row["temperature_2m_min"]) / 2
    rows = pd.DataFrame([row] * history_days)
    rows.insert(0, "date", pd.date_range(end=date, periods=history_days, freq="D"))
    out = predict_features(build_features(rows).tail(1))
    out["date"] = str(date.date()); out["source"] = "user scenario (steady-state history)"
    return out


def predict_live() -> dict:
    """Live: Open-Meteo recent days + today -> features for TODAY -> week-ahead danger probability."""
    from .weather_api import fetch_recent_and_forecast
    daily = fetch_recent_and_forecast(past_days=14, forecast_days=1)
    feats = build_features(daily).tail(1)
    out = predict_features(feats)
    out["date"] = str(pd.Timestamp(feats["date"].iloc[0]).date())
    out["source"] = "Open-Meteo forecast API (past 14 days + today)"
    return out


def forecast_heat_index(forecast_days: int = 7) -> pd.DataFrame:
    """Per-day heat index from the NWP forecast itself (a cross-check, NOT the ML model)."""
    from .weather_api import fetch_recent_and_forecast
    d = fetch_recent_and_forecast(past_days=0, forecast_days=forecast_days)
    rh = rh_from_dewpoint(d["temperature_2m_max"], d["dew_point_2m_mean"])
    d["heat_index_c"] = heat_index_c(d["temperature_2m_max"], rh)
    d["rh_peak"] = rh
    d["above_threshold"] = d["heat_index_c"] >= HI_THRESHOLD_C
    return d
