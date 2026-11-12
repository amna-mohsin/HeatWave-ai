"""Open-Meteo access, shared by data collection (archive) and live inference (forecast)."""
import time
import pandas as pd
import requests
from .config import LAT, LON, TZ, DAILY_VARS, HOURLY_VARS, HOURLY_AGG_NAMES, BASE_FEATURES

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _get(url, params, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:           # rate limited -> back off
                time.sleep(20 * (attempt + 1)); continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Open-Meteo request failed")


def payload_to_daily(payload: dict) -> pd.DataFrame:
    """Daily vars + hourly vars averaged per day -> one row per date (columns = BASE_FEATURES)."""
    daily = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    hourly = pd.DataFrame(payload["hourly"])
    hourly["date"] = pd.to_datetime(hourly["time"]).dt.normalize()
    agg = hourly.groupby("date")[HOURLY_VARS].mean().rename(columns=HOURLY_AGG_NAMES).reset_index()
    return daily.merge(agg, on="date", how="left")[["date"] + BASE_FEATURES]


def fetch_archive(start: str, end: str) -> pd.DataFrame:
    """ERA5 reanalysis via the Open-Meteo Historical Weather API."""
    params = dict(latitude=LAT, longitude=LON, start_date=start, end_date=end, models="era5",
                  daily=",".join(DAILY_VARS), hourly=",".join(HOURLY_VARS), timezone=TZ)
    return payload_to_daily(_get(ARCHIVE_URL, params))


def fetch_recent_and_forecast(past_days=14, forecast_days=8) -> pd.DataFrame:
    """Recent days + forecast days from the Open-Meteo Forecast API (used by the apps)."""
    params = dict(latitude=LAT, longitude=LON, past_days=past_days, forecast_days=forecast_days,
                  daily=",".join(DAILY_VARS), hourly=",".join(HOURLY_VARS), timezone=TZ)
    return payload_to_daily(_get(FORECAST_URL, params))
