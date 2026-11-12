"""Feature engineering + label generation. Used identically in training and live inference."""
import numpy as np
import pandas as pd
from .config import (BASE_FEATURES, ALL_FEATURES, HI_THRESHOLD_C, HORIZON_MIN, HORIZON_MAX)
from .heat import rh_from_dewpoint, heat_index_c


def season_of(month: int) -> str:
    """Karachi-oriented seasons."""
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "pre_monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
            10: "autumn", 11: "autumn"}[month]


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """daily: DataFrame with 'date' + BASE_FEATURES. Returns one row per calendar day with ALL_FEATURES.
    Every rolling/lag feature at day t uses ONLY days <= t (no future information)."""
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date").set_index("date").sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))   # explicit calendar
    df.index.name = "date"

    tmax, tmin = df["temperature_2m_max"], df["temperature_2m_min"]
    rh = df["relative_humidity_2m_mean"]
    df["temp_range"] = tmax - tmin
    df["dew_point_depression"] = df["temperature_2m_mean"] - df["dew_point_2m_mean"]
    rh_peak = rh_from_dewpoint(tmax, df["dew_point_2m_mean"])
    df["heat_index_era"] = pd.Series(heat_index_c(tmax, rh_peak), index=df.index)
    df["tmax_roll3"] = tmax.rolling(3, min_periods=2).mean()
    df["tmax_roll7"] = tmax.rolling(7, min_periods=4).mean()
    df["heat_index_roll3"] = df["heat_index_era"].rolling(3, min_periods=2).mean()
    df["heat_index_roll7"] = df["heat_index_era"].rolling(7, min_periods=4).mean()
    df["humidity_roll3"] = rh.rolling(3, min_periods=2).mean()
    df["dew_point_roll7"] = df["dew_point_2m_mean"].rolling(7, min_periods=4).mean()
    df["pressure_change_3d"] = df["surface_pressure_mean"] - df["surface_pressure_mean"].shift(3)
    m, d = df.index.month, df.index.dayofyear
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * d / 365.25), np.cos(2 * np.pi * d / 365.25)
    df["season"] = [season_of(x) for x in m]
    return df[ALL_FEATURES].reset_index()


def make_label(station_hi: pd.Series) -> pd.Series:
    """station_hi: heat index (C) indexed by date. Returns danger_5to7d aligned to the same dates.
    1 if HI >= threshold on ANY of days t+5..t+7; 0 if all three days are observed and below;
    NaN if unknown. Uses calendar dates (reindexed daily), never row positions."""
    hi = station_hi.sort_index()
    hi = hi.reindex(pd.date_range(hi.index.min(), hi.index.max(), freq="D"))
    fut = pd.concat([hi.shift(-k) for k in range(HORIZON_MIN, HORIZON_MAX + 1)], axis=1)
    any_hot = (fut >= HI_THRESHOLD_C).any(axis=1)
    all_known = fut.notna().all(axis=1)
    label = pd.Series(np.nan, index=hi.index)
    label[any_hot] = 1.0
    label[~any_hot & all_known] = 0.0
    return label
