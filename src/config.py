"""Single source of truth for paths, constants and feature lists."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

# ---- Location -------------------------------------------------------------
LAT, LON = 24.86, 67.01          # Karachi
TZ = "Asia/Karachi"
GSOD_STATION = "41780099999"     # Karachi Jinnah Intl Airport (WMO 417800)

# ---- Collection period ------------------------------------------------------
START_DATE = "1980-01-01"
END_DATE = "2025-12-31"

# ---- Target definition (FROZEN: "Option B" = issue-day features -> future label)
HI_THRESHOLD_C = 40.6            # 105 F, our project's chosen threshold
HORIZON_MIN, HORIZON_MAX = 5, 7  # label looks at days t+5 .. t+7
TARGET = "danger_5to7d"

# ---- Open-Meteo variables ---------------------------------------------------
DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_mean",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]
# hourly variables are averaged to daily values (same code in training & live inference)
HOURLY_VARS = ["relative_humidity_2m", "dew_point_2m", "surface_pressure", "cloud_cover"]
HOURLY_AGG_NAMES = {
    "relative_humidity_2m": "relative_humidity_2m_mean",
    "dew_point_2m": "dew_point_2m_mean",
    "surface_pressure": "surface_pressure_mean",
    "cloud_cover": "cloud_cover_mean",
}
BASE_FEATURES = DAILY_VARS + list(HOURLY_AGG_NAMES.values())   # 14 raw weather inputs

ENGINEERED_NUM = [
    "temp_range", "dew_point_depression", "heat_index_era",
    "tmax_roll3", "tmax_roll7", "heat_index_roll3", "heat_index_roll7",
    "humidity_roll3", "dew_point_roll7", "pressure_change_3d",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
]
NUM_FEATURES = BASE_FEATURES + ENGINEERED_NUM                   # 28 numeric
CAT_FEATURES = ["season"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES                      # 29 inputs
SKEWED = ["precipitation_sum"]                                  # log1p-transformed in the pipeline

RANDOM_STATE = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
GAP_DAYS = 7
