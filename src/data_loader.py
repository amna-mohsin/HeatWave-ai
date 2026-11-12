"""Load, clean, merge and validate the two raw datasets -> final modelling dataset.
Run:  python -m src.data_loader"""
import json
import numpy as np
import pandas as pd
from .config import RAW_DIR, PROC_DIR, REPORT_DIR, BASE_FEATURES, ALL_FEATURES, TARGET
from .feature_engineering import build_features, make_label
from .heat import rh_from_dewpoint, heat_index_c


def f_to_c(x):
    return (x - 32.0) * 5.0 / 9.0


def clean_open_meteo(df: pd.DataFrame):
    log = {"rows_raw": len(df)}
    df = df.copy(); df["date"] = pd.to_datetime(df["date"], errors="coerce")
    log["bad_dates"] = int(df["date"].isna().sum()); df = df.dropna(subset=["date"])
    for c in BASE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    log["duplicate_dates"] = int(df["date"].duplicated().sum())
    df = df.drop_duplicates("date").sort_values("date")
    # physical-range checks (set impossible values to NaN; heatwave extremes are KEPT)
    bounds = {"relative_humidity_2m_mean": (0, 100), "cloud_cover_mean": (0, 100),
              "temperature_2m_max": (-20, 55), "temperature_2m_min": (-20, 50),
              "surface_pressure_mean": (850, 1100), "precipitation_sum": (0, 600)}
    log["out_of_range_set_nan"] = {}
    for c, (lo, hi) in bounds.items():
        bad = (df[c] < lo) | (df[c] > hi)
        log["out_of_range_set_nan"][c] = int(bad.sum()); df.loc[bad, c] = np.nan
    log["rows_clean"] = len(df)
    return df.reset_index(drop=True), log


def clean_gsod(df: pd.DataFrame):
    """GSOD: Fahrenheit + sentinel values (9999.9 / 999.9 / 99.99 = missing). Output in Celsius."""
    log = {"rows_raw": len(df)}
    df = df.copy(); df["date"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["date"])
    log["duplicate_dates"] = int(df["date"].duplicated().sum())
    df = df.drop_duplicates("date").sort_values("date")
    for c in ["TEMP", "DEWP", "MAX", "MIN", "SLP", "WDSP", "PRCP"]:
        if c not in df: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["TEMP", "DEWP", "MAX", "MIN", "SLP", "WDSP"]:
        df.loc[df[c] >= 999, c] = np.nan
    df.loc[df["PRCP"] >= 99, "PRCP"] = np.nan
    out = pd.DataFrame({"date": df["date"],
                        "gsod_temp_c": f_to_c(df["TEMP"]), "gsod_dewp_c": f_to_c(df["DEWP"]),
                        "gsod_max_c": f_to_c(df["MAX"]), "gsod_min_c": f_to_c(df["MIN"])})
    # station-based heat index: peak temperature + mean dew point (dew point cannot exceed air temp)
    dew = np.minimum(out["gsod_dewp_c"], out["gsod_max_c"])
    rh = rh_from_dewpoint(out["gsod_max_c"], dew)
    out["station_rh_peak"] = rh
    out["station_heat_index_c"] = heat_index_c(out["gsod_max_c"], rh)
    log["rows_clean"] = len(out)
    log["missing_station_heat_index"] = int(out["station_heat_index_c"].isna().sum())
    return out.reset_index(drop=True), log


def build_dataset():
    om, log_om = clean_open_meteo(pd.read_csv(RAW_DIR / "open_meteo.csv"))
    gs, log_gs = clean_gsod(pd.read_csv(RAW_DIR / "noaa_gsod.csv", low_memory=False))
    merged = pd.merge(om, gs, on="date", how="inner")                  # both sources on calendar date
    assert merged["date"].is_unique
    merged.to_csv(PROC_DIR / "cleaned_weather.csv", index=False)

    feats = build_features(om)                                          # features from ERA5 ONLY
    label = make_label(gs.set_index("date")["station_heat_index_c"])   # labels from station ONLY
    feats[TARGET] = feats["date"].map(label)
    ds = feats.merge(gs[["date", "station_heat_index_c"]], on="date", how="inner")  # keeps dates in both sources
    ds = ds.dropna(subset=[TARGET]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    ds[TARGET] = ds[TARGET].astype(int)
    ds.to_csv(PROC_DIR / "heatwave_dataset.csv", index=False)

    X, y = ds[ALL_FEATURES], ds[TARGET]
    report = {"open_meteo": log_om, "gsod": log_gs, "merged_rows": len(merged),
              "final_rows": len(ds), "n_features": X.shape[1],
              "date_min": str(ds["date"].min().date()), "date_max": str(ds["date"].max().date()),
              "missing_cells_in_features": int(X.isnull().sum().sum()),
              "class_counts": {int(k): int(v) for k, v in y.value_counts().items()},
              "positive_rate": float(y.mean())}
    # required checkpoint
    assert len(ds) >= 5000, "need >= 5,000 rows"
    assert X.shape[1] >= 15, "need >= 15 features"
    assert y.nunique() == 2, "target must contain both classes"
    assert ds["date"].is_unique
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return ds


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(PROC_DIR / "heatwave_dataset.csv", parse_dates=["date"])


if __name__ == "__main__":
    build_dataset()
