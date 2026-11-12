import numpy as np, pandas as pd, pytest
from src.config import ALL_FEATURES, TARGET, PROC_DIR, BASE_FEATURES
from src.feature_engineering import build_features, make_label
from src.heat import heat_index_c, rh_from_dewpoint


def test_heat_index_matches_nws_table():
    # NWS table: 90 F at 70 % RH -> ~105 F
    hi_f = heat_index_c(np.array([(90 - 32) * 5 / 9]), np.array([70.0]))[0] * 9 / 5 + 32
    assert abs(hi_f - 105) < 1.5


def test_rh_bounds_and_saturation():
    assert rh_from_dewpoint(30, 30) == pytest.approx(100)
    assert 0 <= rh_from_dewpoint(40, 5) <= 100


def test_label_uses_calendar_dates_and_horizon():
    idx = pd.date_range("2024-06-01", periods=15)
    hi = pd.Series(30.0, index=idx); hi.iloc[8] = 45.0          # spike on day index 8
    lab = make_label(hi)
    assert lab.iloc[2] == 1 and lab.iloc[3] == 1 and lab.iloc[1] == 1 and lab.iloc[0] == 0   # t+5..t+7 contains index 8 for t=1,2,3
    assert np.isnan(lab.iloc[-1])


def test_label_handles_missing_rows():
    idx = pd.date_range("2024-06-01", periods=20).delete([8, 9])  # missing calendar dates, not consecutive rows
    hi = pd.Series(30.0, index=idx)
    lab = make_label(hi)
    assert np.isnan(lab.loc["2024-06-03"])                       # t+5..7 includes missing days and none are hot -> unknown


def test_no_future_leakage_in_features():
    rng = np.random.default_rng(0); n = 40
    d = pd.DataFrame({c: rng.normal(30, 3, n) for c in BASE_FEATURES}); d["date"] = pd.date_range("2024-01-01", periods=n)
    a = build_features(d)
    d2 = d.copy(); d2.loc[30:, BASE_FEATURES] += 50                # change the FUTURE
    b = build_features(d2)
    cols = [c for c in ALL_FEATURES if c != "season"]
    pd.testing.assert_frame_equal(a.loc[:29, cols], b.loc[:29, cols])


@pytest.mark.skipif(not (PROC_DIR / "heatwave_dataset.csv").exists(), reason="dataset not built yet")
def test_final_dataset_requirements():
    df = pd.read_csv(PROC_DIR / "heatwave_dataset.csv", parse_dates=["date"])
    assert len(df) >= 5000 and len(ALL_FEATURES) >= 15
    assert df["date"].is_unique and df["date"].is_monotonic_increasing
    assert set(df[TARGET].unique()) == {0, 1}
    assert df[TARGET].notna().all()
    assert df["relative_humidity_2m_mean"].dropna().between(0, 100).all()
    assert not np.isinf(df[[c for c in ALL_FEATURES if c != "season"]].to_numpy(float)).any()
