"""Humidity and heat-index calculations (NOAA/NWS Rothfusz regression)."""
import numpy as np


def rh_from_dewpoint(t_c, td_c):
    """Relative humidity (%) from air temp and dew point (Celsius), Magnus formula."""
    t_c, td_c = np.asarray(t_c, float), np.asarray(td_c, float)
    a, b = 17.625, 243.04
    rh = 100 * np.exp(a * td_c / (b + td_c)) / np.exp(a * t_c / (b + t_c))
    return np.clip(rh, 0, 100)


def heat_index_c(t_c, rh):
    """NWS heat index in Celsius from air temperature (C) and RH (%).
    Steadman simple formula, switching to the Rothfusz regression (+ NWS adjustments)
    when the simple value averaged with T is >= 80 F.
    https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml"""
    T = np.asarray(t_c, float) * 9 / 5 + 32
    R = np.asarray(rh, float)
    simple = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + R * 0.094)
    hi = (-42.379 + 2.04901523 * T + 10.14333127 * R - 0.22475541 * T * R
          - 0.00683783 * T * T - 0.05481717 * R * R + 0.00122874 * T * T * R
          + 0.00085282 * T * R * R - 0.00000199 * T * T * R * R)
    with np.errstate(invalid="ignore"):
        adj1 = np.where((R < 13) & (T >= 80) & (T <= 112),
                        -((13 - R) / 4) * np.sqrt(np.clip((17 - np.abs(T - 95)) / 17, 0, None)), 0.0)
    adj2 = np.where((R > 85) & (T >= 80) & (T <= 87), ((R - 85) / 10) * ((87 - T) / 5), 0.0)
    hi = hi + adj1 + adj2
    hi_f = np.where((simple + T) / 2 >= 80, hi, simple)
    out = (hi_f - 32) * 5 / 9
    return np.where(np.isnan(T) | np.isnan(R), np.nan, out)
