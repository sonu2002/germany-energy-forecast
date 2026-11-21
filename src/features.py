"""Feature engineering for day-ahead load forecasting (no future information leaks).

Setting: at midnight (Europe/Berlin) after day D we forecast all 24 hours of day D+1.
Every feature for a target hour t on day D+1 may only use data up to the end of day D
(lag features are >= 24 h old; "prev_day_*" are statistics of day D). Nothing from day D+1 itself
is used. tests/test_features.py checks this by changing day D+1 and asserting the features stay the same.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

MIN_LAG_HOURS = 24


def easter(year: int) -> date:
    """Gregorian Easter Sunday (Anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def german_holidays(year: int) -> set[date]:
    """Nationwide public holidays in Germany."""
    e = easter(year)
    return {date(year, 1, 1), e - timedelta(days=2), e + timedelta(days=1), date(year, 5, 1),
            e + timedelta(days=39), e + timedelta(days=50), date(year, 10, 3),
            date(year, 12, 25), date(year, 12, 26)}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: hourly frame indexed by UTC time with a 'load' column (MWh). Returns features + target."""
    local = df.index.tz_convert("Europe/Berlin")
    hol = set().union(*(german_holidays(y) for y in range(local.year.min() - 1, local.year.max() + 2)))
    d = pd.DataFrame(index=df.index)
    d["hour"] = local.hour
    d["dow"] = local.dayofweek
    d["month"] = local.month
    doy = local.dayofyear.to_numpy()
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    days = pd.Series(local.date, index=df.index)
    d["holiday"] = days.map(lambda x: x in hol).astype(int)
    d["holiday_next"] = days.map(lambda x: (x + timedelta(days=1)) in hol).astype(int)
    d["holiday_prev"] = days.map(lambda x: (x - timedelta(days=1)) in hol).astype(int)
    d["christmas_period"] = (((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day <= 6))).astype(int)
    load = df["load"]
    for lag in (24, 48, 168, 336):
        d[f"lag_{lag}"] = load.shift(lag, freq="h").reindex(df.index)
    # statistics of the last fully known day (day D) — shift by whole days only
    daily = load.groupby(days).agg(["mean", "max", "min"])
    prev = daily.shift(1)  # value of day D attached to day D+1
    for col in ("mean", "max", "min"):
        d[f"prev_day_{col}"] = days.map(prev[col]).to_numpy()
    d["prev_week_mean"] = days.map(daily["mean"].rolling(7).mean().shift(1)).to_numpy()
    d["target"] = load
    return d


FEATURES = ["hour", "dow", "month", "doy_sin", "doy_cos", "holiday", "holiday_next", "holiday_prev",
            "christmas_period", "lag_24", "lag_48", "lag_168", "lag_336",
            "prev_day_mean", "prev_day_max", "prev_day_min", "prev_week_mean"]
