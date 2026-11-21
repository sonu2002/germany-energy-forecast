import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import FEATURES, easter, german_holidays, make_features  # noqa: E402
from pipeline import mape, residual_quantiles  # noqa: E402


def synthetic(days=40):
    idx = pd.date_range("2025-03-01", periods=days * 24, freq="h", tz="UTC")
    load = 50000 + 8000 * np.sin(2 * np.pi * (idx.hour - 6) / 24) + np.arange(len(idx))
    return pd.DataFrame({"load": load}, index=idx)


def test_easter_dates():
    assert easter(2024) == date(2024, 3, 31)
    assert easter(2025) == date(2025, 4, 20)
    assert easter(2026) == date(2026, 4, 5)


def test_holidays_contain_fixed_and_moving_days():
    h = german_holidays(2026)
    assert {date(2026, 10, 3), date(2026, 4, 3), date(2026, 5, 14), date(2026, 5, 25)} <= h
    assert len(h) == 9


def test_features_ignore_everything_from_the_forecast_day():
    df = synthetic()
    f = make_features(df)
    local = df.index.tz_convert("Europe/Berlin")
    last_day = local[-30].normalize()                       # a complete local day near the end
    day_mask = (local >= last_day) & (local < last_day + pd.Timedelta(days=1))
    t = df.index[day_mask][-1]                              # last hour of that day
    df2 = df.copy()
    df2.loc[day_mask & (df.index < t), "load"] += 99999     # change all earlier hours of the same day
    f2 = make_features(df2)
    cols = [c for c in FEATURES if c.startswith(("lag_", "prev_"))]
    pd.testing.assert_series_equal(f.loc[t, cols], f2.loc[t, cols])


def test_lag_24_is_same_hour_yesterday():
    df = synthetic()
    f = make_features(df)
    t = df.index[-1]
    assert f.loc[t, "lag_24"] == df.loc[t - pd.Timedelta(hours=24), "load"]


def test_mape_and_residual_quantiles():
    assert mape(np.array([100.0, 200.0]), np.array([110.0, 180.0])) == 10.0
    idx = pd.date_range("2025-01-01", periods=48 * 10, freq="h", tz="UTC")
    q = residual_quantiles(pd.Series(np.tile(np.arange(-5, 5), 48), index=idx))
    assert list(q.columns) == ["q10", "q90"] and len(q) == 24
    assert (q["q10"] < q["q90"]).all()
