"""
Backtest + live day-ahead forecast of Germany's electricity demand.

  python src/pipeline.py            # backtest on the last ~16 months and write the dashboard data
Outputs: reports/metrics.json, reports/backtest_hourly.csv, docs/data/dashboard.json
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURES, make_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "smard_hourly.csv"
TEST_START = pd.Timestamp("2025-07-01", tz="Europe/Berlin")
CALIB_START = pd.Timestamp("2025-01-01", tz="Europe/Berlin")  # last 6 months before the test: calibrates the 80% band
PARAMS = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=48, min_samples_leaf=40,
              l2_regularization=1.0, categorical_features=[0, 1, 2], random_state=42)


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def model(loss="squared_error", q=None):
    kw = dict(PARAMS)
    if loss == "quantile":
        kw.update(loss="quantile", quantile=q)
    return HistGradientBoostingRegressor(**kw)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["time"], index_col="time")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def full_days(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only complete local days with load data (SMARD publishes with a short delay)."""
    loc = df["load"].dropna()
    days = pd.Series(loc.index.tz_convert("Europe/Berlin").date, index=loc.index)
    counts = days.value_counts()
    last_full = max(d for d, c in counts.items() if c >= 23)
    return df[df.index.tz_convert("Europe/Berlin").date <= last_full]


def residual_quantiles(res: pd.Series) -> pd.DataFrame:
    """10% / 90% quantiles of forecast errors for each local hour of the day."""
    h = res.index.tz_convert("Europe/Berlin").hour
    return res.groupby(h).quantile([0.1, 0.9]).unstack().rename(columns={0.1: "q10", 0.9: "q90"})


def backtest(feat: pd.DataFrame):
    data = feat.dropna(subset=FEATURES + ["target"])
    train = data[data.index < TEST_START.tz_convert("UTC")]
    test = data[data.index >= TEST_START.tz_convert("UTC")]
    # split-conformal 80% band: residual quantiles per hour on a calibration window the model has not seen
    fit = train[train.index < CALIB_START.tz_convert("UTC")]
    cal = train[train.index >= CALIB_START.tz_convert("UTC")]
    cal_res = cal["target"] - model().fit(fit[FEATURES], fit["target"]).predict(cal[FEATURES])
    q = residual_quantiles(cal_res)
    m = model().fit(train[FEATURES], train["target"])
    pred = m.predict(test[FEATURES])
    hours = test.index.tz_convert("Europe/Berlin").hour
    out = pd.DataFrame({"actual": test["target"], "model": pred,
                        "p10": pred + q.loc[hours, "q10"].to_numpy(), "p90": pred + q.loc[hours, "q90"].to_numpy(),
                        "naive_week": test["lag_168"], "naive_day": test["lag_24"]}, index=test.index)
    y = out["actual"]
    res = {name: {"MAPE_%": round(mape(y, out[col]), 2), "MAE_MW": round(float(np.mean(np.abs(y - out[col]))), 0)}
           for name, col in (("Gradient boosting (this project)", "model"),
                             ("Naive: same hour last week", "naive_week"),
                             ("Naive: same hour yesterday", "naive_day"))}
    cover = float(((y >= out["p10"]) & (y <= out["p90"])).mean() * 100)
    local = out.index.tz_convert("Europe/Berlin")
    by_hour = out.assign(h=local.hour).groupby("h").apply(lambda g: mape(g["actual"], g["model"]), include_groups=False)
    by_dow = out.assign(d=local.dayofweek).groupby("d").apply(lambda g: mape(g["actual"], g["model"]), include_groups=False)
    best_naive = min(res["Naive: same hour last week"]["MAPE_%"], res["Naive: same hour yesterday"]["MAPE_%"])
    metrics = {
        "train_period": [str(train.index.min().tz_convert("Europe/Berlin").date()), str(train.index.max().tz_convert("Europe/Berlin").date())],
        "test_period": [str(test.index.min().tz_convert("Europe/Berlin").date()), str(test.index.max().tz_convert("Europe/Berlin").date())],
        "test_hours": int(len(test)), "models": res,
        "improvement_vs_best_naive_%": round((1 - res["Gradient boosting (this project)"]["MAPE_%"] / best_naive) * 100, 1),
        "interval_80_coverage_%": round(cover, 1),
        "mape_by_hour": [round(v, 2) for v in by_hour.tolist()],
        "mape_by_weekday": [round(v, 2) for v in by_dow.tolist()],
    }
    return out, metrics


def forecast_next_day(df: pd.DataFrame, feat: pd.DataFrame, recent_res: pd.Series):
    """Fit on all history and forecast the 24 hours after the last complete day."""
    data = feat.dropna(subset=FEATURES + ["target"])
    last_local = df["load"].dropna().index.max().tz_convert("Europe/Berlin")
    target_day = (last_local + timedelta(hours=1)).normalize()
    hours = pd.date_range(target_day, periods=25, freq="h", tz="Europe/Berlin")[:24]
    hours = pd.DatetimeIndex([h for h in hours if h.date() == target_day.date()]).tz_convert("UTC")
    ext = pd.concat([df[["load"]], pd.DataFrame({"load": np.nan}, index=hours)])
    ext = ext[~ext.index.duplicated(keep="first")].sort_index()
    f = make_features(ext).loc[hours]
    pred = model().fit(data[FEATURES], data["target"]).predict(f[FEATURES])
    q = residual_quantiles(recent_res)  # out-of-sample errors of the last 180 days
    h = hours.tz_convert("Europe/Berlin").hour
    fc = pd.DataFrame({"model": pred, "p10": pred + q.loc[h, "q10"].to_numpy(),
                       "p90": pred + q.loc[h, "q90"].to_numpy()}, index=hours)
    return target_day.date(), last_local, fc


def main():
    df = full_days(load_data())
    feat = make_features(df)
    out, metrics = backtest(feat)
    (ROOT / "reports").mkdir(exist_ok=True)
    out.round(0).to_csv(ROOT / "reports" / "backtest_hourly.csv", index_label="time")
    recent_res = (out["actual"] - out["model"]).tail(180 * 24)
    day, issued, fc = forecast_next_day(df, feat, recent_res)
    metrics["live_forecast_for"] = str(day)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    loc = lambda idx: [t.tz_convert("Europe/Berlin").strftime("%Y-%m-%dT%H:%M") for t in idx]
    recent = out.tail(14 * 24)
    last = df.tail(14 * 24)
    gw = lambda s: [None if pd.isna(v) else round(float(v) / 1000, 2) for v in s]
    daily_mix = (df[["load", "solar", "wind_onshore", "wind_offshore"]].tail(90 * 24)
                 .groupby(df.tail(90 * 24).index.tz_convert("Europe/Berlin").date).sum())
    price = df["price"].dropna()
    dash = {
        "issued_with_data_until": issued.strftime("%Y-%m-%d %H:%M"),
        "forecast_day": str(day),
        "forecast": {"time": loc(fc.index), "gw": gw(fc["model"]), "p10": gw(fc["p10"]), "p90": gw(fc["p90"])},
        "backtest_recent": {"time": loc(recent.index), "actual": gw(recent["actual"]), "model": gw(recent["model"]), "p10": gw(recent["p10"]), "p90": gw(recent["p90"])},
        "recent_actual": {"time": loc(last.index), "gw": gw(last["load"])},
        "renewable_share_90d": {"date": [str(d) for d in daily_mix.index],
                                "share": [round(float((r.solar + r.wind_onshore + r.wind_offshore) / r.load * 100), 1)
                                          for r in daily_mix.itertuples()]},
        "price_recent": {"time": loc(price.tail(14 * 24).index), "eur_mwh": [round(float(v), 1) for v in price.tail(14 * 24)]},
        "metrics": metrics,
    }
    (ROOT / "docs" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "data" / "dashboard.json").write_text(json.dumps(dash, separators=(",", ":")))
    print(json.dumps({k: v for k, v in metrics.items() if not k.startswith("mape_by")}, indent=2))
    print("forecast for", day, "peak", max(dash["forecast"]["gw"]), "GW")


if __name__ == "__main__":
    main()
