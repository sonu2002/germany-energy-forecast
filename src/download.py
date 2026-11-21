"""
Downloads hourly data for Germany from SMARD.de (Bundesnetzagentur, licence CC BY 4.0):
  - 410  : electricity consumption / grid load (MWh per hour)
  - 4169 : day-ahead wholesale price DE-LU (EUR/MWh)
  - 4068 : solar generation, 4067: wind onshore, 1225: wind offshore (MWh)
Incremental: re-downloads only the last 3 weeks if data/smard_hourly.csv already exists.
Standard library + pandas only.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "smard_hourly.csv"
BASE = "https://www.smard.de/app/chart_data/{f}/DE/{name}"
SERIES = {"load": 410, "price": 4169, "solar": 4068, "wind_onshore": 4067, "wind_offshore": 1225}
START = pd.Timestamp("2022-12-26", tz="Europe/Berlin")


def get(url: str, tries: int = 4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "germany-energy-forecast (github)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def fetch_series(filter_id: int, since: pd.Timestamp) -> pd.Series:
    index = get(BASE.format(f=filter_id, name="index_hour.json"))["timestamps"]
    since_ms = int(since.timestamp() * 1000) - 7 * 24 * 3600 * 1000
    rows = []
    for ts in index:
        if ts < since_ms:
            continue
        chunk = get(BASE.format(f=filter_id, name=f"{filter_id}_DE_hour_{ts}.json"))
        rows += [(t, v) for t, v in chunk["series"] if v is not None]
    s = pd.Series(dict(rows), dtype="float64")
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    return s.sort_index()


def main() -> pd.DataFrame:
    old = pd.read_csv(OUT, parse_dates=["time"], index_col="time") if OUT.is_file() else None
    since = (old.index.max() - pd.Timedelta(days=21)) if old is not None and len(old) else START
    frames = {}
    for name, fid in SERIES.items():
        frames[name] = fetch_series(fid, pd.Timestamp(since))
        print(f"{name:14s} {len(frames[name]):6d} rows  last={frames[name].index.max()}")
    new = pd.DataFrame(frames)
    new.index.name = "time"
    df = new if old is None else pd.concat([old[old.index < new.index.min()], new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index >= START.tz_convert("UTC")]
    OUT.parent.mkdir(exist_ok=True)
    df.round(2).to_csv(OUT)
    print(f"saved {len(df)} hourly rows → {OUT.relative_to(ROOT)}  ({df.index.min()} … {df.index.max()})")
    return df


if __name__ == "__main__":
    main()
