"""Module 1 - Ingest & Integration.

Loads the raw flight+weather CSV, parses the HHMM clock fields into real
datetimes, and emits two clean tables:

  * flights        - one row per flight with parsed sched/actual datetimes
  * airport-hour   - median weather per (airport, hour) on a continuous grid

PortSync ancestor: Terminal State Engine (clean, timestamp, validate, combine).
"""
import pandas as pd
import numpy as np

from . import config


def _hhmm_to_minutes(s: pd.Series) -> pd.Series:
    """Convert a HHMM clock field (e.g. 1548.0) to minutes since midnight."""
    v = pd.to_numeric(s, errors="coerce")
    hh = (v // 100).clip(upper=23)          # 2400 -> treat as 23:xx-safe
    mm = (v % 100).clip(upper=59)
    return hh * 60 + mm


def _build_dt(date: pd.Series, minutes: pd.Series, cross_ref: pd.Series | None = None) -> pd.Series:
    """FL_DATE + minutes-since-midnight. If `cross_ref` is given and the time is
    earlier than it (e.g. arrival before departure), the event is next day."""
    dt = date + pd.to_timedelta(minutes, unit="m")
    if cross_ref is not None:
        plus = (minutes < cross_ref).fillna(False)
        dt = dt + pd.to_timedelta(plus.astype(float) * 1440, unit="m")
    return dt


def load_flights(path=config.DATA_CSV, airports=None) -> pd.DataFrame:
    """Load the CSV and attach parsed datetimes. `airports` optionally restricts
    to flights that touch those airports (as origin or destination)."""
    df = pd.read_csv(path, low_memory=False)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    dep_m = _hhmm_to_minutes(df["DEP_TIME"])
    crs_dep_m = _hhmm_to_minutes(df["CRS_DEP_TIME"])
    arr_m = _hhmm_to_minutes(df["ARR_TIME"])
    crs_arr_m = _hhmm_to_minutes(df["CRS_ARR_TIME"])
    won_m = _hhmm_to_minutes(df["WHEELS_ON"])

    df["sched_dep_dt"] = _build_dt(df["FL_DATE"], crs_dep_m)
    df["sched_arr_dt"] = _build_dt(df["FL_DATE"], crs_arr_m, crs_dep_m)
    df["act_arr_dt"] = _build_dt(df["FL_DATE"], arr_m, dep_m)
    # Actual landing time: prefer WHEELS_ON (runway), fall back to gate ARR_TIME.
    won_dt = _build_dt(df["FL_DATE"], won_m, dep_m)
    df["land_dt"] = won_dt.fillna(df["act_arr_dt"])

    if airports is not None:
        df = df[df["ORIGIN"].isin(airports) | df["DEST"].isin(airports)].copy()
    return df


def build_airport_hour_weather(df: pd.DataFrame, airports=config.AIRPORTS) -> pd.DataFrame:
    """Build a continuous hourly weather series per airport by pooling the
    origin-role and destination-role observations attached to each flight."""
    frames = []
    for role, dt_col in [("ORIGIN", "sched_dep_dt"), ("DEST", "sched_arr_dt")]:
        suff = "_" + role
        rename = {f"{v}{suff}": v for v in config.WEATHER_VARS}
        sub = df[[role, dt_col] + list(rename)].rename(
            columns={role: "airport", dt_col: "dt", **rename}
        )
        frames.append(sub)

    w = pd.concat(frames, ignore_index=True)
    w = w[w["airport"].isin(airports)].dropna(subset=["dt"]).copy()
    w["dt"] = w["dt"].dt.floor("h")
    agg = w.groupby(["airport", "dt"], as_index=False)[config.WEATHER_VARS].median()

    out = []
    for ap, g in agg.groupby("airport"):
        idx = pd.date_range(g["dt"].min(), g["dt"].max(), freq="h")
        g = g.set_index("dt").reindex(idx)
        g[config.WEATHER_VARS] = (
            g[config.WEATHER_VARS].interpolate(limit=3).ffill(limit=6).bfill(limit=3)
        )
        g["airport"] = ap
        g.index.name = "dt"
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":  # quick smoke test
    f = load_flights(airports=config.AIRPORTS)
    w = build_airport_hour_weather(f)
    print("flights:", len(f), "| airport-hours:", len(w))
    print(w.head())
