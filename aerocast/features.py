"""Module 3 - Feature Engineering (leakage-safe).

Two jobs:
  1. Turn wind speed/gust + direction + runway heading into crosswind / headwind
     / tailwind components (the "runway directionality" the brief demands).
  2. Build an airport-hour feature frame where every weather feature is the value
     KNOWN at (target_hour - lead), for lead in LEADS. Because the weather series
     is on a continuous hourly grid, shifting by L rows == shifting by L hours, so
     a feature at row h genuinely only sees data from h-L or earlier. That is the
     data-leakage guard.

PortSync ancestor: State Engine + Forecasting inputs.
"""
import numpy as np
import pandas as pd

from . import config

# Weather/derived columns we lag. (coco is a categorical condition code kept numeric.)
_LAG_VARS = ["temp", "prcp", "wspd", "wpgt", "pres", "snow", "coco",
             "wspd_om", "wgust_om",
             "crosswind", "headwind", "tailwind", "crosswind_gust"]


def add_wind_components(w: pd.DataFrame) -> pd.DataFrame:
    """Add crosswind/headwind/tailwind (and gust crosswind) vs runway heading.

    theta = wind_dir - runway_heading
      crosswind = wind_speed * |sin(theta)|
      headwind  = wind_speed * cos(theta);  tailwind = max(0, -headwind)

    Uses the Open-Meteo wind speed/gust (physically consistent with its wdir);
    falls back to the CSV's own wind speed where enrichment was unavailable.
    """
    w = w.copy()
    heading = w["airport"].map(config.RUNWAY_HEADING)
    theta = np.deg2rad(w["wdir"] - heading)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    spd = w["wspd_om"].fillna(w["wspd"]) if "wspd_om" in w.columns else w["wspd"]
    gst = w["wgust_om"].fillna(w["wpgt"]) if "wgust_om" in w.columns else w["wpgt"]
    w["crosswind"] = spd * np.abs(sin_t)
    headwind = spd * cos_t
    w["headwind"] = headwind
    w["tailwind"] = np.clip(-headwind, 0.0, None)
    w["crosswind_gust"] = gst.fillna(spd) * np.abs(sin_t)
    return w


def build_airport_hour_features(weather_enriched: pd.DataFrame,
                                leads=config.LEADS) -> pd.DataFrame:
    """Return an airport-hour frame: lagged weather/crosswind at each lead plus
    temporal features derived from the target hour. Key: (airport, dt)."""
    w = add_wind_components(weather_enriched).sort_values(["airport", "dt"])
    parts = []
    for ap, g in w.groupby("airport"):
        g = g.set_index("dt").sort_index()
        base = g[["airport"]].copy()
        for L in leads:
            shifted = g[_LAG_VARS].shift(L)           # weather known L hours earlier
            shifted.columns = [f"{c}_lag{L}" for c in _LAG_VARS]
            base = base.join(shifted)
        parts.append(base)
    out = pd.concat(parts).reset_index()              # index name is 'dt'

    out["hour"] = out["dt"].dt.hour
    out["dow"] = out["dt"].dt.dayofweek
    out["month"] = out["dt"].dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


def add_rolling_throughput(aar_df: pd.DataFrame,
                           min_lead=config.MIN_LEAD) -> pd.DataFrame:
    """Recent arrival throughput as a feature, lagged by MIN_LEAD hours so it is
    only ever computed from data available before the forecast is issued."""
    parts = []
    for ap, g in aar_df.sort_values("dt").groupby("airport"):
        g = g.set_index("dt").sort_index()
        roll = g["aar"].rolling(3, min_periods=1).mean().shift(min_lead)
        g["aar_roll3_lag"] = roll
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric model inputs: everything except keys/target/metadata."""
    drop = {"dt", "airport", "aar", "severe", "wdir_source",
            "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN",
            "sched_arr_dt", "ARR_DELAY"}
    cols = [c for c in df.columns if c not in drop and
            pd.api.types.is_numeric_dtype(df[c])]
    return cols


def assemble_aar_dataset(feat_ah: pd.DataFrame, aar_df: pd.DataFrame) -> pd.DataFrame:
    """Join airport-hour features + rolling throughput onto the AAR target."""
    roll = add_rolling_throughput(aar_df)
    d = aar_df.merge(feat_ah, on=["airport", "dt"], how="left")
    d = d.merge(roll[["airport", "dt", "aar_roll3_lag"]], on=["airport", "dt"], how="left")
    # add an airport one-hot so a single pooled model stays airport-aware
    for ap in config.AIRPORTS:
        d[f"is_{ap}"] = (d["airport"] == ap).astype(int)
    return d


def assemble_risk_dataset(feat_ah: pd.DataFrame, labels: pd.DataFrame,
                          top_carriers: int = 12) -> pd.DataFrame:
    """Join DEST airport-hour (lagged) features onto each flight's severe label,
    plus leakage-safe schedule, origin-weather and carrier features.

    All added features are knowable before the flight: schedule times, route, the
    carrier, and ORIGIN weather - the last masked out when the flight is shorter
    than MIN_LEAD (otherwise departure weather would not yet be observed at the
    forecast time t = arrival - lead)."""
    d = labels.copy()
    d["dt"] = d["dt"].dt.floor("h")
    d = d.merge(feat_ah, on=["airport", "dt"], how="left")

    # schedule features (fixed well in advance)
    d["elapsed"] = pd.to_numeric(d.get("CRS_ELAPSED_TIME"), errors="coerce")
    d["dep_hour"] = pd.to_numeric(d.get("CRS_DEP_TIME"), errors="coerce") // 100
    d["arr_hour"] = pd.to_numeric(d.get("CRS_ARR_TIME"), errors="coerce") // 100
    d["distance"] = pd.to_numeric(d.get("DISTANCE"), errors="coerce")

    # origin-side weather, masked when the flight is shorter than the min lead
    minl = config.MIN_LEAD * 60
    for v in config.WEATHER_VARS:
        src = pd.to_numeric(d.get(f"{v}_ORIGIN"), errors="coerce")
        d[f"o_{v}"] = np.where(d["elapsed"] >= minl, src, np.nan)

    # carrier one-hot (top-N by frequency; identity encoding, leakage-free)
    if "OP_UNIQUE_CARRIER" in d.columns:
        for c in d["OP_UNIQUE_CARRIER"].value_counts().head(top_carriers).index:
            d[f"car_{c}"] = (d["OP_UNIQUE_CARRIER"] == c).astype(int)

    for ap in config.AIRPORTS:
        d[f"is_{ap}"] = (d["airport"] == ap).astype(int)

    # drop raw columns so only derived, leakage-safe features remain
    drop_raw = (["CRS_ELAPSED_TIME", "CRS_DEP_TIME", "CRS_ARR_TIME", "DISTANCE"]
                + [f"{v}_ORIGIN" for v in config.WEATHER_VARS])
    return d.drop(columns=[c for c in drop_raw if c in d.columns])
