"""Module 4 - Target Builder.

Single source of truth for the two prediction targets.

  * AAR proxy  - completed arrivals per (airport, hour). The CSV has no published
                 acceptance rate, so realised arrival throughput is the stated
                 proxy. Built on a continuous grid (hours with no arrivals = 0).
                 NOTE: the provided file is a flight SAMPLE, so counts are small;
                 switch config.BUCKET to '3h' for a denser signal if desired.
  * severe     - per-flight binary label: arrival delay > SEVERE_DELAY_MIN.

PortSync ancestor: predicted variables.
"""
import pandas as pd

from . import config


def _focus_arrivals(flights: pd.DataFrame, airports) -> pd.DataFrame:
    return flights[
        flights["DEST"].isin(airports)
        & (flights["CANCELLED"] != 1)
        & (flights["DIVERTED"] != 1)
    ].copy()


def build_aar_targets(flights: pd.DataFrame, airports=config.AIRPORTS,
                      bucket=config.BUCKET) -> pd.DataFrame:
    """Hourly AAR proxy per focus airport, on a continuous (0-filled) grid."""
    f = _focus_arrivals(flights, airports).dropna(subset=["land_dt"])
    f["dt"] = f["land_dt"].dt.floor(bucket)
    g = (f.groupby(["DEST", "dt"]).size().rename("aar").reset_index()
           .rename(columns={"DEST": "airport"}))
    out = []
    for ap, gg in g.groupby("airport"):
        idx = pd.date_range(gg["dt"].min(), gg["dt"].max(), freq=bucket)
        gg = gg.set_index("dt").reindex(idx)
        gg["aar"] = gg["aar"].fillna(0.0)
        gg["airport"] = ap
        gg.index.name = "dt"
        out.append(gg.reset_index())
    res = pd.concat(out, ignore_index=True)
    res["aar"] = res["aar"].astype(float)
    return res


def build_risk_labels(flights: pd.DataFrame, airports=config.AIRPORTS,
                      thr=config.SEVERE_DELAY_MIN) -> pd.DataFrame:
    """Per-flight severe-arrival-delay labels for flights arriving at focus airports.
    Keeps the raw columns the feature builder turns into leakage-safe schedule,
    carrier and origin-weather features."""
    f = _focus_arrivals(flights, airports).dropna(subset=["ARR_DELAY", "sched_arr_dt"])
    f["severe"] = (f["ARR_DELAY"] > thr).astype(int)
    f["airport"] = f["DEST"]
    f["dt"] = f["sched_arr_dt"]
    keep = (["airport", "dt", "severe", "sched_arr_dt", "ARR_DELAY",
             "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN",
             "CRS_ELAPSED_TIME", "CRS_DEP_TIME", "CRS_ARR_TIME", "DISTANCE"]
            + [f"{v}_ORIGIN" for v in config.WEATHER_VARS])
    return f[[c for c in keep if c in f.columns]].reset_index(drop=True)
