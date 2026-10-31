"""Unit tests for the parts that must be correct: the leakage guard, the crosswind
geometry, time parsing, and the severe-delay label.

Run with pytest (`pytest -q`) or standalone (`python tests/test_pipeline.py`).
These use tiny synthetic frames - no CSV, no network.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aerocast import config, ingest, features, targets  # noqa: E402


def test_hhmm_to_minutes():
    m = ingest._hhmm_to_minutes(pd.Series([1548.0, 0.0, 905.0, np.nan]))
    assert m.iloc[0] == 15 * 60 + 48
    assert m.iloc[1] == 0
    assert m.iloc[2] == 9 * 60 + 5
    assert pd.isna(m.iloc[3])


def test_crosswind_geometry():
    """Wind aligned with the runway -> no crosswind; 90deg off -> pure crosswind."""
    w = pd.DataFrame({
        "airport": ["ATL", "ATL"],
        "dt": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
        "wdir": [80.0, 170.0],            # ATL heading is 80
        "wspd": [10.0, 10.0], "wspd_om": [10.0, 10.0],
        "wpgt": [12.0, 12.0], "wgust_om": [12.0, 12.0],
    })
    c = features.add_wind_components(w)
    assert abs(c.loc[0, "crosswind"]) < 1e-6          # aligned
    assert abs(c.loc[0, "headwind"] - 10.0) < 1e-6
    assert abs(c.loc[1, "crosswind"] - 10.0) < 1e-6   # perpendicular
    assert abs(c.loc[1, "headwind"]) < 1e-6


def test_lagged_features_are_leakage_safe():
    """A feature at hour h must equal the raw value at hour h - lead, never later."""
    dts = pd.date_range("2024-01-01", periods=48, freq="h")
    w = pd.DataFrame({
        "airport": "ATL", "dt": dts,
        "wdir": np.linspace(0, 359, 48), "wspd": np.arange(48) * 1.0,
        "wspd_om": np.arange(48) * 1.0, "wpgt": np.arange(48) * 1.5,
        "wgust_om": np.arange(48) * 1.5, "temp": 1.0, "prcp": 0.0,
        "pres": 1013.0, "snow": 0.0, "coco": 1.0,
    })
    feat = features.build_airport_hour_features(w, leads=[3]).sort_values("dt").reset_index(drop=True)
    base = features.add_wind_components(w).sort_values("dt").reset_index(drop=True)
    for i in range(3, 48):
        assert abs(feat.loc[i, "crosswind_lag3"] - base.loc[i - 3, "crosswind"]) < 1e-9
    # the first `lead` rows have no history -> must be NaN, never filled from the future
    assert feat.loc[:2, "crosswind_lag3"].isna().all()


def test_severe_label():
    n = 3
    df = pd.DataFrame({
        "DEST": ["ATL"] * n, "CANCELLED": [0] * n, "DIVERTED": [0] * n,
        "ARR_DELAY": [10.0, 50.0, -5.0],
        "sched_arr_dt": pd.to_datetime(["2024-01-01 10:00"] * n),
        "OP_UNIQUE_CARRIER": ["DL"] * n, "OP_CARRIER_FL_NUM": [1, 2, 3],
        "ORIGIN": ["JFK"] * n, "CRS_ELAPSED_TIME": [120] * n,
        "CRS_DEP_TIME": [800] * n, "CRS_ARR_TIME": [1000] * n, "DISTANCE": [760] * n,
    })
    for v in config.WEATHER_VARS:
        df[f"{v}_ORIGIN"] = [0.0] * n
    lbl = targets.build_risk_labels(df)
    assert list(lbl["severe"]) == [0, 1, 0]   # only the 50-min delay is severe (>45)


def test_origin_weather_masked_for_short_flights():
    """Origin weather is only a feature when the flight is >= MIN_LEAD long,
    otherwise departure weather would not yet be observed at forecast time."""
    labels = pd.DataFrame({
        "airport": ["ATL", "ATL"], "dt": pd.to_datetime(["2024-01-01 10:00"] * 2),
        "severe": [0, 1], "sched_arr_dt": pd.to_datetime(["2024-01-01 10:00"] * 2),
        "ARR_DELAY": [0.0, 60.0], "OP_UNIQUE_CARRIER": ["DL", "DL"],
        "OP_CARRIER_FL_NUM": [1, 2], "ORIGIN": ["JFK", "MIA"],
        "CRS_ELAPSED_TIME": [60, 300],      # 1h (short) vs 5h (long); MIN_LEAD=3h
        "CRS_DEP_TIME": [800, 500], "CRS_ARR_TIME": [900, 1000], "DISTANCE": [500, 1500],
        "temp_ORIGIN": [20.0, 25.0], "prcp_ORIGIN": [0, 0], "wspd_ORIGIN": [5, 5],
        "wpgt_ORIGIN": [8, 8], "pres_ORIGIN": [1013, 1013], "snow_ORIGIN": [0, 0],
        "coco_ORIGIN": [1, 1],
    })
    feat_ah = pd.DataFrame({"airport": ["ATL"], "dt": pd.to_datetime(["2024-01-01 10:00"])})
    d = features.assemble_risk_dataset(feat_ah, labels)
    assert pd.isna(d.loc[0, "o_temp"])          # short flight -> masked
    assert abs(d.loc[1, "o_temp"] - 25.0) < 1e-9  # long flight -> kept


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
