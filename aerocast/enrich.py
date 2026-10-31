"""Module 2 - Weather Enrichment.

The provided CSV has wind SPEED/GUST but no wind DIRECTION, which the crosswind
calculation needs. This module adds real hourly wind direction (+ a consistent
wind speed and gust) per airport-hour.

  Primary : Open-Meteo historical archive (free, no API key) - real observations.
  Fallback: Meteostat, then a deterministic SYNTHETIC wind direction so the
            pipeline still runs fully offline.

Every row carries `wdir_source` so the dashboard can disclose which was used.
This module is also the "automatic public-dataset processing" deliverable: it
downloads and caches a public dataset with no manual intervention.

PortSync ancestor: external signals ingestion.
"""
import numpy as np
import pandas as pd

from . import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _synthetic_wdir(airport: str, dt_index: pd.DatetimeIndex) -> np.ndarray:
    """Deterministic seasonal prevailing wind + seeded noise - a clearly-labelled
    placeholder so crosswind features exist offline."""
    rng = np.random.default_rng(abs(hash((airport, config.RANDOM_SEED))) % (2**32))
    month = dt_index.month.to_numpy()
    prevail = np.where(np.isin(month, [11, 12, 1, 2]), 320.0,
              np.where(np.isin(month, [6, 7, 8]), 190.0, 250.0))
    return (prevail + rng.normal(0.0, 45.0, size=len(dt_index))) % 360.0


def _fetch_open_meteo(airport, start, end):
    """Full-range hourly wind from Open-Meteo, as a (dt, wdir, wspd_om, wgust_om)
    DataFrame, or None on any failure."""
    try:
        import requests
    except Exception:
        return None
    if airport not in config.STATION_COORDS:
        return None
    lat, lon, _alt = config.STATION_COORDS[airport]
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": str(pd.Timestamp(start).date()),
        "end_date": str(pd.Timestamp(end).date()),
        "hourly": "wind_direction_10m,wind_speed_10m,wind_gusts_10m",
        "wind_speed_unit": "kmh",
        "timezone": config.STATION_TZ.get(airport, "UTC"),
    }
    try:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=90)
        resp.raise_for_status()
        h = resp.json().get("hourly", {})
        if not h.get("time"):
            return None
        return pd.DataFrame({
            "dt": pd.to_datetime(h["time"]),
            "wdir": h.get("wind_direction_10m"),
            "wspd_om": h.get("wind_speed_10m"),
            "wgust_om": h.get("wind_gusts_10m"),
        })
    except Exception:
        return None


def _fetch_meteostat_wdir(airport, dts):
    try:
        from meteostat import Hourly, Point
    except Exception:
        return None
    if airport not in config.STATION_COORDS:
        return None
    try:
        lat, lon, alt = config.STATION_COORDS[airport]
        raw = Hourly(Point(lat, lon, alt),
                     pd.Timestamp(dts.min()).to_pydatetime(),
                     pd.Timestamp(dts.max()).to_pydatetime()).fetch()
        if raw.empty or "wdir" not in raw:
            return None
        s = raw["wdir"].reindex(pd.DatetimeIndex(dts)).interpolate(limit=6).ffill().bfill()
        return s if s.notna().any() else None
    except Exception:
        return None


def add_wind_direction(weather_hourly: pd.DataFrame) -> pd.DataFrame:
    """Join real hourly `wdir` (+ `wspd_om`, `wgust_om`) onto the airport-hour
    table, caching the public fetch per airport so re-runs are instant."""
    out = []
    for ap, g in weather_hourly.groupby("airport"):
        g = g.sort_values("dt").reset_index(drop=True).copy()
        dts = pd.DatetimeIndex(g["dt"])

        om, src = None, "synthetic"
        cache = config.ARTIFACTS / f"wind_{ap}.csv"
        if cache.exists():
            try:
                om, src = pd.read_csv(cache, parse_dates=["dt"]), "open-meteo(cache)"
            except Exception:
                om = None
        if om is None:
            om = _fetch_open_meteo(ap, dts.min(), dts.max())
            if om is not None:
                src = "open-meteo"
                try:
                    om.to_csv(cache, index=False)
                except Exception:
                    pass

        if om is not None:
            m = g[["dt"]].merge(om, on="dt", how="left")
            for c in ("wdir", "wspd_om", "wgust_om"):
                g[c] = pd.to_numeric(m[c], errors="coerce").interpolate(limit=6).ffill().bfill().to_numpy()
            if g["wdir"].isna().any():
                syn = _synthetic_wdir(ap, dts)
                g["wdir"] = g["wdir"].fillna(pd.Series(syn, index=g.index))
            g["wdir_source"] = src
        else:
            fetched = _fetch_meteostat_wdir(ap, dts)
            if fetched is not None:
                g["wdir"], g["wdir_source"] = fetched.to_numpy(), "meteostat"
            else:
                g["wdir"], g["wdir_source"] = _synthetic_wdir(ap, dts), "synthetic"
            g["wspd_om"] = np.nan
            g["wgust_om"] = np.nan
        out.append(g)

    res = pd.concat(out, ignore_index=True)
    try:
        res.to_csv(config.ARTIFACTS / "weather_enriched.csv", index=False)
    except Exception:
        pass
    return res


if __name__ == "__main__":
    from . import ingest
    f = ingest.load_flights(airports=config.AIRPORTS)
    w = ingest.build_airport_hour_weather(f)
    e = add_wind_direction(w)
    print(e.groupby(["airport", "wdir_source"]).size())
    print(e[["airport", "dt", "wdir", "wspd_om", "wgust_om"]].head())
