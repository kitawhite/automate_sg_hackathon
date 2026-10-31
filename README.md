# AeroCast AI — Predictive Airport Capacity & Delay-Risk Copilot

Challenge #5 (Autonomous SG 2026 · Airbus Track). A **leakage-safe** pipeline + a
Streamlit control-room dashboard that forecasts, 3–12 h ahead:

1. **Hourly Arrival Acceptance Rate (AAR)** per airport (regression, scored by RMSE), and
2. **Per-flight severe-delay risk** (> 45 min arrival delay, classification).

Demoed on **ATL** and **DFW**, but the pipeline is **airport-generic** (keyed on airport
code + runway heading). The language model only *explains* verified numbers — a human
controller decides. It is the aviation translation of the team's *PortSync AI* plan.

## Quickstart

```bash
pip install -r requirements.txt
python -m aerocast.pipeline     # writes ./artifacts (metrics, predictions, models)
streamlit run app.py            # opens the dashboard
```

…or just `./run.sh`. The data file is expected at
`sources/Flight_Data_With_Weather_Final.csv`.

> The first run fetches real hourly **wind direction** from Open-Meteo (free, no key)
> and caches it to `artifacts/wind_*.csv`, so re-runs are instant. Offline, it falls back
> to a clearly-labelled synthetic wind direction so the pipeline still runs end-to-end.
>
> Runs with only `pandas numpy scikit-learn joblib requests` installed. `xgboost`,
> `meteostat`, `streamlit`, `plotly`, `anthropic` are optional upgrades — the code falls
> back gracefully (scikit-learn models, template brief). Tests: `python tests/test_pipeline.py`.

## Module map (matches the build spec)

| File | Role | PortSync ancestor |
|---|---|---|
| `aerocast/config.py` | airports, runway headings, leads, paths, split | — |
| `aerocast/ingest.py` | load CSV, parse HHMM times, build airport-hour weather | Terminal State Engine |
| `aerocast/enrich.py` | add real wind **direction** (Open-Meteo → meteostat → synthetic) | External signals |
| `aerocast/features.py` | crosswind/tailwind + **leakage-safe lagged** weather + temporal | State Engine + Forecasting |
| `aerocast/targets.py` | AAR proxy + severe-delay label | Predicted variables |
| `aerocast/model_aar.py` | AAR regressor + interval | Forecasting |
| `aerocast/model_risk.py` | risk classifier | Equipment-health risk |
| `aerocast/evaluate.py` | baselines, RMSE/PR-AUC/calibration, failure case, importances | KPI Scorer |
| `aerocast/copilot.py` | drivers, confidence, plain-English brief | Operations Copilot |
| `aerocast/pipeline.py` | orchestrator (one entry point) | Closed decision loop |
| `app.py` | Streamlit ATC dashboard | Streamlit dashboard |

## The leakage guard (read this)

A forecast made at time *t* for target hour *h* (3–12 h ahead) uses **only** weather
known at *t*. Because the airport-hour weather is on a continuous hourly grid,
`shift(L)` moves a feature back exactly `L` hours, so a feature at hour *h* only ever
sees data from *h − L* or earlier (`LEADS = [3, 6, 12]`). We never use the weather
measured *at* hour *h*. `WEATHER_DELAY` / `NAS_DELAY` are **not** used as features
(79% missing and post-hoc cause tags).

## Honest findings (this is the evaluation story)

Inspecting the CSV (148,273 rows, 2023-01 → 2025-07) and running the pipeline:

- **Wind direction is enriched from Open-Meteo** (the CSV omits it), so crosswind is a
  *real* physically-consistent feature. The fetch is cached; offline it falls back to a
  labelled synthetic wdir.
- **The file is a ~0.5% flight sample**, so hourly arrival counts are tiny and
  **schedule-dominated** — the AAR regressor **matches the climatology baseline** at every
  bucket (1h/3h/6h/1D). Weather's marginal effect on *counts* is negligible at this sample
  size. Reported honestly next to the baselines; expected, not a bug.
- **The real signal is delay risk.** With leakage-safe weather + schedule + origin-weather +
  carrier features, the **isotonic-calibrated** classifier reaches **ROC-AUC ≈ 0.60,
  PR-AUC ≈ 0.18** (≈1.5× the ~12% base rate), catching ~55% of severe delays within a
  **top-20% review budget**.
- **Generalisation is checked** (leave-one-airport-out): trained on *one* airport and tested
  on the *other* (very different runway geometry, 80° vs 180°), the classifier still beats
  chance — see the `leave_one_airport_out` block in `metrics.json` / the dashboard. This
  guards the "overfit to a single airport" trap.
- **Honest ceiling**: ATL/DFW see little severe weather and arrival delay is
  propagation-dominated (late aircraft), which isn't a weather feature and is diluted by
  sampling. Expect stronger weather skill at storm-prone hubs with the full BTS feed.

The leakage guard, real public-data enrichment, baselines, calibration, the generalisation
check, and the honest gap are the deliverable.

## Config knobs (`aerocast/config.py`)

- `AIRPORTS`, `RUNWAY_HEADING` — add an airport in one line.
- `LEADS` — forecast horizons (hours ahead).
- `BUCKET` — AAR aggregation (`'1h'` default; `'3h'`/`'1D'` for a denser signal).
- `TEST_START` — temporal split (default `2025-01-01`).
- `SEVERE_DELAY_MIN` — delay threshold (default 45).

## Artifacts (written to `./artifacts`)

`metrics.json`, `aar_predictions.csv`, `risk_predictions.csv`, `drivers.json`,
`context.json`, `weather_enriched.csv`, `wind_ATL.csv` / `wind_DFW.csv` (cached Open-Meteo
wind), `aar_model.joblib`, `risk_model.joblib`.
