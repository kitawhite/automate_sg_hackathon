"""AeroCast AI - Predictive Airport Capacity & Delay-Risk Copilot (Challenge #5).

A leakage-safe pipeline that forecasts hourly Arrival Acceptance Rate (AAR) and
per-flight severe-delay risk 3-12h ahead for ATL/DFW (airport-generic).

Modules map 1:1 to the build spec:
    config   - constants, airports, runway headings, paths
    ingest   - load CSV, parse times, build airport-hour weather + flight tables
    enrich   - add wind direction (meteostat, synthetic fallback)
    features - crosswind + leakage-safe lagged weather + temporal features
    targets  - AAR proxy + severe-delay labels
    model_aar / model_risk - the two predictors
    evaluate - baselines, metrics, failure case, permutation importance
    copilot  - feature drivers, confidence, plain-English ATC brief
    pipeline - orchestrator (ingest -> ... -> train -> evaluate -> artifacts)
"""
__version__ = "0.1.0"
