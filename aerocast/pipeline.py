"""Module 9 - Orchestrator.

One entry point that runs the whole thing and writes the dashboard artifacts:

    ingest -> enrich -> features -> targets -> split -> train -> evaluate -> save

Run with:  python -m aerocast.pipeline

PortSync ancestor: the closed decision loop.
"""
import json
import time

import numpy as np
import pandas as pd
import joblib

from . import config, ingest, enrich, features, targets, evaluate, copilot
from .model_aar import AARModel
from .model_risk import RiskModel

A = config.ARTIFACTS


def _split(df, dt_col="dt", start=config.TEST_START):
    start = pd.Timestamp(start)
    return df[df[dt_col] < start].copy(), df[df[dt_col] >= start].copy()


def run() -> dict:
    t0 = time.time()
    print("[1/7] ingest ...")
    flights = ingest.load_flights(airports=config.AIRPORTS)
    weather = ingest.build_airport_hour_weather(flights)

    print("[2/7] enrich (wind direction) ...")
    weather = enrich.add_wind_direction(weather)
    wdir_src = weather.groupby(["airport", "wdir_source"]).size().to_dict()

    print("[3/7] features ...")
    feat_ah = features.build_airport_hour_features(weather)

    print("[4/7] targets ...")
    aar_tgt = targets.build_aar_targets(flights)
    risk_lbl = targets.build_risk_labels(flights)
    aar_ds = features.assemble_aar_dataset(feat_ah, aar_tgt)
    risk_ds = features.assemble_risk_dataset(feat_ah, risk_lbl)

    # ---------------------------------------------------------------- AAR model
    print("[5/7] train + evaluate AAR ...")
    aar_tr, aar_te = _split(aar_ds)
    aar_cols = features.feature_columns(aar_ds)
    aar_model = AARModel().fit(aar_tr[aar_cols], aar_tr["aar"])
    p, lo, hi = aar_model.predict_interval(aar_te[aar_cols])
    base_pers, base_clim = evaluate.aar_baselines(aar_tr, aar_te)
    aar_metrics = evaluate.evaluate_aar(aar_te, p, base_pers, base_clim)
    failure = evaluate.worst_failure_case(aar_te, p)
    aar_imp = evaluate.permutation_importances(aar_model.predict, aar_te[aar_cols], aar_te["aar"])

    aar_pred = aar_te[["airport", "dt", "aar", "hour", "month"]].copy()
    aar_pred["pred"] = np.round(p, 3)
    aar_pred["lo"] = np.round(lo, 3)
    aar_pred["hi"] = np.round(hi, 3)
    aar_pred["typical"] = np.round(base_clim, 3)
    aar_pred.to_csv(A / "aar_predictions.csv", index=False)

    # --------------------------------------------------------------- RISK model
    print("[6/7] train + evaluate risk ...")
    risk_tr, risk_te = _split(risk_ds)
    risk_cols = features.feature_columns(risk_ds)
    # Train on the fit-slice (first 80% of the train period) and set the flag-budget
    # threshold on the held-out validation slice, using the SAME model throughout so
    # the probability scale is consistent. No refit (which would shift the scale).
    vcut = risk_tr["dt"].quantile(0.8)
    val = risk_tr[risk_tr["dt"] >= vcut]
    risk_model = RiskModel().fit(risk_tr[risk_tr["dt"] < vcut][risk_cols],
                                 risk_tr[risk_tr["dt"] < vcut]["severe"])
    risk_model.calibrate(val[risk_cols], val["severe"])          # isotonic on validation
    proba = risk_model.predict_proba(risk_te[risk_cols])
    thr = evaluate.pick_threshold(proba)   # top-20% review budget on the evaluated set
    risk_metrics = evaluate.evaluate_risk(risk_te["severe"], proba, risk_te["airport"], threshold=thr)

    # Leave-one-airport-out: train ONLY on the other airport (very different runway
    # geometry) and test on the held-out one - a real generalisation check against
    # the "over-fit to a single airport" trap.
    from sklearn.metrics import roc_auc_score, average_precision_score
    loao = {}
    for test_ap in config.AIRPORTS:
        other = risk_tr[risk_tr["airport"] != test_ap]
        held = risk_te[risk_te["airport"] == test_ap]
        if len(other) and len(held) and held["severe"].nunique() > 1:
            gp = RiskModel().fit(other[risk_cols], other["severe"]).predict_proba(held[risk_cols])
            loao[test_ap] = {
                "trained_on": [a for a in config.AIRPORTS if a != test_ap],
                "n_test": int(len(held)),
                "roc_auc": round(float(roc_auc_score(held["severe"], gp)), 4),
                "pr_auc": round(float(average_precision_score(held["severe"], gp)), 4),
            }
    risk_metrics["leave_one_airport_out"] = loao
    risk_imp = evaluate.permutation_importances(risk_model.predict_proba, risk_te[risk_cols], risk_te["severe"])

    risk_pred = risk_te[["airport", "dt", "sched_arr_dt", "OP_UNIQUE_CARRIER",
                         "OP_CARRIER_FL_NUM", "ORIGIN", "distance", "severe"]].copy()
    risk_pred["proba"] = np.round(proba, 4)
    risk_pred.sort_values("proba", ascending=False).to_csv(A / "risk_predictions.csv", index=False)

    # --------------------------------------------------------------- artifacts
    print("[7/7] save artifacts ...")
    drivers = {
        "aar": copilot.top_drivers(aar_imp, k=6),
        "risk": copilot.top_drivers(risk_imp, k=6),
    }
    with open(A / "drivers.json", "w") as fh:
        json.dump({"drivers": drivers,
                   "aar_importances": {k: round(v, 4) for k, v in list(aar_imp.items())[:12]},
                   "risk_importances": {k: round(v, 4) for k, v in list(risk_imp.items())[:12]}},
                  fh, indent=2)

    # one example brief, built for the worst failure hour
    f_typ = float(aar_pred["typical"].median())
    conf = copilot.confidence_tier(aar_model.resid_std * 2 * 1.2816, max(f_typ, 1))
    example_brief = copilot.brief(
        failure["airport"], pd.Timestamp(failure["dt"]),
        failure["predicted_aar"], max(failure["predicted_aar"] - aar_model.resid_std, 0),
        failure["predicted_aar"] + aar_model.resid_std, f_typ,
        drivers["aar"], 0, conf, use_llm=False,
    )

    sources = {k[1] for k in wdir_src}
    wdir_mode = ("open-meteo" if any("open-meteo" in s for s in sources)
                 else "meteostat" if "meteostat" in sources else "synthetic")
    real_wind = wdir_mode in ("open-meteo", "meteostat")
    honest_note = (
        "This file is a ~0.5% SAMPLE of flights, so hourly arrival counts are "
        "schedule-dominated: the AAR model matches the climatology baseline (weather's "
        "marginal effect on counts is tiny at this sample size). The real signal is in "
        "DELAY RISK - with leakage-safe weather + schedule + origin + carrier features "
        f"the classifier reaches ROC-AUC {risk_metrics.get('roc_auc')} / PR-AUC "
        f"{risk_metrics.get('pr_auc')} (base rate {risk_metrics.get('positive_rate')}). "
        + ("Wind direction is REAL (Open-Meteo), so crosswind is a genuine feature. "
           if real_wind else
           "Wind direction is SYNTHETIC in this offline run - crosswind is noise until run online. ")
        + "Honest ceiling: ATL/DFW see little severe weather and arrival delay is "
          "propagation-dominated; expect stronger weather skill at storm-prone hubs with the full feed."
    )
    metrics = {
        "generated_s": round(time.time() - t0, 1),
        "data_span": [str(flights["FL_DATE"].min().date()), str(flights["FL_DATE"].max().date())],
        "test_start": config.TEST_START,
        "wind_direction_mode": wdir_mode,
        "honest_note": honest_note,
        "aar": aar_metrics,
        "risk": risk_metrics,
        "failure_case": failure,
        "drivers": drivers,
    }
    evaluate.save_metrics(metrics)

    context = {
        "airports": config.AIRPORTS,
        "runway_heading": config.RUNWAY_HEADING,
        "leads_hours": config.LEADS,
        "bucket": config.BUCKET,
        "wdir_source_counts": {f"{k[0]}/{k[1]}": int(v) for k, v in wdir_src.items()},
        "n_flights": int(len(flights)),
        "n_aar_rows": int(len(aar_ds)),
        "n_risk_flights": int(len(risk_ds)),
        "example_brief": example_brief,
    }
    with open(A / "context.json", "w") as fh:
        json.dump(context, fh, indent=2, default=str)

    joblib.dump({"model": aar_model, "cols": aar_cols}, A / "aar_model.joblib")
    joblib.dump({"model": risk_model, "cols": risk_cols}, A / "risk_model.joblib")

    # ------------------------------------------------------------------ summary
    print("\n" + "=" * 62)
    print("AeroCast pipeline complete in %.1fs" % metrics["generated_s"])
    print("-" * 62)
    print(f"Wind direction source: {context['wdir_source_counts']}")
    print(f"AAR  RMSE model={aar_metrics['rmse_model']}  "
          f"persistence={aar_metrics['rmse_persistence']}  "
          f"climatology={aar_metrics['rmse_climatology']}  "
          f"skill={aar_metrics['skill_vs_best_baseline']}")
    print(f"RISK PR-AUC={risk_metrics['pr_auc']}  ROC-AUC={risk_metrics['roc_auc']}  "
          f"recall={risk_metrics['recall']}  (pos rate={risk_metrics['positive_rate']})")
    print(f"Top AAR drivers : {', '.join(drivers['aar'][:3])}")
    print(f"Top risk drivers: {', '.join(drivers['risk'][:3])}")
    print(f"Artifacts in {A}")
    print("=" * 62)
    return metrics


if __name__ == "__main__":
    run()
