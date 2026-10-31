"""Module 6 - Baselines & Evaluation.

This is where the honest-evaluation points live. We never report a model number
without a baseline next to it.

  AAR   : persistence (same hour-of-day mean) and climatology (hour x month mean)
          baselines; RMSE, MAE, skill score vs the better baseline.
  Risk  : precision / recall / F1 at threshold, PR-AUC, ROC-AUC, Brier; per airport.
  Plus  : one worst-error failure case, and permutation feature importances for
          the copilot's "drivers".

PortSync ancestor: KPI Scorer / honest evaluation.
"""
import json

import numpy as np
import pandas as pd

from . import config


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


# --------------------------------------------------------------------------- AAR
def aar_baselines(train: pd.DataFrame, test: pd.DataFrame):
    """Persistence (airport x hour) and climatology (airport x hour x month)."""
    pers = train.groupby(["airport", "hour"])["aar"].mean().rename("persistence")
    clim = train.groupby(["airport", "hour", "month"])["aar"].mean().rename("climatology")
    t = test.merge(pers, on=["airport", "hour"], how="left") \
            .merge(clim, on=["airport", "hour", "month"], how="left")
    grand = float(train["aar"].mean())
    t["persistence"] = t["persistence"].fillna(grand)
    t["climatology"] = t["climatology"].fillna(t["persistence"])
    return t["persistence"].to_numpy(), t["climatology"].to_numpy()


def evaluate_aar(test: pd.DataFrame, y_pred, base_pers, base_clim) -> dict:
    y = test["aar"].to_numpy()
    rmse_model = _rmse(y, y_pred)
    rmse_pers = _rmse(y, base_pers)
    rmse_clim = _rmse(y, base_clim)
    best_base = min(rmse_pers, rmse_clim)
    return {
        "n_test": int(len(y)),
        "rmse_model": round(rmse_model, 4),
        "mae_model": round(_mae(y, y_pred), 4),
        "rmse_persistence": round(rmse_pers, 4),
        "rmse_climatology": round(rmse_clim, 4),
        "skill_vs_best_baseline": round(1 - rmse_model / best_base, 4) if best_base else None,
    }


# -------------------------------------------------------------------------- RISK
def pick_threshold(proba, flag_budget: float = 0.20) -> float:
    """Screening operating point as a FLAG BUDGET: flag the top `flag_budget`
    fraction by score ("controllers review the 20% highest-risk arrivals"). A
    budget is a policy choice, not tuned to labels, so it transfers cleanly; set
    it on a validation slice and we then report recall + lift on the test set."""
    return round(float(np.quantile(np.asarray(proba), 1 - flag_budget)), 4)


def evaluate_risk(y_true, proba, airports, threshold=0.5) -> dict:
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 average_precision_score, roc_auc_score, brier_score_loss)
    y = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    out = {
        "n_test": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "threshold": threshold,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "flagged_rate": round(float(pred.mean()), 4),
        "lift_vs_random": (round(float(precision_score(y, pred, zero_division=0)) / float(y.mean()), 3)
                           if y.mean() > 0 else None),
        "pr_auc": round(float(average_precision_score(y, proba)), 4) if y.sum() else None,
        "roc_auc": round(float(roc_auc_score(y, proba)), 4) if 0 < y.sum() < len(y) else None,
        "brier": round(float(brier_score_loss(y, proba)), 4),
        "per_airport": {},
    }
    ap = np.asarray(airports)
    for a in np.unique(ap):
        m = ap == a
        if m.sum() == 0:
            continue
        ya, pa = y[m], np.asarray(proba)[m]
        out["per_airport"][str(a)] = {
            "n": int(m.sum()),
            "positive_rate": round(float(ya.mean()), 4),
            "pr_auc": round(float(average_precision_score(ya, pa)), 4) if ya.sum() else None,
            "recall": round(float(recall_score(ya, (pa >= threshold).astype(int), zero_division=0)), 4),
        }
    return out


# ----------------------------------------------------------------- aux artifacts
def worst_failure_case(test: pd.DataFrame, y_pred) -> dict:
    err = np.abs(test["aar"].to_numpy() - np.asarray(y_pred))
    i = int(np.argmax(err))
    row = test.iloc[i]
    return {
        "airport": str(row["airport"]),
        "dt": str(row["dt"]),
        "actual_aar": float(row["aar"]),
        "predicted_aar": round(float(y_pred[i]), 2),
        "abs_error": round(float(err[i]), 2),
        "note": "Largest AAR error in the hold-out - inspect weather drivers here.",
    }


def permutation_importances(model_predict, X: pd.DataFrame, y, n=1000, repeats=2):
    """Model-agnostic importances on a sample (for the copilot's drivers)."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)
    ys = np.asarray(y)[idx]
    base = _rmse(ys, model_predict(Xs))
    imp = {}
    for c in Xs.columns:
        drops = []
        for _ in range(repeats):
            Xp = Xs.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops.append(_rmse(ys, model_predict(Xp)) - base)
        imp[c] = float(np.mean(drops))
    return dict(sorted(imp.items(), key=lambda kv: -kv[1]))


def save_metrics(metrics: dict, path=None):
    path = path or (config.ARTIFACTS / "metrics.json")
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    return path
