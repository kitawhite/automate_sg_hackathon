"""Module 5a - AAR regressor.

Prefers XGBoost; falls back to scikit-learn's HistGradientBoosting (both handle
NaN natively, so lagged-edge gaps are fine). Point forecast + a residual-based
prediction interval ("visible uncertainty" for the dashboard).

PortSync ancestor: Forecasting.
"""
import numpy as np

from . import config


def make_regressor():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=config.RANDOM_SEED, n_jobs=-1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_depth=8,
            random_state=config.RANDOM_SEED,
        )


class AARModel:
    """Thin wrapper giving a stable fit/predict/predict_interval interface."""

    def __init__(self):
        self.model = make_regressor()
        self.features: list[str] | None = None
        self.resid_std: float = 1.0

    def fit(self, X, y):
        self.features = list(X.columns)
        self.model.fit(X, y)
        self.resid_std = float(np.std(np.asarray(y) - self.model.predict(X)))
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X[self.features]), 0.0, None)

    def predict_interval(self, X, z: float = 1.2816):  # ~80% interval
        p = self.predict(X)
        d = z * self.resid_std
        return p, np.clip(p - d, 0.0, None), p + d
