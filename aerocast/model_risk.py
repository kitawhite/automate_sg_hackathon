"""Module 5b - Severe-delay risk classifier.

Prefers XGBoost (scale_pos_weight for the ~9.6% positive rate); falls back to
scikit-learn HistGradientBoostingClassifier with class_weight="balanced".
Exposes calibrated-ish probabilities via predict_proba.

PortSync ancestor: Equipment-health risk classifier.
"""
import numpy as np

from . import config


class RiskModel:
    def __init__(self):
        self.model = None
        self.features: list[str] | None = None
        self._cal = None  # optional probability calibrator

    def fit(self, X, y):
        self.features = list(X.columns)
        y = np.asarray(y)
        try:
            from xgboost import XGBClassifier
            pos = max(int((y == 1).sum()), 1)
            spw = float((y == 0).sum()) / pos
            self.model = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                random_state=config.RANDOM_SEED, n_jobs=-1, eval_metric="logloss",
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.06, max_depth=8,
                class_weight="balanced", early_stopping=False,
                random_state=config.RANDOM_SEED,
            )
        self.model.fit(X, y)
        return self

    def calibrate(self, X, y):
        """Platt (sigmoid) calibration on a held-out slice, so probabilities read as
        real likelihoods (better Brier / the dashboard's confidence) while keeping a
        smooth, granular ranking. Sigmoid suits this data size better than isotonic,
        which over-flattens with few positives. Robust to sklearn version
        differences; silently falls back to raw probabilities."""
        try:
            from sklearn.calibration import CalibratedClassifierCV
            try:
                from sklearn.frozen import FrozenEstimator   # sklearn >= 1.6
                self._cal = CalibratedClassifierCV(
                    FrozenEstimator(self.model), method="sigmoid"
                ).fit(X[self.features], np.asarray(y))
            except Exception:
                self._cal = CalibratedClassifierCV(
                    self.model, cv="prefit", method="sigmoid"
                ).fit(X[self.features], np.asarray(y))
        except Exception:
            self._cal = None
        return self

    def predict_proba(self, X):
        if self._cal is not None:
            return self._cal.predict_proba(X[self.features])[:, 1]
        return self.model.predict_proba(X[self.features])[:, 1]
