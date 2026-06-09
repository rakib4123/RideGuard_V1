"""
Continual / self-learning engine with concept-drift detection.

This is the "self-learning" requirement, done properly. In deployment the app
streams events: a completed safe trip is a NEGATIVE ("No Accident"); a reported
incident is a POSITIVE (Moderate/Severe). The model updates online, one sample at
a time, and an ADWIN drift detector watches the running error. When drift fires
(seasonal change, new construction, policy change), we flag it for recalibration.

We use river's Adaptive Random Forest, which itself adapts trees on drift; ADWIN
on the 0/1 error stream gives an explicit, reportable drift signal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from river import forest, drift, preprocessing, compose


class ContinualRiskModel:
    def __init__(self, cat_cols, num_cols):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        # one-hot categoricals + standardise numerics, then ARF classifier
        self.model = compose.Pipeline(
            preprocessing.OneHotEncoder(),
            preprocessing.StandardScaler(),
            forest.ARFClassifier(n_models=6, seed=0),
        )
        self.detector = drift.ADWIN(delta=0.01)
        self.n_seen = 0
        self.errors = []          # running 0/1 error
        self.drift_points = []

    def _row(self, X_row: pd.Series) -> dict:
        d = {c: (str(X_row[c]) if c in self.cat_cols else float(X_row[c]))
             for c in self.cat_cols + self.num_cols}
        return d

    def learn_one(self, X_row: pd.Series, y: str):
        x = self._row(X_row)
        if self.n_seen > 0:                       # prequential: predict then learn
            pred = self.model.predict_one(x)
            err = int(pred != y)
            self.errors.append(err)
            self.detector.update(err)
            if self.detector.drift_detected:
                self.drift_points.append(self.n_seen)
        self.model.learn_one(x, y)
        self.n_seen += 1

    def stream_fit(self, X: pd.DataFrame, y: pd.Series):
        for (_, xr), yv in zip(X.iterrows(), y.astype(str).values):
            self.learn_one(xr, yv)
        return self

    def prequential_accuracy(self, window=500) -> dict:
        e = np.array(self.errors)
        if e.size == 0:
            return {"overall_acc": None}
        roll = pd.Series(1 - e).rolling(window, min_periods=50).mean()
        return {"overall_acc": round(float((1 - e).mean()), 4),
                "final_window_acc": round(float(roll.iloc[-1]), 4),
                "n_drift_signals": len(self.drift_points),
                "drift_points": self.drift_points[:20]}
