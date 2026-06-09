"""
Tier-1: calibrated behavioral risk model (bike case-control data).

Because the bike dataset HAS a "No Accident" control class, this tier estimates a
genuine calibrated probability over {No Accident, Moderate, Severe} from
deployable rider+context features. We:
  - split train / calibration / test (calibration split powers isotonic + conformal),
  - fit CatBoost, then isotonic-calibrate per class,
  - map class probabilities to a scalar ordinal risk in [0,1].
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score
from catboost import CatBoostClassifier

from .calibration import expected_calibration_error, onehot

# ordinal risk weight per severity level (No Accident=0 ... Severe=1)
RISK_W = {"No Accident": 0.0, "Moderate Accident": 0.5, "Severe Accident": 1.0}


@dataclass
class Tier1:
    model: CatBoostClassifier
    isotonics: list
    labels: list
    cat_cols: list

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(X[self._cols()])
        cls = list(self.model.classes_)
        raw = raw[:, [cls.index(c) for c in self.labels]]
        cal = np.column_stack([self.isotonics[j].predict(raw[:, j]) for j in range(len(self.labels))])
        cal = np.clip(cal, 1e-6, None)
        return cal / cal.sum(axis=1, keepdims=True)

    def risk_score(self, X: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(X)
        w = np.array([RISK_W[c] for c in self.labels])
        return p @ w

    def _cols(self):
        return self.model.feature_names_


def fit_tier1(X, y, cat_cols, labels, seed=0):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed)

    model = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.06,
                               loss_function="MultiClass", random_seed=seed, verbose=False)
    model.fit(Xtr, ytr, cat_features=cat_cols)
    cls = list(model.classes_)
    idx = [cls.index(c) for c in labels]

    # isotonic calibration per class on the calibration split
    p_cal = model.predict_proba(Xcal)[:, idx]
    ycal_oh = onehot(np.array([labels.index(v) for v in ycal]), len(labels))
    isos = []
    for j in range(len(labels)):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(p_cal[:, j], ycal_oh[:, j])
        isos.append(iso)

    t1 = Tier1(model=model, isotonics=isos, labels=labels, cat_cols=cat_cols)

    # report uncalibrated vs calibrated on test
    p_raw_te = model.predict_proba(Xte)[:, idx]
    p_cal_te = t1.predict_proba(Xte)
    yte_idx = np.array([labels.index(v) for v in yte])
    yte_oh = onehot(yte_idx, len(labels))
    pred = np.array([labels[i] for i in p_cal_te.argmax(1)])
    metrics = {
        "test_accuracy": round(accuracy_score(yte, pred), 4),
        "test_macro_f1": round(f1_score(yte, pred, average="macro"), 4),
        "ece_uncalibrated": round(expected_calibration_error(yte_oh, p_raw_te), 4),
        "ece_calibrated": round(expected_calibration_error(yte_oh, p_cal_te), 4),
    }
    splits = {"Xcal": Xcal, "ycal": ycal, "Xte": Xte, "yte": yte}
    return t1, metrics, splits
