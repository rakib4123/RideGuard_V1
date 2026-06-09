"""
Data-realism diagnostics + noise-robustness demonstration.

Why this matters for publishing: a model can score 0.97 because the data is
*synthetic with a near-deterministic generating rule*, not because it solved a
hard problem. We measure and REPORT this directly, and we show how the framework
behaves once realistic aleatoric uncertainty is present (label-noise injection).
The graceful-degradation + still-calibrated behaviour under noise is a
methodological result that does NOT depend on the (untrustworthy) headline score.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier

from .calibration import expected_calibration_error, onehot

isnum = pd.api.types.is_numeric_dtype


def determinism_receipt(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fraction of distinct feature-combinations that map to >1 label.
    ~0 with ~all-unique combos => target is a (near-)deterministic function of
    features => synthetic data; headline accuracy is not real-world meaningful."""
    combo = X.astype(str).agg("|".join, axis=1)
    g = pd.DataFrame({"c": combo, "y": y.values}).groupby("c")["y"].nunique()
    return {"n_rows": int(len(X)), "n_unique_combos": int(len(g)),
            "frac_unique": round(len(g) / len(X), 3),
            "combos_with_multiple_labels": int((g > 1).sum()),
            "label_determinism": round(1 - (g > 1).mean(), 4)}


def _enc(X):
    X = X.copy()
    cats = [c for c in X.columns if not isnum(X[c])]
    if cats:
        X[cats] = OrdinalEncoder().fit_transform(X[cats].astype(str))
    return X.apply(pd.to_numeric)


def noise_robustness(X, y, labels, noise_levels=(0.0, 0.1, 0.2, 0.3), seed=0) -> pd.DataFrame:
    """Inject symmetric label noise, refit, report macro-F1 + calibrated ECE.
    Demonstrates the framework remains calibrated as the task becomes realistically
    hard (the regime real crash data lives in)."""
    Xe = _enc(X)
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    rows = []
    for eps in noise_levels:
        yn = y.copy().values
        if eps > 0:
            flip = rng.random(len(yn)) < eps
            yn[flip] = rng.choice(labels, size=flip.sum())
        yn = pd.Series(yn)
        cb = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.06,
                                loss_function="MultiClass", verbose=False, random_seed=seed)
        f1 = cross_val_score(cb, Xe, yn, cv=cv, scoring="f1_macro").mean()
        # quick holdout ECE
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(Xe, yn, test_size=0.25,
                                              stratify=yn, random_state=seed)
        cb.fit(Xtr, ytr)
        cls = list(cb.classes_); idx = [cls.index(c) for c in labels]
        p = cb.predict_proba(Xte)[:, idx]
        yte_oh = onehot(np.array([labels.index(v) for v in yte]), len(labels))
        rows.append({"label_noise": eps, "macro_f1": round(float(f1), 3),
                     "ece": round(expected_calibration_error(yte_oh, p), 3)})
    return pd.DataFrame(rows)
