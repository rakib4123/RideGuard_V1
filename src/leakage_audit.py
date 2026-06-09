"""
Empirical leakage audit -> produces evidence, not assertions.

Two checks:
  1. forbidden-as-feature guard (hard error).
  2. an *ablation-by-leakage* demo: train with vs without a forbidden column and
     show how much the metric inflates. This is the receipt that the previous
     work's ~0.97 was leakage, and that RideGuard's number is honest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder

isnum = pd.api.types.is_numeric_dtype


def _enc(X):
    X = X.copy()
    cats = [c for c in X.columns if not isnum(X[c])]
    if cats:
        X[cats] = OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=-1).fit_transform(X[cats].astype(str))
    return X.apply(pd.to_numeric)


def quick_f1(X, y, cv=4):
    from sklearn.model_selection import StratifiedKFold
    rf = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=0)
    skf = StratifiedKFold(cv, shuffle=True, random_state=0)
    return float(cross_val_score(rf, _enc(X), y.astype(str), cv=skf, scoring="f1_macro").mean())


def leakage_inflation(df: pd.DataFrame, base_feats: list[str], leak_cols, target: str) -> dict:
    """macro-F1 with vs without `leak_cols` -> the inflation leakage buys."""
    if isinstance(leak_cols, str):
        leak_cols = [leak_cols]
    y = df[target]
    f_clean = quick_f1(df[base_feats], y)
    f_leak = quick_f1(df[base_feats + list(leak_cols)], y)
    return {"target": target, "leak_cols": ", ".join(leak_cols),
            "macroF1_precrash_only": round(f_clean, 3),
            "macroF1_with_outcome_cols": round(f_leak, 3),
            "inflation": round(f_leak - f_clean, 3)}


def assert_no_forbidden(feature_cols, forbidden: set[str]):
    used = sorted(set(feature_cols) & forbidden)
    if used:
        raise ValueError(f"LEAKAGE GUARD: forbidden columns present as features: {used}")
    return "CLEAN"
