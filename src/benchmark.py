"""
Model benchmark — answers the prior reviewer critique ("CatBoost claimed best with
no comparison"). We compare CatBoost, XGBoost, LightGBM, RandomForest and
LogisticRegression on the SAME gated pre-crash features with shuffled, stratified
5-fold CV, and report macro-F1 mean +/- std so the "best model" claim is evidenced
and its uncertainty is visible (overlapping bands => no significant winner).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

isnum = pd.api.types.is_numeric_dtype


def _enc(X):
    X = X.copy()
    cats = [c for c in X.columns if not isnum(X[c])]
    if cats:
        X[cats] = OrdinalEncoder().fit_transform(X[cats].astype(str))
    return X.apply(pd.to_numeric)


def benchmark(X, y, seed: int = 0) -> pd.DataFrame:
    Xe = _enc(X)
    yi = pd.Series(pd.factorize(y, sort=True)[0])          # integer labels for xgb
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    models = {
        "CatBoost": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.06,
                                       loss_function="MultiClass", verbose=False, random_seed=seed),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.06,
                                 tree_method="hist", verbosity=0, random_state=seed),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.06,
                                   random_state=seed, verbose=-1),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, random_state=seed)),
    }
    rows = []
    for name, mdl in models.items():
        sc = cross_val_score(mdl, Xe, yi, cv=cv, scoring="f1_macro")
        rows.append({"model": name, "macro_f1_mean": round(float(sc.mean()), 4),
                     "macro_f1_std": round(float(sc.std()), 4)})
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
