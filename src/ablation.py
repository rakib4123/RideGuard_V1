"""
Ablation and sensitivity analyses (directly answer Reviewer #2, point 4).

1. Feature-group ablation (Tier-1): remove each semantic group of features and
   measure the macro-F1 drop. This has a real label, so the contribution of each
   group is meaningful.
2. Fusion-weight sensitivity: sweep lambda and report how the advisory-level
   distribution (Low/Medium/High) shifts, showing how strongly the spatial prior
   modulates the calibrated behavioural risk.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier

from .fusion import FusionConfig, fuse, advisory

isnum = pd.api.types.is_numeric_dtype

FEATURE_GROUPS = {
    "risk_behaviors": ["Biker_Alcohol", "Talk_While_Riding", "Smoke_While_Riding"],
    "protective":     ["Wearing_Helmet", "Valid_Driving_License", "Bike_Condition"],
    "rider_profile":  ["Biker_Age", "Riding_Experience", "Biker_Occupation",
                       "Biker_Education_Level", "Daily_Travel_Distance", "Motorcycle_Ownership"],
    "environment":    ["Weather", "Road_Type", "Road_condition", "Time_of_Day", "Traffic_Density"],
    "speed":          ["Bike_Speed", "Speed_Limit"],
}


def _enc(X):
    X = X.copy()
    cats = [c for c in X.columns if not isnum(X[c])]
    if cats:
        X[cats] = OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=-1).fit_transform(X[cats].astype(str))
    return X.apply(pd.to_numeric)


def _f1(X, y, seed=0):
    cb = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.06,
                            loss_function="MultiClass", verbose=False, random_seed=seed)
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    return float(cross_val_score(cb, _enc(X), y.astype(str), cv=cv, scoring="f1_macro").mean())


def feature_group_ablation(X, y, seed=0) -> pd.DataFrame:
    full = _f1(X, y, seed)
    rows = [{"removed_group": "(none) full model", "features_dropped": 0,
             "macro_f1": round(full, 4), "drop_vs_full": 0.0}]
    for name, cols in FEATURE_GROUPS.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        f = _f1(X.drop(columns=cols), y, seed)
        rows.append({"removed_group": name, "features_dropped": len(cols),
                     "macro_f1": round(f, 4), "drop_vs_full": round(full - f, 4)})
    return pd.DataFrame(rows).sort_values("drop_vs_full", ascending=False).reset_index(drop=True)


def fusion_sensitivity(tier1, spatial, X, lambdas=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7),
                       sample_n=1500, seed=0) -> pd.DataFrame:
    Xs = X.sample(min(sample_n, len(X)), random_state=seed)
    s_beh = tier1.risk_score(Xs)
    # sample real hotspot coordinates for spatial prior values
    rng = np.random.default_rng(seed)
    if len(spatial.centers):
        ci = rng.integers(0, len(spatial.centers), size=len(Xs))
        priors = np.array([spatial.prior(spatial.centers[i, 0], spatial.centers[i, 1]) for i in ci])
    else:
        priors = np.full(len(Xs), 0.5)
    rows = []
    for lam in lambdas:
        cfg = FusionConfig(lam=lam)
        R = np.array([fuse(float(b), float(p), cfg) for b, p in zip(s_beh, priors)])
        lvl = pd.Series([advisory(r, cfg) for r in R]).value_counts(normalize=True)
        rows.append({"lambda": lam, "mean_R": round(float(R.mean()), 4),
                     "frac_High": round(float(lvl.get("High", 0)), 3),
                     "frac_Medium": round(float(lvl.get("Medium", 0)), 3),
                     "frac_Low": round(float(lvl.get("Low", 0)), 3)})
    return pd.DataFrame(rows)
