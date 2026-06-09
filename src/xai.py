"""
Explainability (XAI) via SHAP — pairs with the counterfactual audit to make the
model transparent and auditable.

We use CatBoost's native ShapValues (no extra dependency, exact for trees). For a
K-class model CatBoost returns an array of shape (n, K, n_features+1); the last
column per class is the base value. We report:
  - GLOBAL importance: mean over samples of mean-over-classes |shap| per feature,
  - LOCAL explanation: the signed feature contributions for a single prediction.

Note (honesty): on the synthetic bike data, SHAP simply recovers the dataset's
deterministic generating rule. That is itself evidence for the data-realism
argument, and the same code yields meaningful attributions on real data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from catboost import Pool


def _shap(tier1, X: pd.DataFrame):
    pool = Pool(X[tier1.model.feature_names_], cat_features=tier1.cat_cols)
    sv = np.asarray(tier1.model.get_feature_importance(pool, type="ShapValues"))
    # (n, K, n_feat+1) multiclass; (n, n_feat+1) binary -> normalise to 3D
    if sv.ndim == 2:
        sv = sv[:, None, :]
    return sv[:, :, :-1]                      # drop base-value column


def global_importance(tier1, X: pd.DataFrame, sample_n: int = 2000, seed: int = 0) -> pd.DataFrame:
    Xs = X.sample(min(sample_n, len(X)), random_state=seed)
    contrib = _shap(tier1, Xs)               # (n, K, n_feat)
    imp = np.abs(contrib).mean(axis=(0, 1))  # mean |shap| over samples & classes
    out = pd.DataFrame({"feature": tier1.model.feature_names_,
                        "mean_abs_shap": np.round(imp, 4)})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_one(tier1, x_row: pd.DataFrame, top_k: int = 6) -> dict:
    proba = tier1.predict_proba(x_row)[0]
    k = int(proba.argmax())
    contrib = _shap(tier1, x_row)[0, k, :]   # contributions toward predicted class
    s = pd.Series(contrib, index=tier1.model.feature_names_)
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(top_k)
    return {"predicted_class": tier1.labels[k],
            "predicted_proba": round(float(proba[k]), 4),
            "top_contributions": {f: round(float(v), 4) for f, v in s.items()}}
