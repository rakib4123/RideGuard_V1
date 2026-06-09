"""
Deep-tabular baselines as a *negative result*.

Motivation: a reviewer may ask "why not deep learning?". We answer empirically.
On tabular data, gradient-boosted trees typically match or beat deep nets
[Grinsztajn et al., NeurIPS 2022; Shwartz-Ziv & Armon, Information Fusion 2022],
and deep nets are often worse-calibrated [Guo et al., 2017]. We therefore add an
MLP (always available) and TabNet (where PyTorch is installed, e.g. Colab) and
report macro-F1 AND calibration (ECE) against CatBoost. The expected finding is
that deep models do not improve macro-F1 and calibrate worse -- which strengthens,
rather than weakens, the paper's calibration-first thesis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

from .calibration import expected_calibration_error, onehot

isnum = pd.api.types.is_numeric_dtype


def _prep(X):
    X = X.copy()
    cats = [c for c in X.columns if not isnum(X[c])]
    if cats:
        X[cats] = OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=-1).fit_transform(X[cats].astype(str))
    return X.apply(pd.to_numeric)


def _metrics(proba, ytr_labels, yte, labels):
    pred = np.array([labels[i] for i in proba.argmax(1)])
    f1 = f1_score(yte, pred, average="macro")
    yte_oh = onehot(np.array([labels.index(v) for v in yte]), len(labels))
    ece = expected_calibration_error(yte_oh, proba)
    return round(float(f1), 4), round(float(ece), 4)


def deep_vs_gbdt(X, y, cat_cols, labels, seed=0) -> pd.DataFrame:
    Xe = _prep(X)
    Xtr, Xte, ytr, yte = train_test_split(Xe, y, test_size=0.25, stratify=y, random_state=seed)
    rows = []

    # --- CatBoost reference (uncalibrated, to compare like-for-like) ---
    cb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.06,
                            loss_function="MultiClass", verbose=False, random_seed=seed)
    cb.fit(Xtr, ytr)
    cls = list(cb.classes_); idx = [cls.index(c) for c in labels]
    f1, ece = _metrics(cb.predict_proba(Xte)[:, idx], ytr, yte, labels)
    rows.append({"model": "CatBoost (GBDT)", "macro_f1": f1, "ece": ece})

    # --- MLP (always available) ---
    scaler = StandardScaler().fit(Xtr)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=seed)
    mlp.fit(scaler.transform(Xtr), ytr)
    cls = list(mlp.classes_); idx = [cls.index(c) for c in labels]
    f1, ece = _metrics(mlp.predict_proba(scaler.transform(Xte))[:, idx], ytr, yte, labels)
    rows.append({"model": "MLP (deep)", "macro_f1": f1, "ece": ece})

    # --- TabNet (only if PyTorch is present, e.g. on Colab) ---
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        yi = {l: i for i, l in enumerate(labels)}
        clf = TabNetClassifier(seed=seed, verbose=0)
        clf.fit(scaler.transform(Xtr).astype(np.float32),
                np.array([yi[v] for v in ytr]),
                max_epochs=60, patience=15, batch_size=1024, eval_metric=["accuracy"])
        proba = clf.predict_proba(scaler.transform(Xte).astype(np.float32))
        # TabNet columns are ordered by integer label = labels order
        f1, ece = _metrics(proba, ytr, yte, labels)
        rows.append({"model": "TabNet (deep)", "macro_f1": f1, "ece": ece})
    except Exception as e:
        rows.append({"model": "TabNet (deep)", "macro_f1": None, "ece": None,
                     "note": f"skipped: {type(e).__name__}"})

    # --- TensorFlow / Keras MLP (only if TensorFlow is present, e.g. on Colab) ---
    try:
        import os as _os
        _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        tf.random.set_seed(seed)
        yi = {l: i for i, l in enumerate(labels)}
        ytr_i = np.array([yi[v] for v in ytr])
        Xtr_s = scaler.transform(Xtr).astype(np.float32)
        Xte_s = scaler.transform(Xte).astype(np.float32)
        net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(Xtr_s.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(len(labels), activation="softmax"),
        ])
        net.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
        net.fit(Xtr_s, ytr_i, epochs=40, batch_size=256, verbose=0)
        proba = net.predict(Xte_s, verbose=0)          # columns already in labels order
        f1, ece = _metrics(proba, ytr, yte, labels)
        rows.append({"model": "TensorFlow/Keras (deep)", "macro_f1": f1, "ece": ece})
    except Exception as e:
        rows.append({"model": "TensorFlow/Keras (deep)", "macro_f1": None, "ece": None,
                     "note": f"skipped: {type(e).__name__}"})

    return pd.DataFrame(rows)
