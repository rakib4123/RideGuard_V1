"""
Calibration: the honest replacement for headline accuracy.

A risk advisory is only trustworthy if predicted probabilities match observed
frequencies. We report Expected Calibration Error (ECE) and a reliability curve,
and we calibrate with isotonic regression on a held-out split.
"""
from __future__ import annotations
import numpy as np


def expected_calibration_error(y_true_onehot: np.ndarray, proba: np.ndarray,
                               n_bins: int = 10) -> float:
    """Multiclass ECE (confidence of the top class vs its accuracy)."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    true = y_true_onehot.argmax(axis=1)
    correct = (pred == true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_points(y_true_onehot, proba, n_bins=10):
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    true = y_true_onehot.argmax(axis=1)
    correct = (pred == true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def onehot(y_idx, n_classes):
    o = np.zeros((len(y_idx), n_classes))
    o[np.arange(len(y_idx)), y_idx] = 1
    return o
