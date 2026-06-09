"""
Split-conformal risk sets (Vovk; Romano et al. APS) -> distribution-free coverage.

Instead of a single point class, RideGuard returns a SET of plausible severity
levels with a guarantee: the true level is in the set >= (1-alpha) of the time.
When the set is large (model is unsure) or contains "Severe", the engine
ESCALATES caution. This is the trustworthy, OOD-aware advisory that replaces a
meaningless 0.97 accuracy headline.
"""
from __future__ import annotations
import numpy as np


class ConformalRisk:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.qhat = None

    def calibrate(self, proba_cal: np.ndarray, y_cal_idx: np.ndarray):
        """APS nonconformity: 1 - p(true class). qhat = (1-alpha) quantile."""
        n = len(y_cal_idx)
        scores = 1.0 - proba_cal[np.arange(n), y_cal_idx]
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        k = min(max(k, 1), n)
        self.qhat = float(np.sort(scores)[k - 1])
        return self

    def predict_set(self, proba_row: np.ndarray, labels: list[str]) -> list[str]:
        keep = [labels[j] for j in range(len(labels)) if (1.0 - proba_row[j]) <= self.qhat]
        return keep or [labels[int(proba_row.argmax())]]   # never empty

    def empirical_coverage(self, proba: np.ndarray, y_idx: np.ndarray) -> float:
        inset = (1.0 - proba[np.arange(len(y_idx)), y_idx]) <= self.qhat
        return float(inset.mean())

    def avg_set_size(self, proba: np.ndarray) -> float:
        sizes = ((1.0 - proba) <= self.qhat).sum(axis=1)
        sizes = np.maximum(sizes, 1)        # predict_set never returns empty
        return float(sizes.mean())
