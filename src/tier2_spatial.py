"""
Tier-2: spatial relative-risk PRIOR from the Dhaka crash-only data.

Honest epistemic stance (the core fix vs prior work): with crash-only data and
no exposure denominator, we CANNOT estimate P(crash) here. We estimate a
*relative* spatial intensity from where severity-weighted crashes concentrate,
using DBSCAN hotspots + a Gaussian KDE over coordinates. The output is a
normalised relative prior in [0,1], never a probability.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

R_EARTH_M = 6.371e6


def _haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    d1, d2 = lat2 - lat1, lon2 - lon1
    a = np.sin(d1 / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d2 / 2) ** 2
    return R_EARTH_M * 2 * np.arcsin(np.sqrt(a))


@dataclass
class SpatialPrior:
    centers: np.ndarray
    cluster_scores: np.ndarray
    counts: np.ndarray
    kde: KernelDensity
    kde_lo: float
    kde_hi: float
    noise_frac: float

    def prior(self, lat: float, lon: float) -> float:
        """Relative spatial risk in [0,1] from the KDE (smooth, deployable)."""
        log_d = self.kde.score_samples(np.radians([[lat, lon]]))[0]
        v = (log_d - self.kde_lo) / (self.kde_hi - self.kde_lo + 1e-9)
        return float(np.clip(v, 0, 1))


def fit_spatial_prior(df, lat="Latitude", lon="Longitude", intensity="Accident_Intensity",
                      weights=None, eps_m=250.0, min_samples=20, kde_bw_m=300.0) -> SpatialPrior:
    weights = weights or {}
    coords = df[[lat, lon]].values
    coords_rad = np.radians(coords)

    db = DBSCAN(eps=eps_m / R_EARTH_M, min_samples=min_samples,
                metric="haversine").fit(coords_rad)
    lab = db.labels_
    noise_frac = float((lab == -1).mean())
    w = df[intensity].map(weights).fillna(0.5).values if intensity in df else np.ones(len(df))

    centers, scores, counts = [], [], []
    for c in sorted(set(lab) - {-1}):
        m = lab == c
        centers.append(coords[m].mean(0)); counts.append(int(m.sum())); scores.append(float(w[m].sum()))
    centers = np.array(centers) if centers else np.empty((0, 2))
    scores = np.array(scores) if scores else np.empty((0,))
    counts = np.array(counts) if counts else np.empty((0,), int)
    if scores.size:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    # KDE over crash coordinates (haversine), severity-weighted by sample_weight.
    # Subsample for tractability: haversine KDE scoring is ~O(n_query * n_fit).
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(len(coords_rad), size=min(6000, len(coords_rad)), replace=False)
    kde = KernelDensity(bandwidth=kde_bw_m / R_EARTH_M, metric="haversine")
    kde.fit(coords_rad[fit_idx], sample_weight=w[fit_idx])
    scan_idx = rng.choice(len(coords_rad), size=min(3000, len(coords_rad)), replace=False)
    ld = kde.score_samples(coords_rad[scan_idx])
    lo, hi = np.percentile(ld, 2), np.percentile(ld, 98)

    return SpatialPrior(centers=centers, cluster_scores=scores, counts=counts,
                        kde=kde, kde_lo=lo, kde_hi=hi, noise_frac=noise_frac)
