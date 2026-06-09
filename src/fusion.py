"""
Fusion + point/route scoring.

Key honesty: Tier-1 is a CALIBRATED PROBABILITY (rider behavioral risk); Tier-2 is
a RELATIVE PRIOR (spatial). We do not pretend they are the same quantity. The
fused advisory keeps the behavioral probability primary and uses the spatial
prior only to MODULATE it within a bounded band -> the spatial layer can raise or
lower caution but cannot manufacture a probability it has no exposure data to support.

    R = clip( s_behavior * (1 + lambda*(2*spatial_prior - 1)), 0, 1 )

lambda (default 0.30) caps how far the spatial prior may push the calibrated risk.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class FusionConfig:
    lam: float = 0.30          # spatial modulation strength
    low: float = 0.33          # advisory thresholds on R
    high: float = 0.60


def fuse(s_behavior: float, spatial_prior: float, cfg: FusionConfig) -> float:
    return float(np.clip(s_behavior * (1 + cfg.lam * (2 * spatial_prior - 1)), 0, 1))


def advisory(R: float, cfg: FusionConfig) -> str:
    return "Low" if R < cfg.low else ("Medium" if R < cfg.high else "High")


def sample_polyline(points, step_m=75.0):
    from .tier2_spatial import _haversine_m
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    for (a1, o1), (a2, o2) in zip(points[:-1], points[1:]):
        seg = _haversine_m(a1, o1, a2, o2)
        n = max(1, int(seg // step_m))
        for k in range(1, n + 1):
            t = k / n
            out.append((a1 + t * (a2 - a1), o1 + t * (o2 - o1)))
    return out


def score_route(sampled_pts, s_behavior, prior_fn, cfg: FusionConfig):
    seg = [{"lat": la, "lon": lo, "R": round(fuse(s_behavior, prior_fn(la, lo), cfg), 4)}
           for la, lo in sampled_pts]
    Rs = np.array([p["R"] for p in seg]) if seg else np.array([0.0])
    return {"max_risk": round(float(Rs.max()), 4),
            "mean_risk": round(float(Rs.mean()), 4),
            "p90_risk": round(float(np.percentile(Rs, 90)), 4),
            "level": advisory(float(Rs.mean()), cfg), "segments": seg}
