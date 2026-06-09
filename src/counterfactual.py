"""
Counterfactual scenario stress-testing  (the rigorous version of the user's
"test on synthetic scenarios since the app isn't ready").

A safety model can hit high accuracy yet behave nonsensically. So we audit it
causally: take real rows, INTERVENE on one risk factor at a time (do-operator
style), and check the predicted risk moves in the medically/physically expected
direction. We report a Monotonicity Violation Rate (MVR) per factor and an
average intervention effect. This is a model-AUDITING contribution, not just
another test split, and it is exactly what a deployable safety tool must pass.

Known-direction priors (risk should INCREASE under these changes):
  helmet Yes->No, alcohol 0->1, speed up, talk/smoke while riding No->Yes,
  weather Clear->Rainy/Foggy, valid license Yes->No, bike condition New->Old.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# expected sign of risk change when we apply the intervention (+1 = risk up)
SCENARIOS = {
    "remove_helmet":      ("Wearing_Helmet", "No", +1),
    "add_alcohol":        ("Biker_Alcohol", 1, +1),
    "talk_while_riding":  ("Talk_While_Riding", "Regularly", +1),
    "smoke_while_riding": ("Smoke_While_Riding", "Regularly", +1),
    "no_license":         ("Valid_Driving_License", "No", +1),
    "old_bike":           ("Bike_Condition", "Old", +1),
    "make_rainy":         ("Weather", "Rainy", +1),
    "make_foggy":         ("Weather", "Foggy", +1),
}


def speed_scenario(tier1, X: pd.DataFrame, delta=20):
    base = tier1.risk_score(X)
    Xp = X.copy()
    Xp["Bike_Speed"] = Xp["Bike_Speed"] + delta
    new = tier1.risk_score(Xp)
    d = new - base
    return {"scenario": f"speed +{delta} km/h",
            "mean_risk_delta": round(float(d.mean()), 4),
            "monotonicity_violation_rate": round(float((d < 0).mean()), 4)}


def run_stress_test(tier1, X: pd.DataFrame, sample_n=2000, seed=0) -> pd.DataFrame:
    Xs = X.sample(min(sample_n, len(X)), random_state=seed).reset_index(drop=True)
    base = tier1.risk_score(Xs)
    rows = []
    for name, (col, val, sign) in SCENARIOS.items():
        if col not in Xs.columns:
            continue
        Xp = Xs.copy()
        Xp[col] = val
        new = tier1.risk_score(Xp)
        d = (new - base) * sign            # positive = behaves as expected
        rows.append({"scenario": name, "factor": col,
                     "mean_risk_delta": round(float(d.mean()), 4),
                     "monotonicity_violation_rate": round(float((d < 0).mean()), 4)})
    rows.append(speed_scenario(tier1, Xs))
    return pd.DataFrame(rows)
