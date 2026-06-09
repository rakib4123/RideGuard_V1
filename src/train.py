"""
RideGuard end-to-end pipeline (trained on the REAL uploaded datasets).

    python -m src.train

Stages
  1. Load + gate both real datasets.
  2. Leakage audit (forbidden guard + empirical inflation receipts).
  3. Tier-1 calibrated behavioral model (bike case-control): acc / macro-F1 / ECE.
  4. Conformal risk sets: empirical coverage + average set size.
  5. Tier-2 spatial relative-risk prior (Dhaka crash-only): DBSCAN + KDE.
  6. Counterfactual scenario stress-test: monotonicity violation rates.
  7. Continual self-learning + ADWIN drift detection (with a controlled drift probe).
  8. Fusion demo: point + route advisory.
Artifacts -> outputs/.
"""
from __future__ import annotations
import os, json, pathlib
import numpy as np
import pandas as pd

from .data import load_gate, load_bike, load_dhaka, load_bike_raw
from .leakage_audit import assert_no_forbidden, leakage_inflation
from .tier1_behavioral import fit_tier1, RISK_W
from .conformal import ConformalRisk
from .tier2_spatial import fit_spatial_prior
from .fusion import FusionConfig, fuse, advisory, sample_polyline, score_route
from .counterfactual import run_stress_test
from .continual import ContinualRiskModel
from .realism import determinism_receipt, noise_robustness
from .benchmark import benchmark
from .xai import global_importance, explain_one
from .deep_baseline import deep_vs_gbdt
from .ablation import feature_group_ablation, fusion_sensitivity

OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)
isnum = pd.api.types.is_numeric_dtype


def banner(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main():
    gate = load_gate()
    results = {"gate_version": gate["version"]}

    banner("STAGE 1-2  Load + leakage audit (REAL data)")
    X, y, cat_cols, labels = load_bike(gate)
    dh, gdh = load_dhaka(gate)
    print(f"Bike (tier-1): {X.shape}  classes={dict(y.value_counts())}")
    print(f"Dhaka (tier-2): {dh.shape}")
    assert_no_forbidden(X.columns, set(gate["tier1_behavioral"]["forbidden"]))
    print("Forbidden-feature guard: CLEAN")

    # DATA REALISM RECEIPT: is the target a deterministic function of features?
    det = determinism_receipt(X, y)
    print("Data-realism (bike) determinism receipt:", det)
    results["data_realism_bike"] = det

    # empirical leakage receipts (subsampled for runtime)
    bike_raw = load_bike_raw()
    inf_bike = leakage_inflation(bike_raw, list(X.columns), "Number_of_Vehicles", "Accident_Severity")
    dh_s = dh.sample(6000, random_state=0)
    # clean dramatic receipt: Severity_Category is a deterministic function of the
    # post-crash Death+Injury counts, so feeding those = near-perfect "prediction".
    pre_cols = ["Road nature", "Weather", "Lighting", "Junction", "Latitude", "Longitude"]
    inf_dh = leakage_inflation(dh_s, pre_cols, ["Death Count", "Injury Count"], "Severity_Category")
    print("Leakage receipt (bike, Number_of_Vehicles):", inf_bike)
    print("Leakage receipt (Dhaka, Severity_Score):  ", inf_dh)
    results["leakage_receipts"] = [inf_bike, inf_dh]
    pd.DataFrame([inf_bike, inf_dh]).to_csv(OUT / "leakage_receipts.csv", index=False)

    banner("STAGE 3  Tier-1 calibrated behavioral risk model")
    t1, m1, splits = fit_tier1(X, y, cat_cols, labels)
    print("Tier-1 metrics:", m1)
    results["tier1"] = m1
    # persist test-set arrays for figures
    p_te_full = t1.predict_proba(splits["Xte"])
    np.savez(OUT / "tier1_test.npz",
             proba=p_te_full,
             y_true=np.array([labels.index(v) for v in splits["yte"]]),
             labels=np.array(labels))

    banner("STAGE 4  Conformal risk sets (coverage guarantee)")
    p_cal = t1.predict_proba(splits["Xcal"])
    ycal_idx = np.array([labels.index(v) for v in splits["ycal"]])
    conf = ConformalRisk(alpha=0.1).calibrate(p_cal, ycal_idx)
    p_te = t1.predict_proba(splits["Xte"])
    yte_idx = np.array([labels.index(v) for v in splits["yte"]])
    cov = conf.empirical_coverage(p_te, yte_idx)
    setsz = conf.avg_set_size(p_te)
    print(f"target coverage=0.90  empirical coverage={cov:.3f}  avg set size={setsz:.2f}")
    results["conformal"] = {"target": 0.90, "empirical_coverage": round(cov, 3),
                            "avg_set_size": round(setsz, 2)}

    banner("STAGE 5  Tier-2 spatial relative-risk prior (Dhaka)")
    sp = fit_spatial_prior(dh, weights=gdh["intensity_weights"])
    print(f"DBSCAN clusters={len(sp.centers)}  noise_frac={sp.noise_frac:.4f}")
    hot = pd.DataFrame({"lat": sp.centers[:, 0], "lon": sp.centers[:, 1],
                        "count": sp.counts, "norm_density": np.round(sp.cluster_scores, 4)})
    hot.sort_values("norm_density", ascending=False).to_csv(OUT / "hotspots.csv", index=False)
    print(hot.sort_values("norm_density", ascending=False).head().to_string(index=False))
    results["tier2"] = {"n_clusters": int(len(sp.centers)), "noise_frac": round(sp.noise_frac, 4)}

    banner("STAGE 6  Counterfactual scenario stress-test (model auditing)")
    stress = run_stress_test(t1, X)
    print(stress.to_string(index=False))
    stress.to_csv(OUT / "counterfactual_stress.csv", index=False)
    mvr = stress["monotonicity_violation_rate"].mean()
    results["stress_test"] = {"mean_violation_rate": round(float(mvr), 4),
                              "n_scenarios": int(len(stress))}

    banner("STAGE 6b  Noise-robustness (graceful degradation + calibration under uncertainty)")
    nr = noise_robustness(X, y, labels)
    print(nr.to_string(index=False))
    nr.to_csv(OUT / "noise_robustness.csv", index=False)
    results["noise_robustness"] = nr.to_dict(orient="records")

    banner("STAGE 7  Continual self-learning + ADWIN drift detection")
    num_cols = [c for c in X.columns if isnum(X[c])]
    # build a stream of 4000: stationary first half, then a CONTROLLED concept drift
    # (flip the protective effect of the helmet) to verify the detector fires.
    S = X.sample(2500, random_state=3)
    ys = y.loc[S.index]
    S = S.reset_index(drop=True); ys = ys.reset_index(drop=True)
    half = len(S) // 2
    drifted = ys.copy()
    # CONTROLLED concept reversal in the second half: swap Moderate <-> Severe.
    # This is a sharp, known drift used to validate that ADWIN detects it.
    sec = S.index >= half
    swap = {"Moderate Accident": "Severe Accident", "Severe Accident": "Moderate Accident"}
    drifted.loc[sec] = drifted.loc[sec].map(lambda v: swap.get(v, v))
    cm = ContinualRiskModel(cat_cols=cat_cols, num_cols=num_cols)
    cm.stream_fit(S, drifted)
    preq = cm.prequential_accuracy()
    # recovery: accuracy in the window right after the drift point vs end
    err = np.array(cm.errors)
    preq["drift_detected_near_midpoint"] = any(abs(d - half) < 250 for d in cm.drift_points)
    print("Prequential (controlled drift at sample %d):" % half, preq)
    np.savez(OUT / "drift_stream.npz", errors=np.array(cm.errors),
             drift_points=np.array(cm.drift_points), half=half)
    results["continual"] = preq

    banner("STAGE 8  Fusion demo: point + route advisory")
    cfg = FusionConfig()
    risky = X.iloc[[int(t1.risk_score(X).argmax())]]            # highest-risk real rider
    s_beh = float(t1.risk_score(risky)[0])
    la0, lo0 = sp.centers[int(sp.cluster_scores.argmax())]
    R_pt = fuse(s_beh, sp.prior(la0, lo0), cfg)
    pt = {"s_behavior": round(s_beh, 4), "spatial_prior": round(sp.prior(la0, lo0), 4),
          "R": round(R_pt, 4), "level": advisory(R_pt, cfg),
          "conformal_set": conf.predict_set(t1.predict_proba(risky)[0], labels)}
    print("Point advisory:", pt)
    route_pts = [(23.74, 90.39), (23.78, 90.40), (23.81, 90.42)]
    route = score_route(sample_polyline(route_pts, 75), s_beh, sp.prior, cfg)
    print(f"Route: max={route['max_risk']} mean={route['mean_risk']} "
          f"p90={route['p90_risk']} level={route['level']}")
    results["demo"] = {"point": pt, "route": {k: route[k] for k in ("max_risk","mean_risk","p90_risk","level")}}

    banner("STAGE 9  Model benchmark (answers 'is CatBoost actually best?')")
    bm = benchmark(X, y)
    print(bm.to_string(index=False))
    bm.to_csv(OUT / "model_benchmark.csv", index=False)
    results["benchmark"] = bm.to_dict(orient="records")

    banner("STAGE 10  Explainability (SHAP): global + local")
    gi = global_importance(t1, X)
    gi.to_csv(OUT / "shap_global.csv", index=False)
    print("Global SHAP (top 8):"); print(gi.head(8).to_string(index=False))
    local = explain_one(t1, risky)
    print("Local explanation (highest-risk rider):", local)
    results["xai"] = {"global_top": gi.head(8).to_dict(orient="records"), "local_example": local}

    banner("STAGE 11  Deep-tabular baselines (negative result): DL vs GBDT")
    deep = deep_vs_gbdt(X, y, cat_cols, labels)
    print(deep.to_string(index=False))
    deep.to_csv(OUT / "deep_vs_gbdt.csv", index=False)
    results["deep_vs_gbdt"] = deep.to_dict(orient="records")

    banner("STAGE 12  Ablation + fusion-weight sensitivity (answers Reviewer #2.4)")
    abl = feature_group_ablation(X, y)
    print("Feature-group ablation:"); print(abl.to_string(index=False))
    abl.to_csv(OUT / "ablation.csv", index=False)
    fs = fusion_sensitivity(t1, sp, X)
    print("\nFusion-weight sensitivity:"); print(fs.to_string(index=False))
    fs.to_csv(OUT / "fusion_sensitivity.csv", index=False)
    results["ablation"] = abl.to_dict(orient="records")
    results["fusion_sensitivity"] = fs.to_dict(orient="records")

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    banner("DONE -> outputs/")
    for p in sorted(OUT.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
