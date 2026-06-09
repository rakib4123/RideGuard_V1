"""Generate publication figures from saved pipeline artifacts. Run after train."""
from __future__ import annotations
import json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .calibration import reliability_points, onehot

OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def fig_confusion():
    d = np.load(OUT / "tier1_test.npz", allow_pickle=True)
    labels = list(d["labels"]); yt = d["y_true"]; pred = d["proba"].argmax(1)
    cm = confusion_matrix(yt, pred)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    short = [l.replace(" Accident", "") for l in labels]
    ax.set_xticks(range(len(labels)), short); ax.set_yticks(range(len(labels)), short)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Tier-1 confusion (held-out)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(FIG / "fig1_confusion.png"); plt.close(fig)


def fig_reliability():
    d = np.load(OUT / "tier1_test.npz", allow_pickle=True)
    proba = d["proba"]; yt = d["y_true"]
    oh = onehot(yt, proba.shape[1])
    xs, ys, ns = reliability_points(oh, proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(xs, ys, "o-", color="#c0392b", label="calibrated model")
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig2_reliability.png"); plt.close(fig)


def fig_noise():
    nr = pd.read_csv(OUT / "noise_robustness.csv")
    fig, ax1 = plt.subplots(figsize=(4.8, 4))
    ax1.plot(nr.label_noise, nr.macro_f1, "o-", color="#2c3e50", label="macro-F1")
    ax1.set_xlabel("Injected label noise"); ax1.set_ylabel("macro-F1", color="#2c3e50")
    ax2 = ax1.twinx()
    ax2.plot(nr.label_noise, nr.ece, "s--", color="#e67e22", label="ECE")
    ax2.set_ylabel("ECE", color="#e67e22")
    ax1.set_title("Graceful degradation under noise"); ax1.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig3_noise_robustness.png"); plt.close(fig)


def fig_counterfactual():
    st = pd.read_csv(OUT / "counterfactual_stress.csv").dropna(subset=["factor"])
    st = st.sort_values("mean_risk_delta")
    fig, ax = plt.subplots(figsize=(5.4, 4))
    colors = ["#27ae60" if v > 0 else "#c0392b" for v in st.mean_risk_delta]
    ax.barh(st.scenario, st.mean_risk_delta, color=colors)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Mean risk change (expected sign = positive)")
    ax.set_title("Counterfactual intervention sensitivity")
    fig.tight_layout(); fig.savefig(FIG / "fig4_counterfactual.png"); plt.close(fig)


def fig_hotspots():
    hot = pd.read_csv(OUT / "hotspots.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    sc = ax.scatter(hot.lon, hot.lat, s=8 + 40 * hot.norm_density,
                    c=hot.norm_density, cmap="hot_r", alpha=.8, edgecolor="k", lw=.2)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Dhaka severity-weighted hotspots (relative prior)")
    fig.colorbar(sc, label="normalised density"); fig.tight_layout()
    fig.savefig(FIG / "fig5_hotspots.png"); plt.close(fig)


def fig_drift():
    d = np.load(OUT / "drift_stream.npz")
    err = d["errors"]; dp = d["drift_points"]; half = int(d["half"])
    acc = pd.Series(1 - err).rolling(150, min_periods=30).mean()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(acc.values, color="#2c3e50", label="rolling accuracy (150)")
    ax.axvline(half, color="#e67e22", ls="--", label="concept drift injected")
    for k, p in enumerate(dp):
        ax.axvline(p, color="#c0392b", ls=":", label="ADWIN detection" if k == 0 else None)
    ax.set_xlabel("stream step"); ax.set_ylabel("accuracy")
    ax.set_title("Online learning: drift detection"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig6_drift.png"); plt.close(fig)


def fig_benchmark():
    bm = pd.read_csv(OUT / "model_benchmark.csv").sort_values("macro_f1_mean")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.barh(bm.model, bm.macro_f1_mean, xerr=bm.macro_f1_std,
            color="#34495e", ecolor="#e67e22", capsize=3)
    ax.set_xlabel("macro-F1 (5-fold, shuffled)  ± std")
    ax.set_title("Model comparison on gated pre-crash features")
    ax.set_xlim(0.6, 1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig7_benchmark.png"); plt.close(fig)


def fig_shap():
    gi = pd.read_csv(OUT / "shap_global.csv").head(10).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.barh(gi.feature, gi.mean_abs_shap, color="#2980b9")
    ax.set_xlabel("mean |SHAP|  (global importance)")
    ax.set_title("Tier-1 explainability (SHAP)")
    fig.tight_layout(); fig.savefig(FIG / "fig8_shap_global.png"); plt.close(fig)


def fig_deep():
    d = pd.read_csv(OUT / "deep_vs_gbdt.csv").dropna(subset=["macro_f1"])
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    x = np.arange(len(d)); w = 0.38
    ax.bar(x - w/2, d.macro_f1, w, label="macro-F1", color="#2c3e50")
    ax.bar(x + w/2, d.ece, w, label="ECE (lower=better)", color="#e67e22")
    ax.set_xticks(x, d.model, rotation=15, ha="right", fontsize=8)
    ax.set_title("Deep tabular vs GBDT: accuracy and calibration")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "fig9_deep_vs_gbdt.png"); plt.close(fig)


def fig_ablation():
    a = pd.read_csv(OUT / "ablation.csv")
    a = a[a.removed_group != "(none) full model"].sort_values("drop_vs_full")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.barh(a.removed_group, a.drop_vs_full, color="#16a085")
    ax.set_xlabel("macro-F1 drop when group removed")
    ax.set_title("Feature-group ablation (Tier-1)")
    fig.tight_layout(); fig.savefig(FIG / "fig10_ablation.png"); plt.close(fig)


def fig_fusion():
    f = pd.read_csv(OUT / "fusion_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.stackplot(f["lambda"], f.frac_Low, f.frac_Medium, f.frac_High,
                 labels=["Low", "Medium", "High"],
                 colors=["#a9dfbf", "#f9e79f", "#e6707f"])
    ax.set_xlabel("fusion weight $\\lambda$"); ax.set_ylabel("advisory fraction")
    ax.set_title("Fusion-weight sensitivity"); ax.legend(fontsize=8, loc="center left")
    ax.set_ylim(0, 1); fig.tight_layout(); fig.savefig(FIG / "fig11_fusion_sensitivity.png"); plt.close(fig)


def main():
    for fn in [fig_confusion, fig_reliability, fig_noise, fig_counterfactual,
               fig_hotspots, fig_drift, fig_benchmark, fig_shap,
               fig_deep, fig_ablation, fig_fusion]:
        try:
            fn(); print("ok:", fn.__name__)
        except Exception as e:
            print("FAIL:", fn.__name__, e)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
