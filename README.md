# RideGuard

**A Calibrated, Auditable, and Continually Learning Framework for Pre-Crash Motorcyclist Risk in Dhaka.**

> 📄 **Paper:** *RideGuard: A Behavioural-Spatial Risk Framework with Conformal Prediction for Motorcycle Safety*, presented at **IEEE COMPAS 2026**.
>
> 📱 **The app built on this model:** [Ride_Guard_WebApp](https://github.com/rakib4123/Ride_Guard_WebApp) · [Live demo](https://ride-guard-web-app-web.vercel.app)

RideGuard couples a calibrated behavioural risk model with a spatial relative-risk prior, wraps both in conformal prediction, continual learning with drift detection, and a self-auditing layer (the **Realism Audit**, Algorithm 1 in the paper). Its central finding: a model can reach ~0.97 macro-F1 with good calibration while the audit shows the data has essentially no learnable pre-crash signal — so high accuracy and good calibration are **not**, by themselves, evidence of trustworthiness.

> The evaluation datasets are near-deterministic by the framework's own audit (`label_determinism = 1.00`). Results below are reproducible and honest; the framework's value is the calibration-and-audit machinery, which would measure deployment readiness directly on real or prospective data.

## What runs

`python -m src.train` trains and writes all artifacts to `outputs/`; `python -m src.figures` renders all figures.

- **Tier-1** calibrated CatBoost + isotonic recalibration (`tier1_behavioral.py`, `calibration.py`)
- **Tier-2** spatial DBSCAN + severity-weighted KDE prior (`tier2_spatial.py`)
- **Fusion** of behavioural score and spatial prior (`fusion.py`)
- **Conformal** split-conformal prediction sets (`conformal.py`)
- **Continual** adaptive random forest + ADWIN drift detection (`continual.py`)
- **Benchmark** CatBoost / XGBoost / LightGBM / RandomForest / LogReg (`benchmark.py`)
- **Deep baselines** MLP / TabNet / TensorFlow-Keras (`deep_baseline.py`)
- **Counterfactual** single-factor monotonicity stress test (`counterfactual.py`)
- **Ablation** semantic feature-group drops (`ablation.py`)
- **XAI** global + local SHAP (`xai.py`)
- **Realism audit** determinism D, leakage-inflation L (`realism.py`, `leakage_audit.py`)

## Quick start

### Colab (no setup)
Open `notebooks/RideGuard_full_pipeline_colab.ipynb` in Google Colab. It contains the entire source inline: it writes every module, you upload the two datasets, and it trains everything and plots all figures.

### Local
```bash
pip install -r requirements.txt
# put MotorBike_Accident.xlsx and Accident_Dataset_Dhaka.xlsx here (see data/README.md)
python -m src.train      # trains everything -> outputs/
python -m src.figures    # renders figures   -> outputs/figures/
```

## Reproduced results (from a full `src.train` run)

| Quantity | Value |
|---|---|
| Behavioural label determinism `D` | **1.00** |
| Dhaka leakage probe (severity macro-F1) | **0.228 -> 1.000** when casualty counts added (inflation 0.772) |
| Tier-1 macro-F1 / accuracy (held-out) | **0.9718 / 0.9715** |
| Calibration ECE (uncalibrated -> isotonic) | **0.0429 -> 0.0076** |
| Conformal coverage (target 0.90) / set size | **0.895 / 1.0** |
| Benchmark macro-F1 (CatBoost / LGBM / XGB / RF / LogReg) | 0.9733 / 0.9703 / 0.9693 / 0.969 / 0.6865 |
| Deep baselines (MLP / TabNet / TF-Keras) | 0.9518 / 0.9486 / 0.9563 (all < GBDT 0.978) |
| Spatial DBSCAN clusters / noise | 253 / 3.6% |
| Counterfactual mean monotonicity-violation rate | 0.060 |
| Concept-drift: injected step 1,250 / detected | 1,472 |

Pre-rendered figures and the full `metrics.json` from a verified run are in `results/`.

## Repository layout
```
src/        pipeline modules (entry points: src.train, src.figures)
config/     feature_gate.yaml (the leakage-aware pre-crash gate)
notebooks/  all-in-one Colab notebook (full source inline)
results/    figures + metrics.json + artifact CSVs from a verified run
data/       how to obtain the two datasets (not redistributed)
```

## Notes
- All runs use fixed seeds; exact decimals can vary slightly across library/OS versions.
- The deep baselines need `torch`, `tensorflow`, and `pytorch-tabnet` (in `requirements.txt`; preinstalled on Colab).
- Data is not redistributed — see `data/README.md` for the Mendeley DOIs.

## License
MIT (see `LICENSE`).
