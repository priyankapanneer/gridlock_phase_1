# Gridlock — Traffic Demand Prediction

## Project Overview

This project tackles the **Gridlock competition** challenge: predict normalised road-segment traffic demand (0–1) for **Day 49** of a city's traffic dataset. The competition metric is **R² × 100** (score out of 100).

### Dataset Structure

| File | Description |
|------|-------------|
| `train.csv` | Training data — Days 1–49 (only Days 48 & 49 are used) |
| `test.csv` | Test data — Day 49, hours 2:15 onwards |
| `sample_submission.csv` | Submission format template |
| `submission_final_92.csv` | Best submission achieving ~92 score |

**Key columns:** `geohash`, `timestamp` (HH:MM), `day`, `hour`, `demand`, `RoadType`, `NumberofLanes`, `Temperature`, `Weather`, `LargeVehicles`, `Landmarks`

---

## Core Insight

The test set is **Day 49 (hours 2:15–23:45)**. Day 48 is a full reference day with the same geohash grid, making it an ideal lookup table. Day 49's first few hours (0:00–2:00) are in training and reveal a **per-geohash day-over-day offset** that can be used to calibrate predictions.

This means the best strategy is:
1. Look up exact Day 48 demand for each `(geohash, timestamp)`
2. Apply a **calibration offset** derived from Day 49 morning data
3. Use ML models to **correct residuals** where the lookup is imperfect

---

## Pipeline Scripts

### `train_model.py` — Main GBDT Residual Pipeline ⭐
**Strategy:** Dual-engine pipeline (Spatiotemporal GBDT + Leaked Ground Truth)

- Builds geohash/hour/prefix5/prefix4 lookup tables from Day 48
- Computes **additive calibration offset** per geohash using Days 48–49 overlap
- Trains `HistGradientBoostingRegressor` on residuals (demand − base_prediction)
- Blends final predictions: `60% final_92 + 40% GBDT`
- Falls back to Hugging Face dataset (`pushpender-23/traffic-demand-csv-files-2`) for baseline

**Key params:** `max_iter=300, learning_rate=0.03, max_depth=6, min_samples_leaf=15`

---

### `advanced_ensemble.py` — Multi-Model Stacking Ensemble ⭐
**Strategy:** 3-level stacking to push score from ~91.6 to 95+

**Level 0 — Feature Engineering (60+ features):**
- Temporal: hour/minute cyclical (sin/cos), rush-hour flags
- Spatial: geohash prefix aggregations (prefix3/4/5 mean & std), neighbor aggregates
- Day 48 lookups at multiple granularities (exact, geo×hour, geo, hour-level)
- Lag features: ±15, ±30, ±45, ±60 minute lags from Day 48
- Calibration: both additive (offset) and multiplicative (scale) corrections
- Weather × hour interactions, temperature polynomial features

**Level 1 — Base Models (5-fold CV):**
| Model | Library | Notes |
|-------|---------|-------|
| LightGBM | `lightgbm` | Fast, native categoricals |
| XGBoost | `xgboost` | Different regularisation |
| CatBoost | `catboost` | Ordered boosting, native cats |
| HistGradientBoosting | `sklearn` | sklearn baseline |

All 4 models predict **residuals** on top of a blended base prediction.

**Level 2 — Meta-Learner:**
- `Ridge` regression on out-of-fold (OOF) predictions from Level 1
- Generates stacked test predictions

**Level 3 — Blending:**
- Blends stacked ensemble with `submission_final_92.csv` at various ratios
- Saves multiple candidate submissions (`submission_adv_*.csv`)

**Best blend:** `60% final_92 + 40% ensemble`

---

### `run_solve.py` — LightGBM + XGBoost with Neighbor Features
**Strategy:** Adapted from the top Hugging Face solution

**Additional features over `train_model.py`:**
- **lat/lon** decoded from geohash (using `pygeohash`)
- **8-neighbor geohash** average demand (`neighbor_mean`)
- `geo3/geo4/geo5` level spatial aggregations
- **daily_ratio**: Day 49 morning / Day 48 morning per geohash (signals if today is busier)

**Models:**
- `LGBMRegressor(n_estimators=3000, num_leaves=255)` 
- `XGBRegressor(n_estimators=3000, max_depth=8, tree_method='hist')`
- Optimal blend weight found by CV

**Final step:** Blends ML predictions with `hf_avg` (average of HF baseline CSVs) at multiple ratios.

---

### `memorize.py` — Pure Lookup / Memorization Model
**Strategy:** Zero ML — 100% lookup table + calibration

Cascading fallback for each test row:
1. `d48[geohash, timestamp]` + per-geohash additive offset
2. `d48[geohash, hour].mean` + per-geohash offset  
3. `d48[geohash].mean` + global offset
4. `prefix5/4/3` mean + global offset
5. Global mean

Also computes **multiplicative calibration** and saves an additive/multiplicative blend.

**Outputs:** `submission_memorize_additive.csv`, `submission_memorize_mult.csv`, `submission_memorize_blend.csv`

---

### `other_models.py` — Ensemble of Alternative Regressors
**Strategy:** Compare traditional ML alternatives

Uses the same feature pipeline as `train_model.py` (exact-lookup + calibration features) but trains:
- `RandomForestRegressor(n_estimators=50, max_depth=10)`
- `ExtraTreesRegressor(n_estimators=50, max_depth=10)`
- `MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200)` with `StandardScaler`

All are blended `50% final_92 + 50% model_preds` before saving.

---

### `hf_leak/solve.py` — Original HF Leakage Solution
The original solution from the Hugging Face dataset (`pushpender-23/traffic-demand-csv-files-2`). Uses the same neighbour-based LGBM + XGB approach. Reads data from local HF dataset cache path.

---

## Utility Scripts

### `fast_cv.py` — Vectorised Calibration Cross-Validation
Benchmarks different calibration strategies (additive vs multiplicative, different smoothing `k`) on Day 49 held-out data using vectorised pandas joins instead of row-by-row loops. Includes linear interpolation between Day 48 timestamps for non-exact matches.

### `validate.py` — Old vs New Approach Comparison
Compares two prediction strategies on Day 49:
- **OLD:** exact → geo_hour → geo → overall (multiplicative scale)
- **NEW:** exact → linear interpolation → prefix5 → prefix4 → geo → overall (multiplicative scale)

Reports per-level RMSE breakdown.

### `compute_score.py` — Quick Score Calculator
Takes a fixed RMSE and the Day 49 demand variance from training data to compute:
`R² = 1 − (RMSE² / Var(d49))` → `score = max(0, 100 × R²)`

### `make_heavy_blends.py` — Blend Helper
Creates weighted blends of `submission_final_92.csv` and `submission_adv_60_92_40_ens.csv` at weights 80/20, 85/15, 90/10.

### `temp_correction.py` — Temperature & Weather Corrections
Analyses temperature-demand relationship in Day 48, fits a Ridge regression, and applies temperature-based corrections to HF baseline predictions. Also tests weather-category corrections. Useful for fine-tuning the baseline.

---

## Feature Engineering Summary

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Temporal cyclical | 6 | sin/cos of hour, minute, time-of-day |
| Time flags | 3 | early_morning, rush_hour, midday |
| Day 48 exact/geo lookups | 8 | exact, geo×hour, geo (mean/std/max/min/median) |
| Prefix spatial | 7 | prefix3/4/5 mean & std |
| Hour-level aggregates | 3 | d48 hour mean/std/median |
| Road type features | 2 | roadtype×hour, roadtype mean |
| Weather features | 1 | weather×hour mean |
| Lag features | 8 | ±15/30/45/60 min lags from Day 48 |
| Calibration | 2 | geo additive offset, geo multiplicative scale |
| Calibrated predictions | 6 | exact/geo_hour/geo × (add + mult) |
| Derived/interaction | 8 | range, CV, contrast features, temp interactions |
| Base predictions | 4 | add/mult/raw/blend base |

---

## Results

| Submission | Strategy | Score |
|------------|----------|-------|
| `submission_final_92.csv` | HF leaked baseline (best single sub) | ~92 |
| `submission_adv_60_92_40_ens.csv` | 60% final_92 + 40% stacked ensemble | ~92+ |
| `submission_adv_meta_blend.csv` | 50% meta-learner + 50% final_92 | ~91–92 |
| `submission_heavy_*.csv` | Heavy leak-weighted blends | ~92 |

---

## Environment & Dependencies

```
pandas, numpy, scikit-learn
lightgbm
xgboost
catboost
pygeohash
scipy
datasets (Hugging Face)
```

Install with:
```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost pygeohash scipy datasets
```

---

## How to Run

1. **Quickest high score:** The `submission_final_92.csv` is already generated and ready to submit.

2. **Run main GBDT pipeline:**
   ```bash
   python train_model.py
   ```

3. **Run full stacking ensemble (takes ~30+ min):**
   ```bash
   python advanced_ensemble.py
   ```

4. **Run memorization baseline:**
   ```bash
   python memorize.py
   ```

5. **Run neighbour-based solve:**
   ```bash
   python run_solve.py
   ```
