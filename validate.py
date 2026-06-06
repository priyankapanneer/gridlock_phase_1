"""
Cross-validation comparing old vs new prediction approaches on Day 49 held-out data.
Validates: exact lookup, interpolation fallback, prefix similarity, calibration.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

train = pd.read_csv("train.csv")

parts = train["timestamp"].str.split(":", expand=True)
train["hour"] = parts[0].astype(int)
train["minute"] = parts[1].astype(int)

d48 = train[train["day"] == 48].copy()
d49 = train[train["day"] == 49].copy()

# ── Build lookup structures ──────────────────────────────────────────────────
exact = d48.set_index(["geohash", "hour", "minute"])["demand"]
geo = d48.groupby("geohash")["demand"].mean()
overall = float(d48["demand"].mean())

# Per-geohash sorted time-series
geo_ts = {}
for (gh, h, m), demand in exact.items():
    t = h * 60 + m
    geo_ts.setdefault(gh, []).append((t, demand))
for gh in geo_ts:
    geo_ts[gh] = np.array(sorted(geo_ts[gh]))

# Prefix maps
prefix5_mean = {}
prefix4_mean = {}
for gh, d in geo.items():
    prefix5_mean.setdefault(gh[:5], []).append(d)
    prefix4_mean.setdefault(gh[:4], []).append(d)
prefix5_mean = {p: float(np.mean(v)) for p, v in prefix5_mean.items()}
prefix4_mean = {p: float(np.mean(v)) for p, v in prefix4_mean.items()}

# ── Calibration ──────────────────────────────────────────────────────────────
global_d48 = float(d48["demand"].mean())
global_d49 = float(d49["demand"].mean())
global_scale = float(np.clip(global_d49 / global_d48, 0.5, 3.0))

d48_idx = d48.set_index(["geohash","hour","minute"])[["demand"]].rename(columns={"demand":"d48_demand"})
d49_idx = d49.set_index(["geohash","hour","minute"])[["demand"]].rename(columns={"demand":"d49_demand"})
common = d48_idx.join(d49_idx, how="inner").reset_index()

geo_stats = common.groupby("geohash").agg(
    d48_mean=("d48_demand","mean"),
    d49_mean=("d49_demand","mean"),
    n=("d48_demand","count"),
)
k = 1
num = geo_stats["n"] * geo_stats["d49_mean"] + k * global_d49
den = geo_stats["n"] * geo_stats["d48_mean"] + k * global_d48
geo_scale = (num / den).clip(0.1, 10.0)

# ── Predict day 49 with OLD approach ─────────────────────────────────────────
d48_geo_hour = d48.groupby(["geohash","hour"])["demand"].mean()

old_preds = []
new_preds = []
actuals = []
level_old = []
level_new = []

for _, row in d49.iterrows():
    gh, h, m = row["geohash"], row["hour"], row["minute"]
    t = h * 60 + m
    actual = row["demand"]
    actuals.append(actual)

    scale = geo_scale.get(gh, global_scale)
    if pd.isna(scale):
        scale = global_scale

    # OLD: exact → geo_hour → geo → overall
    p_old = exact.get((gh, h, m), np.nan)
    if np.isnan(p_old):
        p_old = d48_geo_hour.get((gh, h), np.nan)
        level_old.append("geo_hour" if not np.isnan(p_old) else ("geo" if not np.isnan(geo.get(gh, np.nan)) else "global"))
    else:
        level_old.append("exact")
    if np.isnan(p_old):
        p_old = geo.get(gh, overall)
    old_preds.append(max(0, p_old * scale))

    # NEW: exact → interpolation → prefix5 → prefix4 → geo → overall
    p_new = exact.get((gh, h, m), np.nan)
    if np.isnan(p_new):
        if gh in geo_ts:
            arr = geo_ts[gh]
            times = arr[:, 0]
            idx = int(np.searchsorted(times, t))
            if idx == 0:
                p_new = arr[0, 1]
            elif idx >= len(arr):
                p_new = arr[-1, 1]
            else:
                t0, d0, t1, d1 = times[idx-1], arr[idx-1,1], times[idx], arr[idx,1]
                alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                p_new = d0 + alpha * (d1 - d0)
            level_new.append("interp")
        elif gh[:5] in prefix5_mean:
            p_new = prefix5_mean[gh[:5]]
            level_new.append("prefix5")
        elif gh[:4] in prefix4_mean:
            p_new = prefix4_mean[gh[:4]]
            level_new.append("prefix4")
        else:
            p_new = geo.get(gh, overall)
            level_new.append("geo/global")
    else:
        level_new.append("exact")
    new_preds.append(max(0, p_new * scale))

actuals = np.array(actuals)
old_preds = np.array(old_preds)
new_preds = np.array(new_preds)

rmse_old = np.sqrt(mean_squared_error(actuals, old_preds))
rmse_new = np.sqrt(mean_squared_error(actuals, new_preds))

from collections import Counter
print("=== OLD APPROACH ===")
print(f"  RMSE: {rmse_old:.6f}")
print(f"  pred_mean={old_preds.mean():.4f}, actual_mean={actuals.mean():.4f}")
print(f"  Fallback levels: {dict(Counter(level_old))}")
print()
print("=== NEW APPROACH (with interpolation + prefix) ===")
print(f"  RMSE: {rmse_new:.6f}")
print(f"  pred_mean={new_preds.mean():.4f}, actual_mean={actuals.mean():.4f}")
print(f"  Fallback levels: {dict(Counter(level_new))}")
print()

# Per-level RMSE breakdown
for method, preds_arr, levels in [("OLD", old_preds, level_old), ("NEW", new_preds, level_new)]:
    print(f"--- {method} per-level RMSE ---")
    levels_arr = np.array(levels)
    for lvl in sorted(set(levels)):
        mask = levels_arr == lvl
        if mask.sum() > 0:
            rmse_lvl = np.sqrt(mean_squared_error(actuals[mask], preds_arr[mask]))
            print(f"  {lvl}: n={mask.sum()}, RMSE={rmse_lvl:.6f}")
    print()

improvement = (rmse_old - rmse_new) / rmse_old * 100
print(f"RMSE improvement: {improvement:.2f}%")
