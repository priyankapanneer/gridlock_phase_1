"""
Fast vectorized CV for calibration experiments on Day 49.

Instead of row-by-row loops, uses pandas joins + numpy for speed.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv("train.csv")
parts = train["timestamp"].str.split(":", expand=True)
train["hour"] = parts[0].astype(int)
train["minute"] = parts[1].astype(int)
train["t"] = train["hour"] * 60 + train["minute"]

d48 = train[train["day"] == 48].copy()
d49 = train[train["day"] == 49].copy()

# ── Build base predictions for d49 via vectorized fallbacks ─────────────────
# Step 1: exact match
d48_exact = d48.set_index(["geohash", "hour", "minute"])["demand"].rename("d48_exact")
d49_with_base = d49.join(d48_exact, on=["geohash", "hour", "minute"])

# Step 2: for non-exact rows, do interpolation via merge + searchsorted
# Build geo_ts as a DataFrame
d48_ts = d48[["geohash", "t", "demand"]].sort_values(["geohash", "t"])

# For each non-exact row, find prev/next t within same geohash
non_exact_mask = d49_with_base["d48_exact"].isna()
non_exact_rows = d49_with_base[non_exact_mask][["geohash", "t"]].copy()

# Merge on geohash to get all d48 timestamps for that geohash
merged = non_exact_rows.reset_index().merge(
    d48_ts.rename(columns={"t": "t48", "demand": "d48_demand"}),
    on="geohash", how="left"
)

# For each row, find prev (largest t48 <= t) and next (smallest t48 > t)
merged["is_prev"] = merged["t48"] <= merged["t"]
merged["is_next"] = merged["t48"] > merged["t"]

prev = (merged[merged["is_prev"]]
        .sort_values("t48")
        .groupby("index")
        .last()[["t48", "d48_demand"]]
        .rename(columns={"t48": "t_prev", "d48_demand": "d_prev"}))

nxt = (merged[merged["is_next"]]
       .sort_values("t48")
       .groupby("index")
       .first()[["t48", "d48_demand"]]
       .rename(columns={"t48": "t_next", "d48_demand": "d_next"}))

interp = prev.join(nxt)
orig_t = d49_with_base[non_exact_mask]["t"]
interp["t"] = orig_t

# Linear interpolation where both prev and next exist
both = interp.dropna()
alpha = (both["t"] - both["t_prev"]) / (both["t_next"] - both["t_prev"])
alpha = alpha.clip(0, 1)
both_pred = both["d_prev"] + alpha * (both["d_next"] - both["d_prev"])

# Only prev (extrapolate left)
only_prev = interp[interp["d_next"].isna() & interp["d_prev"].notna()]["d_prev"]
# Only next (extrapolate right)
only_next = interp[interp["d_prev"].isna() & interp["d_next"].notna()]["d_next"]

interp_preds = pd.concat([both_pred, only_prev, only_next]).rename("interp_pred")
d49_with_base = d49_with_base.join(interp_preds)

# Step 3: prefix and geo fallbacks
geo_mean = d48.groupby("geohash")["demand"].mean().rename("geo_mean")
d49_with_base = d49_with_base.join(geo_mean, on="geohash")

d48["prefix5"] = d48["geohash"].str[:5]
d48["prefix4"] = d48["geohash"].str[:4]
prefix5_mean = d48.groupby("prefix5")["demand"].mean().rename("prefix5_mean")
prefix4_mean = d48.groupby("prefix4")["demand"].mean().rename("prefix4_mean")

d49_with_base["prefix5"] = d49_with_base["geohash"].str[:5]
d49_with_base["prefix4"] = d49_with_base["geohash"].str[:4]
d49_with_base = d49_with_base.join(prefix5_mean, on="prefix5")
d49_with_base = d49_with_base.join(prefix4_mean, on="prefix4")

overall = float(d48["demand"].mean())

# Combine fallbacks
d49_with_base["base"] = (
    d49_with_base["d48_exact"]
    .combine_first(d49_with_base["interp_pred"])
    .combine_first(d49_with_base["prefix5_mean"])
    .combine_first(d49_with_base["prefix4_mean"])
    .combine_first(d49_with_base["geo_mean"])
    .fillna(overall)
)

print(f"Day 49 rows: {len(d49_with_base)}")
print(f"Base coverage: {d49_with_base['base'].notna().mean():.4f}")
print()

# ── Build calibration stats ───────────────────────────────────────────────────
d48_idx = d48.set_index(["geohash", "hour", "minute"])[["demand"]].rename(columns={"demand": "d48_demand"})
d49_idx = d49.set_index(["geohash", "hour", "minute"])[["demand"]].rename(columns={"demand": "d49_demand"})
common = d48_idx.join(d49_idx, how="inner").reset_index()

geo_stats = common.groupby("geohash").agg(
    d48_mean=("d48_demand", "mean"),
    d49_mean=("d49_demand", "mean"),
    n=("d48_demand", "count"),
)

global_d49 = float(d49["demand"].mean())
global_d48 = float(d48["demand"].mean())
global_offset = global_d49 - global_d48

actuals = d49_with_base["demand"].values
base = d49_with_base["base"].values

# ── Vectorized CV for different calibration strategies ────────────────────────
def evaluate(preds, label):
    preds = np.clip(preds, 0, 1)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    print(f"{label}: RMSE={rmse:.6f}, pred_mean={preds.mean():.4f}")
    return rmse

geo_key = d49_with_base["geohash"].values

print("=== Multiplicative: different k and max_clip ===")
best_rmse = 999
best_label = ""
for k in [0, 0.5, 1, 1.5, 2, 3, 5]:
    for max_clip in [2.0, 3.0, 5.0, 10.0]:
        num = geo_stats["n"] * geo_stats["d49_mean"] + k * global_d49
        den = geo_stats["n"] * geo_stats["d48_mean"] + k * global_d48
        geo_scale = (num / den).clip(0.1, max_clip)
        global_scale = float(np.clip(global_d49 / global_d48, 0.5, max_clip))

        scale_arr = pd.Series(geo_scale).reindex(geo_key).fillna(global_scale).values
        preds = base * scale_arr
        label = f"mult k={k} clip={max_clip}"
        rmse = evaluate(preds, label)
        if rmse < best_rmse:
            best_rmse = rmse
            best_label = label

print(f"\nBest multiplicative: {best_label} -> RMSE={best_rmse:.6f}")

print("\n=== Additive: different k ===")
for k in [0, 1, 2, 5, 10, 20]:
    raw_offset = geo_stats["d49_mean"] - geo_stats["d48_mean"]
    geo_offset = (geo_stats["n"] * raw_offset + k * global_offset) / (geo_stats["n"] + k)
    global_off = global_offset

    offset_arr = pd.Series(geo_offset).reindex(geo_key).fillna(global_off).values
    preds = base + offset_arr
    evaluate(preds, f"add k={k}")

print("\n=== Blend (best mult + best add, k=1 each) ===")
k = 1
num = geo_stats["n"] * geo_stats["d49_mean"] + k * global_d49
den = geo_stats["n"] * geo_stats["d48_mean"] + k * global_d48
geo_scale_k1 = (num / den).clip(0.1, 10.0)
global_scale_k1 = float(np.clip(global_d49 / global_d48, 0.5, 5.0))
scale_arr_k1 = pd.Series(geo_scale_k1).reindex(geo_key).fillna(global_scale_k1).values

raw_offset = geo_stats["d49_mean"] - geo_stats["d48_mean"]
geo_offset_k1 = (geo_stats["n"] * raw_offset + k * global_offset) / (geo_stats["n"] + k)
offset_arr_k1 = pd.Series(geo_offset_k1).reindex(geo_key).fillna(global_offset).values

mult_preds = base * scale_arr_k1
add_preds = base + offset_arr_k1

for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    preds = alpha * mult_preds + (1 - alpha) * add_preds
    evaluate(preds, f"blend alpha={alpha}")
