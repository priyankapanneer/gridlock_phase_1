"""
Pure Memorization Model for Gridlock Traffic Demand
====================================================
Strategy: No ML model. 100% lookup table + calibration.

The test set is Day 49 (hours 2:15 onwards).
We have:
  - Day 49 train (hours 0:0 - 2:0): used to compute per-geohash day-over-day offset
  - Day 48 (full day): used as the primary lookup source

For each test row:
  1. Try exact lookup: d48[geohash, timestamp] + per_geohash_offset
  2. Fallback: d48[geohash, hour] mean + per_geohash_offset
  3. Fallback: d48[geohash] mean + global_offset
  4. Fallback: prefix5/4/3 mean + global_offset
  5. Fallback: global mean

This is essentially memorizing the training data and applying
a calibration shift — maximally overfit to training patterns.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path('d:/gridlock')

print("Loading data...")
train = pd.read_csv(BASE / 'train.csv')
test  = pd.read_csv(BASE / 'test.csv')
train['demand'] = pd.to_numeric(train['demand'], errors='coerce')

# Parse timestamps
def parse_ts(df):
    df = df.copy()
    parts = df['timestamp'].str.split(':', expand=True)
    df['hour']   = parts[0].astype(int)
    df['minute'] = parts[1].astype(int)
    return df

train = parse_ts(train)
test  = parse_ts(test)

d48 = train[train['day'] == 48].copy()
d49 = train[train['day'] == 49].copy()

print(f"Day 48: {len(d48)} rows | Day 49 train: {len(d49)} rows | Test: {len(test)} rows")
print(f"Day 48 timestamps: {sorted(d48['timestamp'].unique())[:5]}...")
print(f"Day 49 timestamps: {sorted(d49['timestamp'].unique())}")
print(f"Test  timestamps:  {sorted(test['timestamp'].unique())[:5]}...")

# ── 1. BUILD DAY-48 LOOKUP TABLES ──────────────────────────────────────────────

# Exact: (geohash, timestamp) -> demand
d48_exact = d48.groupby(['geohash', 'timestamp'])['demand'].mean()

# (geohash, hour) -> mean demand
d48_geo_hour = d48.groupby(['geohash', 'hour'])['demand'].mean()

# geohash -> mean demand
d48_geo = d48.groupby('geohash')['demand'].mean()

# Prefix lookups
d48['prefix5'] = d48['geohash'].str[:5]
d48['prefix4'] = d48['geohash'].str[:4]
d48['prefix3'] = d48['geohash'].str[:3]
prefix5_mean = d48.groupby('prefix5')['demand'].mean()
prefix4_mean = d48.groupby('prefix4')['demand'].mean()
prefix3_mean = d48.groupby('prefix3')['demand'].mean()
overall_mean = float(d48['demand'].mean())

print(f"\nOverall mean demand (day 48): {overall_mean:.6f}")

# ── 2. COMPUTE PER-GEOHASH CALIBRATION OFFSETS FROM DAY 49 ────────────────────

# Day 49 early-morning timestamps that overlap with day 48
d49_ts = set(d49['timestamp'].unique())
d48_early = d48[d48['timestamp'].isin(d49_ts)].copy()

# Merge to get paired (d48, d49) demand for same (geohash, timestamp)
merged = d49.merge(
    d48_early[['geohash', 'timestamp', 'demand']].rename(columns={'demand': 'd48_demand'}),
    on=['geohash', 'timestamp'],
    how='inner'
)
print(f"Paired rows for calibration: {len(merged)} covering {merged['geohash'].nunique()} geohashes")

# Per-geohash additive offset: how much does day 49 differ from day 48?
geo_offset = (merged['demand'] - merged['d48_demand']).groupby(merged['geohash']).mean()

# Global fallback offset
global_d48_early_mean = float(d48_early['demand'].mean())
global_d49_mean = float(d49['demand'].mean())
global_offset = global_d49_mean - global_d48_early_mean
print(f"Global offset (d49 - d48 early): {global_offset:.6f}")
print(f"Per-geohash offset: mean={geo_offset.mean():.4f}, std={geo_offset.std():.4f}, n={len(geo_offset)}")

# Also compute per-geohash multiplicative scale
geo_d48_mean = merged.groupby('geohash')['d48_demand'].mean()
geo_d49_mean = merged.groupby('geohash')['demand'].mean()
# Only use scale where d48 mean is non-zero
geo_scale = (geo_d49_mean / geo_d48_mean.replace(0, np.nan)).clip(0.1, 10.0)
global_scale = global_d49_mean / global_d48_early_mean if global_d48_early_mean > 0 else 1.0
print(f"Global scale (d49 / d48 early): {global_scale:.6f}")

# ── 3. BUILD TEST LOOKUP TABLES ───────────────────────────────────────────────

test = test.copy()
test['prefix5'] = test['geohash'].str[:5]
test['prefix4'] = test['geohash'].str[:4]
test['prefix3'] = test['geohash'].str[:3]

# Look up day-48 exact demand
test['d48_exact']    = test.set_index(['geohash', 'timestamp']).index.map(d48_exact.to_dict())
test['d48_geo_hour'] = test.set_index(['geohash', 'hour']).index.map(d48_geo_hour.to_dict())
test['d48_geo']      = test['geohash'].map(d48_geo)
test['p5']           = test['prefix5'].map(prefix5_mean)
test['p4']           = test['prefix4'].map(prefix4_mean)
test['p3']           = test['prefix3'].map(prefix3_mean)

# Calibration per geohash
test['offset'] = test['geohash'].map(geo_offset).fillna(global_offset)
test['scale']  = test['geohash'].map(geo_scale).fillna(global_scale)

# ── 4. PREDICTION: CASCADING FALLBACK WITH CALIBRATION ────────────────────────

# Additive calibrated predictions at each level
test['pred_exact']    = test['d48_exact']    + test['offset']
test['pred_geo_hour'] = test['d48_geo_hour'] + test['offset']
test['pred_geo']      = test['d48_geo']      + test['offset']
test['pred_p5']       = test['p5']           + test['offset']
test['pred_p4']       = test['p4']           + test['offset']
test['pred_p3']       = test['p3']           + test['offset']
test['pred_global']   = overall_mean         + global_offset

# Final prediction: exact lookup, then cascading fallback
test['prediction'] = (
    test['pred_exact']
    .fillna(test['pred_geo_hour'])
    .fillna(test['pred_p5'])
    .fillna(test['pred_p4'])
    .fillna(test['pred_geo'])
    .fillna(test['pred_p3'])
    .fillna(test['pred_global'])
)
test['prediction'] = test['prediction'].clip(0.0, 1.0)

# Coverage stats
exact_mask    = test['d48_exact'].notna()
geo_h_mask    = test['d48_geo_hour'].notna() & ~exact_mask
geo_mask      = test['d48_geo'].notna() & ~exact_mask & ~geo_h_mask
prefix_mask   = ~exact_mask & ~geo_h_mask & ~geo_mask

print(f"\nCoverage breakdown:")
print(f"  Exact (geohash+timestamp): {exact_mask.sum():>6} ({exact_mask.mean()*100:.1f}%)")
print(f"  Geo+hour fallback:         {geo_h_mask.sum():>6} ({geo_h_mask.mean()*100:.1f}%)")
print(f"  Geo fallback:              {geo_mask.sum():>6} ({geo_mask.mean()*100:.1f}%)")
print(f"  Prefix fallback:           {prefix_mask.sum():>6} ({prefix_mask.mean()*100:.1f}%)")

# ── 5. MULTIPLICATIVE VERSION (ALTERNATIVE) ────────────────────────────────────

test['pred_mult_exact']    = test['d48_exact']    * test['scale']
test['pred_mult_geo_hour'] = test['d48_geo_hour'] * test['scale']
test['pred_mult_geo']      = test['d48_geo']      * test['scale']

test['prediction_mult'] = (
    test['pred_mult_exact']
    .fillna(test['pred_mult_geo_hour'])
    .fillna(test['pred_mult_geo'])
    .fillna(test['pred_p5'])
    .fillna(overall_mean * global_scale)
)
test['prediction_mult'] = test['prediction_mult'].clip(0.0, 1.0)

# Blend additive + multiplicative
test['prediction_blend'] = 0.5 * test['prediction'] + 0.5 * test['prediction_mult']
test['prediction_blend'] = test['prediction_blend'].clip(0.0, 1.0)

# ── 6. SAVE SUBMISSIONS ────────────────────────────────────────────────────────

print("\nPrediction stats:")
for col, label in [('prediction', 'Additive'), ('prediction_mult', 'Multiplicative'), ('prediction_blend', 'Blend')]:
    vals = test[col].values
    print(f"  {label}: mean={vals.mean():.6f}, std={vals.std():.6f}, min={vals.min():.6f}, max={vals.max():.6f}")

test_ids = test['Index'].values

# Save all 3 variants
for col, name in [
    ('prediction',       'memorize_additive'),
    ('prediction_mult',  'memorize_mult'),
    ('prediction_blend', 'memorize_blend'),
]:
    out = BASE / f'submission_{name}.csv'
    pd.DataFrame({'Index': test_ids, 'demand': test[col].values}).to_csv(out, index=False)
    print(f"Saved: {out.name}")

print("\nDone! Best pick: submission_memorize_blend.csv or submission_memorize_additive.csv")
print("These are pure lookup-table predictions — zero ML, maximum memorization.")
