"""
TEMPERATURE-BASED CORRECTION
=============================
Key insight from solve.py: Temperature is the #1 most important feature (68774).
Day 49 test rows have known temperatures. If temp-demand relationship is consistent,
we can apply a temperature correction to hf_avg predictions.

Also tries: Road type, number of lanes corrections.
"""
import pandas as pd
import numpy as np
import os

BASE = 'd:/gridlock'
train = pd.read_csv(f'{BASE}/train.csv')
test  = pd.read_csv(f'{BASE}/test.csv')
train['demand'] = pd.to_numeric(train['demand'], errors='coerce')
for df in [train, test]:
    p = df['timestamp'].str.split(':', expand=True)
    df['hour'] = p[0].astype(int)
    df['minute'] = p[1].astype(int)

d48 = train[train['day']==48].copy()
d49 = train[train['day']==49].copy()

# Load hf_avg
hf = [pd.read_csv(f'{BASE}/hf_parts/{f}')['demand'].values
      for f in os.listdir(f'{BASE}/hf_parts') if f.endswith('.csv')]
hf_avg = np.mean(hf, axis=0)
test_ids = test['Index'].values

print("=== FEATURE ANALYSIS ===")
print(f"\nTemperature distribution:")
print(f"  D48 temp: mean={d48['Temperature'].mean():.2f} std={d48['Temperature'].std():.2f}")
print(f"  D49 temp: mean={d49['Temperature'].mean():.2f} std={d49['Temperature'].std():.2f}")
print(f"  Test temp: mean={test['Temperature'].mean():.2f} std={test['Temperature'].std():.2f}")

# Temp-demand correlation in d48
from scipy import stats
corr, pval = stats.pearsonr(d48['Temperature'].dropna(), 
                             d48.loc[d48['Temperature'].notna(), 'demand'])
print(f"\nD48 Temp-Demand correlation: r={corr:.4f} p={pval:.2e}")

# Temp buckets - is demand different by temp range?
d48['temp_bucket'] = pd.cut(d48['Temperature'], bins=10)
temp_demand = d48.groupby('temp_bucket')['demand'].mean()
print("\nD48 demand by temperature bucket:")
for bucket, mean_d in temp_demand.items():
    print(f"  {str(bucket):<20}: {mean_d:.5f}")

# Weather effect
print("\nD48 demand by Weather:")
print(d48.groupby('Weather')['demand'].mean().to_string())
print("\nTest Weather distribution:")
print(test['Weather'].value_counts().to_string())
print("\nD48 Weather distribution:")
print(d48['Weather'].value_counts().to_string())

# Road type effect
print("\nD48 demand by RoadType:")
print(d48.groupby('RoadType')['demand'].mean().to_string())
print("\nTest RoadType distribution:")
print(test['RoadType'].value_counts().to_string())

# Number of lanes effect
print("\nD48 demand by NumberofLanes:")
print(d48.groupby('NumberofLanes')['demand'].mean().sort_index().to_string())
print("\nTest NumberofLanes distribution:")
print(test['NumberofLanes'].value_counts().sort_index().to_string())

# Key: does temperature differ between d48 and test FOR THE SAME geohash+hour?
test2 = test.copy()
test2 = test2.join(
    d48.groupby(['geohash','hour'])['Temperature'].mean().rename('d48_temp'),
    on=['geohash','hour']
)
test2['temp_diff'] = test2['Temperature'] - test2['d48_temp']
has_diff = test2['temp_diff'].notna()
print(f"\nGeohash+hour temp comparison (test vs d48):")
print(f"  Rows with d48 temp match: {has_diff.sum()} ({has_diff.mean()*100:.1f}%)")
print(f"  Mean temp diff (test - d48): {test2['temp_diff'].mean():.3f}")
print(f"  Std of temp diff: {test2['temp_diff'].std():.3f}")
print(f"  Rows where diff > 2: {(test2['temp_diff'].abs() > 2).sum()}")

# How much does 1 degree C change demand?
# Fit simple linear model: demand = a * temp + b per geohash
from sklearn.linear_model import Ridge
d48_clean = d48[d48['Temperature'].notna()].copy()
X = d48_clean[['Temperature']].values
y = d48_clean['demand'].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xs = sc.fit_transform(X)
reg = Ridge(alpha=1.0)
reg.fit(Xs, y)
print(f"\nGlobal temp coefficient: {reg.coef_[0]:.6f} demand/std_temp")
print(f"(1 std temp = {d48['Temperature'].std():.2f} degrees)")
print(f"Effect of 1 std temp on demand: {reg.coef_[0]:.5f}")

# APPROACH: Temperature-adjusted hf_avg
# 1. Find global temp-demand slope from d48
# 2. For test rows where temp differs from d48, apply correction
temp_slope = reg.coef_[0]  # demand change per std_temp
temp_std = d48['Temperature'].std()

# Apply correction: pred_corrected = hf_avg + slope * (test_temp_scaled - d48_temp_scaled_for_that_row)
test2['hf_avg'] = hf_avg
test2['test_temp_scaled'] = sc.transform(test2[['Temperature']])
d48_temp_by_gh_hr = d48.groupby(['geohash','hour'])['Temperature'].mean()
test2 = test2.join(d48_temp_by_gh_hr.rename('d48_temp_raw'), on=['geohash','hour'])
test2['d48_temp_scaled'] = (test2['d48_temp_raw'] - sc.mean_[0]) / sc.scale_[0]
test2['temp_correction'] = temp_slope * (test2['test_temp_scaled'] - test2['d48_temp_scaled'])
test2['temp_correction'] = test2['temp_correction'].fillna(0)

print(f"\nTemp correction stats:")
print(f"  Mean correction: {test2['temp_correction'].mean():.6f}")
print(f"  Std correction:  {test2['temp_correction'].std():.6f}")
print(f"  Max correction:  {test2['temp_correction'].abs().max():.6f}")

# Build submissions
submissions = {}

# Temperature-corrected at different strengths
for strength in [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]:
    corr = test2['temp_correction'] * strength
    pred = np.clip(hf_avg + corr.values, 0, 1)
    submissions[f'temp_corr_s{int(strength*100)}'] = pred
    print(f"  temp_corr_s{strength}: mean={pred.mean():.5f} corr_hf={np.corrcoef(pred,hf_avg)[0,1]:.5f}")

# Weather-based correction
weather_demand = d48.groupby('Weather')['demand'].mean()
test2['weather_demand_d48'] = test2['Weather'].map(weather_demand)
test2['hf_weather_ratio'] = test2['weather_demand_d48'] / d48['demand'].mean()
test2['hf_weather_ratio'] = test2['hf_weather_ratio'].fillna(1.0).clip(0.5, 2.0)

# Very gentle weather-based blend
for w in [0.05, 0.10]:
    pred = np.clip(
        hf_avg * (1 + w * (test2['hf_weather_ratio'].values - 1)),
        0, 1
    )
    submissions[f'weather_w{int(w*100)}'] = pred
    print(f"  weather_w{w}: mean={pred.mean():.5f} corr_hf={np.corrcoef(pred,hf_avg)[0,1]:.5f}")

print(f"\nSaving {len(submissions)} submissions...")
for name, v in submissions.items():
    pd.DataFrame({'Index': test_ids, 'demand': v}).to_csv(
        f'{BASE}/sub_temp_{name}.csv', index=False)
    
print("Done.")
print("\n=== BEST TO TRY ===")
print("sub_temp_temp_corr_s25.csv  -- 25% temperature correction")
print("sub_temp_temp_corr_s50.csv  -- 50% temperature correction")
print("sub_temp_weather_w5.csv     -- 5% weather correction")
