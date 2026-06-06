import numpy as np
import pandas as pd
import pygeohash as pgh
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

DATA = 'e88186124ec611f1/dataset'

print("Loading data...")
train = pd.read_csv(f'{DATA}/train.csv')
test  = pd.read_csv(f'{DATA}/test.csv')

# ── Parse / encode ────────────────────────────────────────────────────────────
def parse_ts(df):
    df = df.copy()
    df['hour']   = df['timestamp'].map(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].map(lambda x: int(x.split(':')[1]))
    t = (df['hour'] * 60 + df['minute']) / (24 * 60) * 2 * np.pi
    df['time_sin'] = np.sin(t)
    df['time_cos'] = np.cos(t)
    d = df['day'] / 7 * 2 * np.pi
    df['day_sin'] = np.sin(d)
    df['day_cos'] = np.cos(d)
    return df

def decode_geo(df):
    df = df.copy()
    decoded  = df['geohash'].map(pgh.decode)
    df['lat'] = decoded.map(lambda x: x[0])
    df['lon'] = decoded.map(lambda x: x[1])
    df['geo3'] = df['geohash'].str[:3]
    return df

def encode_cats(df):
    df = df.copy()
    df['RoadType_enc']      = df['RoadType'].map({'Residential': 0, 'Street': 1, 'Highway': 2}).fillna(-1)
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(float)
    df['Landmarks_enc']     = (df['Landmarks'] == 'Yes').astype(float)
    df['Weather_enc']       = df['Weather'].map({'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}).fillna(-1)
    return df

train = parse_ts(train); train = decode_geo(train); train = encode_cats(train)
test  = parse_ts(test);  test  = decode_geo(test);  test  = encode_cats(test)

# ── Neighbor cache ────────────────────────────────────────────────────────────
print("Building neighbor cache...")
all_geohashes = list(set(train['geohash']) | set(test['geohash']))
neighbor_cache = {}
for gh in all_geohashes:
    t = pgh.get_adjacent(gh, 'top');    b = pgh.get_adjacent(gh, 'bottom')
    l = pgh.get_adjacent(gh, 'left');   r = pgh.get_adjacent(gh, 'right')
    tl = pgh.get_adjacent(t, 'left');   tr = pgh.get_adjacent(t, 'right')
    bl = pgh.get_adjacent(b, 'left');   br = pgh.get_adjacent(b, 'right')
    neighbor_cache[gh] = [t, b, l, r, tl, tr, bl, br]

def ts_offset(ts, delta_min):
    h, m = int(ts.split(':')[0]), int(ts.split(':')[1])
    total = (h * 60 + m + delta_min) % (24 * 60)
    return f"{total // 60}:{total % 60}"

# ── Core feature builder (given a reference day's lookup dict) ───────────────
def build_features(df, ref_df):
    """
    df      : dataframe to featurize
    ref_df  : training subset used to compute all statistics (no leakage)
    """
    df = df.copy()

    # Lag lookup from ref_df (day 48 in CV, all train for final)
    lag_lookup = dict(zip(zip(ref_df['geohash'], ref_df['timestamp']), ref_df['demand']))

    # Rolling lags: T, T-15, T-30, T-45, T-60
    for delta in [0, 15, 30, 45, 60]:
        col = 'lag1d' if delta == 0 else f'lag1d_m{delta}'
        if delta == 0:
            df[col] = [lag_lookup.get((gh, ts), np.nan)
                       for gh, ts in zip(df['geohash'], df['timestamp'])]
        else:
            df[col] = [lag_lookup.get((gh, ts_offset(ts, -delta)), np.nan)
                       for gh, ts in zip(df['geohash'], df['timestamp'])]

    # Neighbor mean at same timestamp from ref_df
    df['neighbor_mean'] = [
        np.nanmean([lag_lookup.get((n, ts), np.nan)
                    for n in neighbor_cache.get(gh, [])]) or np.nan
        for gh, ts in zip(df['geohash'], df['timestamp'])
    ]

    # Aggregations from ref_df
    geo_stats = (ref_df.groupby('geohash')['demand']
                 .agg(['mean','std','median','max']).reset_index()
                 .rename(columns={'mean':'geo_mean','std':'geo_std',
                                  'median':'geo_median','max':'geo_max'}))
    geo_stats['geo_std'] = geo_stats['geo_std'].fillna(0)
    geo_hour = (ref_df.groupby(['geohash','hour'])['demand']
                .mean().reset_index().rename(columns={'demand':'geo_hour_mean'}))
    geo_ts   = (ref_df.groupby(['geohash','timestamp'])['demand']
                .mean().reset_index().rename(columns={'demand':'geo_ts_mean'}))
    geo3_h   = (ref_df.groupby(['geo3','hour'])['demand']
                .mean().reset_index().rename(columns={'demand':'geo3_hour_mean'}))
    geo3_m   = (ref_df.groupby('geo3')['demand']
                .mean().reset_index().rename(columns={'demand':'geo3_mean'}))
    hr_mean  = (ref_df.groupby('hour')['demand']
                .mean().reset_index().rename(columns={'demand':'hour_global_mean'}))
    # Day-49 early hours baseline (if available in ref_df)
    day49_early = ref_df[ref_df['day'] == 49] if 'day' in ref_df.columns else ref_df.iloc[0:0]
    if len(day49_early):
        d49_base = (day49_early.groupby('geohash')['demand']
                    .mean().reset_index().rename(columns={'demand':'geo_d49_mean'}))
    else:
        d49_base = pd.DataFrame({'geohash': [], 'geo_d49_mean': []})

    df = df.merge(geo_stats, on='geohash', how='left')
    df = df.merge(geo_hour,  on=['geohash','hour'], how='left')
    df = df.merge(geo_ts,    on=['geohash','timestamp'], how='left')
    df = df.merge(geo3_h,    on=['geo3','hour'], how='left')
    df = df.merge(geo3_m,    on='geo3', how='left')
    df = df.merge(hr_mean,   on='hour', how='left')
    df = df.merge(d49_base,  on='geohash', how='left')

    # Daily ratio: day49_morning / day48_morning per geohash
    # Signals if today is busier/quieter than yesterday overall
    day48_am = (ref_df[ref_df['day'] == 48][ref_df['hour'] < 4] if len(ref_df[ref_df['day'] == 48]) else ref_df.iloc[0:0])
    day49_am = (day49_early[day49_early['hour'] < 4] if len(day49_early) else pd.DataFrame())

    if len(day48_am) and len(day49_am):
        d48_am_mean = day48_am.groupby('geohash')['demand'].mean().rename('d48_am')
        d49_am_mean = day49_am.groupby('geohash')['demand'].mean().rename('d49_am')
        ratio_df = pd.concat([d48_am_mean, d49_am_mean], axis=1).reset_index()
        ratio_df['daily_ratio'] = ratio_df['d49_am'] / ratio_df['d48_am'].replace(0, np.nan)
        ratio_df = ratio_df[['geohash', 'daily_ratio']]
        df = df.merge(ratio_df, on='geohash', how='left')
    else:
        df['daily_ratio'] = np.nan

    # Impute missing lags with fallback chain
    fallback = df['neighbor_mean'].fillna(df['geo_ts_mean']).fillna(df['geo_hour_mean'])
    for col in ['lag1d','lag1d_m15','lag1d_m30','lag1d_m45','lag1d_m60']:
        df[col] = df[col].fillna(fallback)

    return df

FEATURES = [
    'lat', 'lon', 'hour', 'minute', 'day',
    'time_sin', 'time_cos', 'day_sin', 'day_cos',
    'RoadType_enc', 'NumberofLanes', 'LargeVehicles_enc', 'Landmarks_enc',
    'Temperature', 'Weather_enc',
    'lag1d', 'lag1d_m15', 'lag1d_m30', 'lag1d_m45', 'lag1d_m60',
    'neighbor_mean',
    'geo_mean', 'geo_std', 'geo_median', 'geo_max',
    'geo_hour_mean', 'geo_ts_mean', 'geo3_hour_mean', 'geo3_mean',
    'hour_global_mean', 'geo_d49_mean', 'daily_ratio',
]

LGBM_PARAMS = dict(
    n_estimators=3000, learning_rate=0.02, num_leaves=255,
    min_child_samples=15, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.1,
    random_state=42, verbose=-1, n_jobs=-1,
)
XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.02, max_depth=8,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1,
    random_state=42, verbosity=0, n_jobs=-1, tree_method='hist',
)

# ── Proper CV: ref = day48 only ───────────────────────────────────────────────
print("\nBuilding CV features (ref=day48 only)...")
train48 = train[train['day'] == 48]
train49 = train[train['day'] == 49]

tr_cv = build_features(train48, train48)   # train on day48, featurized vs day48
va_cv = build_features(train49, train48)   # val on day49, but stats from day48 only

X_tr = tr_cv[FEATURES].fillna(-1);  y_tr = train48['demand'].values
X_va = va_cv[FEATURES].fillna(-1);  y_va = train49['demand'].values

print(f"CV — train: {X_tr.shape}  val: {X_va.shape}")

print("\nTraining LGBM CV...")
lgbm_cv = LGBMRegressor(**LGBM_PARAMS)
lgbm_cv.fit(X_tr, y_tr)
lp = lgbm_cv.predict(X_va)
lgbm_r2 = r2_score(y_va, lp)
print(f"LGBM CV R2: {lgbm_r2:.4f}  score: {max(0, 100*lgbm_r2):.2f}")

print("\nTraining XGB CV...")
xgb_cv = XGBRegressor(**XGB_PARAMS)
xgb_cv.fit(X_tr, y_tr)
xp = xgb_cv.predict(X_va)
xgb_r2 = r2_score(y_va, xp)
print(f"XGB CV R2:  {xgb_r2:.4f}  score: {max(0, 100*xgb_r2):.2f}")

best_w, best_r2 = 0, -999
for w in np.arange(0, 1.05, 0.05):
    r2 = r2_score(y_va, w * lp + (1 - w) * xp)
    if r2 > best_r2:
        best_r2, best_w = r2, w
print(f"\nBest blend {best_w:.2f}*LGBM + {1-best_w:.2f}*XGB: R2={best_r2:.4f}  score={max(0, 100*best_r2):.2f}")

# ── Final model: ref = ALL train ──────────────────────────────────────────────
print("\nBuilding final features (ref=all train)...")
train_full = build_features(train, train)
test_full  = build_features(test,  train)

X_all  = train_full[FEATURES].fillna(-1)
y_all  = train['demand'].values
X_test = test_full[FEATURES].fillna(-1)

print("Training final LGBM...")
lgbm_f = LGBMRegressor(**LGBM_PARAMS)
lgbm_f.fit(X_all, y_all)

print("Training final XGB...")
xgb_f = XGBRegressor(**XGB_PARAMS)
xgb_f.fit(X_all, y_all)

preds = np.clip(
    best_w * lgbm_f.predict(X_test) + (1 - best_w) * xgb_f.predict(X_test),
    0, None
)

submission = pd.DataFrame({'Index': test['Index'], 'demand': preds})
submission.to_csv('submission.csv', index=False)
print(f"\nSaved submission.csv  ({len(submission)} rows)")
print(submission.head())

fi = pd.Series(lgbm_f.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop feature importances:")
print(fi.head(15))
