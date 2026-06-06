import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLUMNS = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]

def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["timestamp"].fillna("0:0").astype(str).str.split(":", n=1, expand=True)
    df = df.copy()
    df["hour"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    df["minute"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
    df["t"] = df["hour"] * 60 + df["minute"]
    return df

def main():
    base_dir = Path("d:/gridlock")
    train_df = pd.read_csv(base_dir / "train.csv")
    test_df = pd.read_csv(base_dir / "test.csv")
    train_df["demand"] = pd.to_numeric(train_df["demand"], errors="coerce")
    test_ids = test_df["Index"].copy()

    # Load baseline final_92
    final_92_path = base_dir / "submission_final_92.csv"
    if final_92_path.exists():
        final_92 = pd.read_csv(final_92_path)["demand"].values
    else:
        raise ValueError("final_92 baseline not found")

    train_df = parse_timestamp(train_df)
    test_df = parse_timestamp(test_df)

    d48 = train_df[train_df["day"] == 48].copy()
    d49 = train_df[train_df["day"] == 49].copy()

    # Lookups
    d48_exact = d48.set_index(["geohash", "hour", "minute"])["demand"].rename("d48_exact")
    d48_geo_hour = d48.groupby(["geohash", "hour"])["demand"].mean().rename("d48_geo_hour")
    d48_geo = d48.groupby("geohash")["demand"].mean().rename("d48_geo")
    
    d48["prefix5"] = d48["geohash"].str[:5]
    d48["prefix4"] = d48["geohash"].str[:4]
    prefix5_mean = d48.groupby("prefix5")["demand"].mean().rename("prefix5_mean")
    prefix4_mean = d48.groupby("prefix4")["demand"].mean().rename("prefix4_mean")
    overall_mean = float(d48["demand"].mean())
    d48_t = d48.groupby(["geohash", "t"])["demand"].mean().rename("d48_lag")

    global_offset = float(d49["demand"].mean()) - float(d48[d48["hour"] <= 2]["demand"].mean())
    d48_idx = d48.set_index(["geohash", "hour", "minute"])[["demand"]].rename(columns={"demand": "d48_demand"})
    d49_idx = d49.set_index(["geohash", "hour", "minute"])[["demand"]].rename(columns={"demand": "d49_demand"})
    common = d48_idx.join(d49_idx, how="inner").reset_index()
    geo_stats = common.groupby("geohash").agg(d48_mean=("d48_demand", "mean"), d49_mean=("d49_demand", "mean"), n=("d48_demand", "count"))
    
    k = 0
    geo_offset = (geo_stats["n"] * (geo_stats["d49_mean"] - geo_stats["d48_mean"]) + k * global_offset) / (geo_stats["n"] + k)

    def process_features(df_orig):
        df = df_orig.copy()
        df["prefix5"] = df["geohash"].str[:5]
        df["prefix4"] = df["geohash"].str[:4]
        
        df = df.join(d48_exact, on=["geohash", "hour", "minute"])
        df = df.join(d48_geo_hour, on=["geohash", "hour"])
        df = df.join(d48_geo, on="geohash")
        df = df.join(prefix5_mean, on="prefix5")
        df = df.join(prefix4_mean, on="prefix4")
        
        df["geo_offset"] = df["geohash"].map(geo_offset).fillna(global_offset)
        
        df["d48_exact"] = df["d48_exact"].fillna(df["d48_geo_hour"]).fillna(df["prefix5_mean"]).fillna(df["prefix4_mean"]).fillna(df["d48_geo"]).fillna(overall_mean)
        df["d48_geo_hour"] = df["d48_geo_hour"].fillna(df["prefix5_mean"]).fillna(df["prefix4_mean"]).fillna(df["d48_geo"]).fillna(overall_mean)
        df["prefix5_mean"] = df["prefix5_mean"].fillna(df["prefix4_mean"]).fillna(df["d48_geo"]).fillna(overall_mean)
        df["prefix4_mean"] = df["prefix4_mean"].fillna(df["d48_geo"]).fillna(overall_mean)
        df["d48_geo"] = df["d48_geo"].fillna(overall_mean)
        
        for lag in [-15, 15, -30, 30]:
            name = f"lag_{lag}"
            df[name] = df["t"] + lag
            df = df.join(d48_t.rename(name+"_val"), on=["geohash", name])
            df[name+"_calib"] = df[name+"_val"].fillna(df["d48_geo_hour"]).fillna(overall_mean) + df["geo_offset"]
        
        df["exact_calib"] = df["d48_exact"] + df["geo_offset"]
        df["geo_hour_calib"] = df["d48_geo_hour"] + df["geo_offset"]
        df["geo_calib"] = df["d48_geo"] + df["geo_offset"]
        
        df["base_pred"] = 0.40 * df["exact_calib"] + 0.30 * df["geo_hour_calib"] + 0.30 * df["geo_calib"]
        
        return df

    train_ml = process_features(d49)
    test_ml = process_features(test_df)

    # Encode categoricals for Random Forest
    for col in CATEGORICAL_COLUMNS:
        train_ml[col] = train_ml[col].fillna("Missing").astype("category").cat.codes
        test_ml[col] = test_ml[col].fillna("Missing").astype("category").cat.codes

    features = [
        "hour", "minute", "t", "NumberofLanes", "Temperature",
        "exact_calib", "geo_hour_calib", "geo_calib", "geo_offset",
        "lag_-15_calib", "lag_15_calib", "lag_-30_calib", "lag_30_calib"
    ] + CATEGORICAL_COLUMNS

    # Handle missing numericals
    for col in features:
        train_ml[col] = train_ml[col].fillna(0)
        test_ml[col] = test_ml[col].fillna(0)

    X_train = train_ml[features].values
    y_train = (train_ml["demand"] - train_ml["base_pred"]).values
    X_test = test_ml[features].values
    base_test = test_ml["base_pred"].values

    print("Training RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = np.clip(base_test + rf.predict(X_test), 0, 1)

    print("Training ExtraTreesRegressor...")
    et = ExtraTreesRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    et.fit(X_train, y_train)
    et_preds = np.clip(base_test + et.predict(X_test), 0, 1)

    print("Training MLPRegressor (Neural Network)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    mlp_preds = np.clip(base_test + mlp.predict(X_test_scaled), 0, 1)

    print("Blending new models with final_92...")
    rf_blend = 0.5 * final_92 + 0.5 * rf_preds
    et_blend = 0.5 * final_92 + 0.5 * et_preds
    mlp_blend = 0.5 * final_92 + 0.5 * mlp_preds
    
    # Advanced Meta Blend of the new models
    mega_new = 0.4 * final_92 + 0.2 * rf_preds + 0.2 * et_preds + 0.2 * mlp_preds

    pd.DataFrame({"Index": test_ids, "demand": rf_blend}).to_csv(base_dir / "submission_rf.csv", index=False)
    pd.DataFrame({"Index": test_ids, "demand": et_blend}).to_csv(base_dir / "submission_et.csv", index=False)
    pd.DataFrame({"Index": test_ids, "demand": mlp_blend}).to_csv(base_dir / "submission_mlp.csv", index=False)
    pd.DataFrame({"Index": test_ids, "demand": mega_new}).to_csv(base_dir / "submission_other_mega.csv", index=False)
    
    print("Done! Saved submission_rf.csv, submission_et.csv, submission_mlp.csv, and submission_other_mega.csv")

if __name__ == "__main__":
    main()
