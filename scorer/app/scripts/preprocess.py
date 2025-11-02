import argparse
import os
import numpy as np
import pandas as pd
from app.common.io_utils import read_config, read_parquet, write_parquet

def parse_datetime_features(df: pd.DataFrame, src_col: str) -> pd.DataFrame:
    targets = [
        f"{src_col}_year",
        f"{src_col}_month",
        f"{src_col}_day",
        f"{src_col}_dow",
        f"{src_col}_hour",
    ]
    if src_col in df.columns:
        dt = pd.to_datetime(df[src_col], errors="coerce", utc=True)
        df[f"{src_col}_year"] = dt.dt.year
        df[f"{src_col}_month"] = dt.dt.month
        df[f"{src_col}_day"] = dt.dt.day
        df[f"{src_col}_dow"] = dt.dt.weekday
        df[f"{src_col}_hour"] = dt.dt.hour
    for c in targets:
        if c not in df.columns:
            df[c] = np.nan
    return df

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.radians(lat1.astype(float))
    p2 = np.radians(lat2.astype(float))
    dlon = np.radians(lon2.astype(float) - lon1.astype(float))
    dphi = p2 - p1
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def ensure_cols(df: pd.DataFrame, cols, fill_value=np.nan, dtype=None):
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
        if dtype is not None:
            try:
                df[c] = df[c].astype(dtype)
            except Exception:
                pass
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_config(args.config)
    test_path = os.path.join(args.work_dir, cfg["paths"]["intermediate_test_parquet"])
    out_path  = os.path.join(args.work_dir, cfg["paths"]["preprocessed_features_parquet"])

    df = read_parquet(test_path)

    cat_cols = [
        "merch","cat_id","name_1","name_2","gender","street","one_city","us_state","post_code"
    ]

    dt_src = "transaction_time"
    dt_feats = [
        f"{dt_src}_year",
        f"{dt_src}_month",
        f"{dt_src}_day",
        f"{dt_src}_dow",
        f"{dt_src}_hour",
    ]

    base_num_cols = [
        "amount","lat","lon","population_city","jobs","merchant_lat","merchant_lon"
    ]

    extra_num_cols = [
        "distance_km","merch_amount_mean","cat_amount_mean"
    ]

    num_cols = base_num_cols + dt_feats + extra_num_cols

    df = ensure_cols(df, base_num_cols, fill_value=np.nan)
    df = ensure_cols(df, cat_cols, fill_value="missing", dtype="string")

    df = parse_datetime_features(df, dt_src)

    has_coords = {"lat","lon","merchant_lat","merchant_lon"}.issubset(df.columns)
    if has_coords:
        for c in ["lat","lon","merchant_lat","merchant_lon"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["distance_km"] = haversine(df["lat"], df["lon"], df["merchant_lat"], df["merchant_lon"])
    else:
        df["distance_km"] = np.nan

    if "amount" not in df.columns:
        df["amount"] = np.nan
    amount_num = pd.to_numeric(df["amount"], errors="coerce")
    global_med = float(np.nanmedian(amount_num.values)) if amount_num.notna().any() else 0.0

    try:
        agg_merch = df.groupby("merch", dropna=False)["amount"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    except Exception:
        agg_merch = pd.Series(dtype=float)
    df["merch_amount_mean"] = df["merch"].map(agg_merch).astype(float)
    df["merch_amount_mean"] = pd.to_numeric(df["merch_amount_mean"], errors="coerce").fillna(global_med)

    try:
        agg_cat = df.groupby("cat_id", dropna=False)["amount"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    except Exception:
        agg_cat = pd.Series(dtype=float)
    df["cat_amount_mean"] = df["cat_id"].map(agg_cat).astype(float)
    df["cat_amount_mean"] = pd.to_numeric(df["cat_amount_mean"], errors="coerce").fillna(global_med)

    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        med = float(np.nanmedian(df[c].values)) if df[c].notna().any() else 0.0
        df[c] = df[c].fillna(med)

    for c in cat_cols:
        if c not in df.columns:
            df[c] = "missing"
        try:
            df[c] = df[c].astype("string")
        except Exception:
            pass
        df[c] = df[c].fillna("missing")

    features = num_cols + cat_cols
    feat_df = df[features].copy()

    write_parquet(feat_df, out_path)
    print(f"Preprocessed -> X shape: {feat_df.shape} (num_cols={len(num_cols)}, cat_cols={len(cat_cols)})")

if __name__ == "__main__":
    main()