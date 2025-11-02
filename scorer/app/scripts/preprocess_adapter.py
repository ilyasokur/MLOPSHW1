import numpy as np
import pandas as pd

from app.scripts.preprocess import (
    parse_datetime_features,
    haversine,
    ensure_cols,
)

CAT_COLS = ["merch","cat_id","name_1","name_2","gender","street","one_city","us_state","post_code"]
DT_SRC = "transaction_time"
DT_FEATS = [f"{DT_SRC}_{x}" for x in ["year","month","day","dow","hour"]]
BASE_NUM_COLS = ["amount","lat","lon","population_city","jobs","merchant_lat","merchant_lon"]
EXTRA_NUM_COLS = ["distance_km","merch_amount_mean","cat_amount_mean"]
NUM_COLS = BASE_NUM_COLS + DT_FEATS + EXTRA_NUM_COLS

def preprocess_event(rec: dict) -> pd.DataFrame:
    df = pd.DataFrame([rec])

    df = ensure_cols(df, BASE_NUM_COLS, fill_value=np.nan)
    df = ensure_cols(df, CAT_COLS, fill_value="missing", dtype="string")

    df = parse_datetime_features(df, DT_SRC)

    if {"lat","lon","merchant_lat","merchant_lon"}.issubset(df.columns):
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

    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        med = float(np.nanmedian(df[c].values)) if df[c].notna().any() else 0.0
        df[c] = df[c].fillna(med)

    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "missing"
        try:
            df[c] = df[c].astype("string")
        except Exception:
            pass
        df[c] = df[c].fillna("missing")

    return df[NUM_COLS + CAT_COLS]