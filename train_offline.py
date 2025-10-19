import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

def parse_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[col], errors="coerce", utc=True)
    df[col + "_year"] = dt.dt.year
    df[col + "_month"] = dt.dt.month
    df[col + "_day"] = dt.dt.day
    df[col + "_dow"] = dt.dt.weekday
    df[col + "_hour"] = dt.dt.hour
    return df

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    dphi = p2 - p1
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if "transaction_time" in df.columns:
        df = parse_datetime(df, "transaction_time")

    if set(["lat","lon","merchant_lat","merchant_lon"]).issubset(df.columns):
        df["distance_km"] = haversine(df["lat"], df["lon"], df["merchant_lat"], df["merchant_lon"])
    else:
        df["distance_km"] = np.nan

    return df, [
        "amount","lat","lon","population_city","jobs","merchant_lat","merchant_lon","distance_km",
        "transaction_time_year","transaction_time_month","transaction_time_day","transaction_time_dow","transaction_time_hour",
        "merch","cat_id","name_1","name_2","gender","street","one_city","us_state","post_code"
    ]

def fit_and_save(train_path: str, model_dir: str, task: str = "classification") -> None:
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(train_path)

    assert "target" in df.columns, "В train.csv должна быть колонка 'target'"
    y = df["target"]
    df_feat, base_features = build_features(df.copy())

    idx = np.arange(len(df_feat))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    val_size = max(2000, int(0.2 * len(idx))) if len(idx) > 10000 else int(0.2*len(idx))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    num_cols = [
        "amount","lat","lon","population_city","jobs","merchant_lat","merchant_lon","distance_km",
        "transaction_time_year","transaction_time_month","transaction_time_day","transaction_time_dow","transaction_time_hour",
    ]
    cat_cols = ["merch","cat_id","name_1","name_2","gender","street","one_city","us_state","post_code"]

    for c in num_cols:
        if c not in df_feat.columns:
            df_feat[c] = np.nan
    for c in cat_cols:
        if c not in df_feat.columns:
            df_feat[c] = "missing"

    agg_merch = df_feat.loc[train_idx].groupby("merch")["amount"].mean().rename("merch_amount_mean")
    agg_cat   = df_feat.loc[train_idx].groupby("cat_id")["amount"].mean().rename("cat_amount_mean")

    df_feat = df_feat.merge(agg_merch, how="left", left_on="merch", right_index=True)
    df_feat = df_feat.merge(agg_cat,   how="left", left_on="cat_id", right_index=True)

    num_cols += ["merch_amount_mean","cat_amount_mean"]

    for c in num_cols:
        df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")
        med = df_feat.loc[train_idx, c].median()
        df_feat[c] = df_feat[c].fillna(med)

    for c in cat_cols:
        df_feat[c] = df_feat[c].astype("string").fillna("missing")

    X_train = df_feat.iloc[train_idx][num_cols + cat_cols]
    y_train = y.iloc[train_idx]
    X_val   = df_feat.iloc[val_idx][num_cols + cat_cols]
    y_val   = y.iloc[val_idx]

    cat_idx = list(range(len(num_cols), len(num_cols)+len(cat_cols)))

    if task == "regression":
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            od_type="Iter",
            od_wait=50,
            task_type="CPU",
            verbose=False
        )
    else:
        model = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.05,
            depth=8,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            od_type="Iter",
            od_wait=80,
            auto_class_weights="Balanced",
            task_type="CPU",
            verbose=False
        )

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_idx)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    if task == "regression":
        from sklearn.metrics import mean_squared_error
        rmse = mean_squared_error(y_val, model.predict(val_pool), squared=False)
        print(f"[VAL] RMSE: {rmse:.5f}")
    else:
        from sklearn.metrics import roc_auc_score
        p = model.predict_proba(val_pool)[:,1]
        auc = roc_auc_score(y_val, p)
        print(f"[VAL] AUC: {auc:.5f}")

    artefacts = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_medians": {c: float(df_feat.loc[train_idx, c].median()) for c in num_cols},
        "cat_fill": {c: "missing" for c in cat_cols},
        "agg_merch": agg_merch.to_dict(),
        "agg_cat": agg_cat.to_dict(),
        "task": task
    }

    bundle = {
        "model": model,
        "artefacts": artefacts
    }
    joblib.dump(bundle, os.path.join(model_dir, "model.pkl"))

    with open(os.path.join(model_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(num_cols + cat_cols, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="train.csv")
    ap.add_argument("--model_dir",  default="app/model")
    ap.add_argument("--task",       choices=["classification","regression"], default="classification")
    args = ap.parse_args()
    fit_and_save(args.train_path, args.model_dir, task=args.task)