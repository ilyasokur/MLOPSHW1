import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from app.common.io_utils import read_config, read_parquet, write_csv

def load_model_bundle(model_dir: str):
    path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("artefacts", None)
    return obj, None

def load_feature_names(model_dir: str):
    feat_path = os.path.join(model_dir, "feature_names.json")
    if os.path.exists(feat_path):
        with open(feat_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def get_feature_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(imp))]
        return dict(zip(feature_names, map(float, imp)))

    try:
        get_fi = getattr(model, "get_feature_importance", None)
        if callable(get_fi):
            imp = get_fi()
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(len(imp))]
            return dict(zip(feature_names, map(float, imp)))
    except Exception:
        pass

    return {}

def save_density_plot(preds: np.ndarray, path: str):
    plt.figure()
    plt.hist(preds, bins=50, density=True)
    plt.xlabel("prediction")
    plt.ylabel("density")
    plt.title("Prediction Density")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_config(args.config)
    X_path = os.path.join(args.work_dir, cfg["paths"]["preprocessed_features_parquet"])
    preds_path = os.path.join(args.work_dir, cfg["paths"]["predictions_csv"])

    X = read_parquet(X_path)

    model, artefacts = load_model_bundle(args.model_dir)

    feature_names = load_feature_names(args.model_dir)
    if feature_names is not None:
        cols_for_model = [c for c in feature_names if c in X.columns]
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing features for model: {missing}")
        X = X[cols_for_model]

    task_type = cfg["model"].get("task_type", "classification")
    use_proba = cfg["model"].get("use_predict_proba", True)
    proba_idx = cfg["model"].get("proba_class_index", 1)

    if task_type == "classification" and use_proba and hasattr(model, "predict_proba"):
        preds = model.predict_proba(X)[:, proba_idx]
    else:
        preds = model.predict(X)

    df_preds = pd.DataFrame({"prediction": preds})
    write_csv(df_preds, preds_path, index=False)

    fi = get_feature_importances(model, list(X.columns))
    if fi:
        top5 = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True)[:5])
        with open(os.path.join(args.work_dir, "feature_importances_top5.json"), "w", encoding="utf-8") as f:
            json.dump(top5, f, ensure_ascii=False, indent=2)

    save_density_plot(np.asarray(preds), os.path.join(args.work_dir, "prediction_density.png"))

    print(f"Scored: {len(preds)} predictions")

if __name__ == "__main__":
    main()