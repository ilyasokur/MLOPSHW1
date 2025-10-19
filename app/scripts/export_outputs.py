import argparse
import os
import json
import pandas as pd
from app.common.io_utils import read_config, read_parquet, write_csv, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_config(args.config)
    ensure_dir(args.output_dir)

    test_parquet = os.path.join(args.work_dir, cfg["paths"]["intermediate_test_parquet"])
    preds_csv    = os.path.join(args.work_dir, cfg["paths"]["predictions_csv"])

    raw_test = read_parquet(test_parquet)
    preds_df = pd.read_csv(preds_csv)

    id_col      = cfg["kaggle_submission"]["id_column"]
    target_col  = cfg["kaggle_submission"]["submission_target_column"]
    sub_cols    = cfg["kaggle_submission"]["submission_columns"]
    thr         = cfg["kaggle_submission"].get("binarize_threshold", None)

    n = len(preds_df)
    if len(raw_test) != n:
        print(f"[warn] test rows = {len(raw_test)} != preds rows = {n}. "
              f"Submission will use range index 0..{n-1}")
    y = preds_df["prediction"].values
    if thr is not None:
        y = (y >= float(thr)).astype(int)

    sub = pd.DataFrame({
        id_col: range(n),
        target_col: y
    })
    sub = sub[sub_cols]

    sample_path = os.path.join(args.output_dir, "sample_submission.csv")
    write_csv(sub, sample_path, index=False)

    fi_src = os.path.join(args.work_dir, "feature_importances_top5.json")
    if os.path.exists(fi_src):
        with open(fi_src, "r", encoding="utf-8") as f:
            fi_obj = json.load(f)
        with open(os.path.join(args.output_dir, "feature_importances_top5.json"), "w", encoding="utf-8") as f:
            json.dump(fi_obj, f, ensure_ascii=False, indent=2)

    dens_src = os.path.join(args.work_dir, "prediction_density.png")
    if os.path.exists(dens_src):
        with open(dens_src, "rb") as fsrc, open(os.path.join(args.output_dir, "prediction_density.png"), "wb") as fdst:
            fdst.write(fsrc.read())

    print(f"Exported: {sample_path}")

if __name__ == "__main__":
    main()