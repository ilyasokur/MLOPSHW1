import argparse
import os
from app.common.io_utils import read_config, read_csv, write_parquet, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_config(args.config)
    raw_name = cfg["paths"]["raw_test_filename"]
    inter_name = cfg["paths"]["intermediate_test_parquet"]

    in_path = os.path.join(args.input_dir, raw_name)
    out_path = os.path.join(args.work_dir, inter_name)

    ensure_dir(args.work_dir)

    df = read_csv(in_path)
    write_parquet(df, out_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols")

if __name__ == "__main__":
    main()