from __future__ import annotations
import os
import json
import yaml
import pandas as pd
from typing import Dict, Any

def read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def write_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    df.to_csv(path, index=index)

def write_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)