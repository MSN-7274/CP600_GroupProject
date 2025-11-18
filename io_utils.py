import json
import os
from typing import Tuple, List, Dict
import pandas as pd


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def ensure_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

# load csv
def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    return df

# extract non-sensitive features
def split_features(df: pd.DataFrame,data_cfg: Dict) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    sensitive_cols = data_cfg.get("sensitive_columns", [])
    target_col = data_cfg.get("target_column", None)
    ns_cols = data_cfg.get("non_sensitive_columns", [])

    all_cols = df.columns.tolist()

    if not ns_cols:
        # use all columns except sensitive + target as non-sensitive
        exclude = set(sensitive_cols)
        if target_col and target_col in all_cols:
            exclude.add(target_col)
        ns_cols = [c for c in all_cols if c not in exclude]

    # target
    if target_col and target_col in df.columns:
        y = df[target_col]
    else:
        y = None

    X_ns = df[ns_cols].copy()

    # identify categorical columns (simple heuristic)
    categorical_cols = [
        c for c in ns_cols
        if X_ns[c].dtype == "object" or str(X_ns[c].dtype).startswith("category")
    ]

    return X_ns, y, ns_cols, categorical_cols


def save_sample(df_sample: pd.DataFrame, output_cfg: Dict):
    output_dir = output_cfg.get("output_dir", "outputs")
    ensure_output_dir(output_dir)
    filename = output_cfg.get("sample_filename", "sampled_S.csv")
    out_path = os.path.join(output_dir, filename)
    df_sample.to_csv(out_path, index=False)
    print(f"[INFO] Sampled dataset S saved to: {out_path}")
