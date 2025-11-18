import os
import numpy as np
import pandas as pd
from io_utils import load_config, load_dataset, split_features, save_sample, ensure_output_dir
from auto_k import auto_select_k_ch_db
from clustering import preprocess_X_ns, run_kmeans
from optimization import local_swap, distribution_diff
from metrics import evaluate_dataset_separability, print_summary
import argparse


def main(config_path: str = "config.json"):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    sampling_cfg = cfg["sampling"]
    opt_cfg = cfg["optimization"]
    metrics_cfg = cfg["metrics"] # remember to delete it if it's useless.
    output_cfg = cfg["output"]
    random_state = data_cfg.get("random_state", 42)

    # load data and split features
    df = load_dataset(data_cfg["dataset_path"])
    X_ns, y, ns_cols, categorical_cols = split_features(df, data_cfg)

    print(f"[INFO] Loaded dataset with shape {df.shape}")
    print(f"[INFO] Non-sensitive columns: {ns_cols}")
    print(f"[INFO] Categorical non-sensitive columns: {categorical_cols}")

    # identify numeric non-sensitive columns (for later use in comparative distribution)
    numeric_cols = [ c for c in ns_cols if c not in categorical_cols]

    # one-hot (for clustering)
    X_ns_for_clustering = pd.get_dummies(X_ns, drop_first=False)
    X_scaled, scaler = preprocess_X_ns(X_ns_for_clustering)

    n_total = X_ns.shape[0]

    # 2. decide sample size m
    sample_ratio = sampling_cfg.get("sample_ratio", None)
    sample_size = sampling_cfg.get("sample_size", None)
    if sample_size is None:
        if sample_ratio is None:
            raise ValueError("Either 'sample_size' or 'sample_ratio' must be set in config.")
        sample_size = int(round(n_total * float(sample_ratio)))
    print(f"[INFO] Sample size (m) = {sample_size}")

    # find suitable k (for kmeans)
    k = sampling_cfg.get("k", None)
    if k is None:
        k_min = sampling_cfg.get("k_min", 8)
        k_max = sampling_cfg.get("k_max", 32)
        k_step = sampling_cfg.get("k_step", 4)
        k = auto_select_k_ch_db(
            X_scaled,
            k_min=k_min,
            k_max=k_max,
            k_step=k_step,
            random_state=random_state
        )
    # print(f"[DEBUG] Using k = {k} clusters.")

    # KMeans + cluster-proportional sampling
    S_indices, labels, cluster_sizes = run_kmeans(X_scaled, sample_size=sample_size, k=k, random_state=random_state)

    # refinement
    if opt_cfg.get("enable_refine", False):
        max_it = opt_cfg.get("max_iterations", 500)
        early_stop = opt_cfg.get("early_stop_rounds", 100)
        print("[INFO] Starting local refinement of S ...")
        S_indices = local_swap(
            df_D=X_ns,   # non-sensitive features
            S_indices=S_indices,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            labels=labels,
            max_iterations=max_it,
            early_stop_rounds=early_stop,
            random_state=random_state
        )
    else:
        print("[INFO] Refinement disabled.")

    # metrics
    print_summary(n_total, S_indices, k)

    df_S = df.iloc[S_indices].copy()
    overall_diff = distribution_diff(
        df_D=X_ns,
        df_S=X_ns.iloc[S_indices],
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols
    )
    print("=== Sampling Metrics ===")
    print(f"overall_diff (mean/std + freq) = {overall_diff:.6f}")

    ds_info = evaluate_dataset_separability(X_ns, S_indices)
    # print("\n=== Dataset Separability (D & S) ===")
    print(f"ds_mean                       = {ds_info['ds_mean']:.6f}")
    print(f"ds_max                        = {ds_info['ds_max']:.6f}")

    # write dataset S to csv
    save_sample(df_S, output_cfg)

    print("\n[INFO] Sampling Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privacy-preserving sampling experiment")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.json",
        help="Path to the JSON config file (default: config.json)"
    )
    args = parser.parse_args()

    # call main() with the given config path
    main(config_path=args.config)
