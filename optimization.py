# optimization.py
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# get mean & std for numeric columns
def compute_numeric_stats(X: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for c in numeric_cols:
        col = X[c].astype(float)
        stats[c] = (col.mean(), col.std(ddof=0))
    return stats

# compute normalized frequency for each categorical column
def compute_categorical_freq(X: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, pd.Series]:
    freqs = {}
    for c in categorical_cols:
        value_counts = X[c].value_counts(normalize=True)
        freqs[c] = value_counts
    return freqs

# comparing the distribution differences between D and S
def distribution_diff(df_D: pd.DataFrame, df_S: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> float:
    diff = 0.0

    # numeric: sum of |mean_D - mean_S| + |std_D - std_S|
    stats_D = compute_numeric_stats(df_D, numeric_cols)
    stats_S = compute_numeric_stats(df_S, numeric_cols)

    for c in numeric_cols:
        mu_D, sigma_D = stats_D[c]
        mu_S, sigma_S = stats_S[c]
        diff += abs(mu_D - mu_S)
        diff += abs(sigma_D - sigma_S)

    # categorical: sum of L1 distance between freq vectors
    freq_D = compute_categorical_freq(df_D, categorical_cols)
    freq_S = compute_categorical_freq(df_S, categorical_cols)

    for c in categorical_cols:
        fD = freq_D[c]
        fS = freq_S.get(c, pd.Series(dtype=float))

        # align categories
        all_cats = fD.index.union(fS.index)
        fD_aligned = fD.reindex(all_cats, fill_value=0.0)
        fS_aligned = fS.reindex(all_cats, fill_value=0.0)

        diff += float(np.abs(fD_aligned - fS_aligned).sum())

    return diff

# local search to refine S by swapping points within the same cluster label, ami to reduce diff(D,S)
def local_swap(
    df_D: pd.DataFrame,
    S_indices: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    labels: np.ndarray,
    max_iterations: int = 500,
    early_stop_rounds: int = 100,
    random_state: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(random_state)

    current_indices = S_indices.copy()
    best_indices = current_indices.copy()

    df_S = df_D.iloc[current_indices]
    best_diff = distribution_diff(df_D, df_S, numeric_cols, categorical_cols)
    print(f"[INFO] Initial distribution diff: {best_diff:.6f}")

    no_improve_rounds = 0
    n = df_D.shape[0]

    # Pre-group indices by cluster label for swap
    unique_labels = np.unique(labels)
    cluster_to_indices = {lab: np.where(labels == lab)[0] for lab in unique_labels}

    for it in range(max_iterations):
        # choose a random cluster
        lab = int(rng.choice(unique_labels))
        cluster_idx = cluster_to_indices[lab]
        if len(cluster_idx) <= 1: continue

        # candidates in S and outside S within this cluster
        in_S = np.intersect1d(cluster_idx, current_indices, assume_unique=False)
        out_S = np.setdiff1d(cluster_idx, current_indices, assume_unique=False)

        if len(in_S) == 0 or len(out_S) == 0: continue

        x_out = int(rng.choice(in_S))
        x_in = int(rng.choice(out_S))

        new_indices = current_indices.copy()
        # replace x_out by x_in
        out_pos = np.where(new_indices == x_out)[0]
        if len(out_pos) == 0: continue
        new_indices[out_pos[0]] = x_in

        df_S_new = df_D.iloc[new_indices]
        new_diff = distribution_diff(df_D, df_S_new, numeric_cols, categorical_cols)

        if new_diff + 1e-12 < best_diff:
            best_diff = new_diff
            current_indices = new_indices
            best_indices = new_indices.copy()
            df_S = df_S_new
            no_improve_rounds = 0
            print(f"[DEBUG] iter={it:4d}  improved diff={best_diff:.6f}")
        else:
            no_improve_rounds += 1

        if no_improve_rounds >= early_stop_rounds:
            print(f"[INFO] Early stopping at iter={it}, no improvement for {early_stop_rounds} rounds.")
            break

    print(f"[INFO] Final distribution diff: {best_diff:.6f}")
    return best_indices
