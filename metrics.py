from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# calculate the similarity between datasets S and D; the closer the value is to 0, the greater the correlation.
def evaluate_dataset_separability(X_ns: pd.DataFrame, S_indices: np.ndarray) -> Dict:

    n = X_ns.shape[0]
    source = np.zeros(n, dtype=int)
    source[S_indices] = 1

    # one-hot encode categoricals
    X_enc = pd.get_dummies(X_ns, drop_first=False)
    X_array = X_enc.values

    ds = mutual_info_classif(X_array, source, discrete_features=False, random_state=0)
    ds_mean = float(np.mean(ds))
    ds_max = float(np.max(ds))

    return {
        "ds_mean": ds_mean,
        "ds_max": ds_max,
    }


def print_summary(n_total: int, S_indices: np.ndarray, k: int):
    m = len(S_indices)
    print("\n=== Sampling Summary ===")
    print(f"n_total         = {n_total}")
    print(f"n_sampled       = {m} ({m / n_total * 100:.2f}%)")
    print(f"num of k        = {k}")

