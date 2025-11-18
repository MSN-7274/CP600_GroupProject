from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# subsample rows of X
def subsample_X(X: np.ndarray, max_samples: int, random_state: int = 42) -> np.ndarray:
    n = X.shape[0]
    if n <= max_samples:
        return X
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    return X[idx]

# calculate the CH and DB values obtained after clustering with kmeans at a given K value
def evaluate_k_with_ch_db(X: np.ndarray, k: int, random_state: int = 42, n_init: int = 10) -> Tuple[float, float]:
    km = KMeans(n_clusters=k,random_state=random_state,n_init=n_init)
    labels = km.fit_predict(X)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return ch, db

# find suitabel k with CH and DB
def auto_select_k_ch_db(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    k_step: int,
    random_state: int = 42,
    max_subsample: int = 10000
) -> int:
    X_sub = subsample_X(X, max_subsample, random_state)
    ks = list(range(k_min, k_max + 1, k_step))

    ch_scores = []
    db_scores = []

    print("[INFO] CH and DB corresponding to different k values:")
    for k in ks:
        ch, db = evaluate_k_with_ch_db(X_sub, k, random_state)
        ch_scores.append(ch)
        db_scores.append(db)
        print(f"  k={k:3d}  CH={ch:.2f}  DB={db:.2f}")

    ch_arr = np.array(ch_scores, dtype=float)
    db_arr = np.array(db_scores, dtype=float)

    # normalize
    def norm(x: np.ndarray, reverse: bool = False) -> np.ndarray:
        x_min, x_max = x.min(), x.max()
        if x_max == x_min:
            return np.ones_like(x)
        res = (x - x_min) / (x_max - x_min)
        if reverse:
            res = 1.0 - res
        return res

    ch_norm = norm(ch_arr)           # higher is better
    db_norm = norm(db_arr, True)     # lower DB -> higher score

    combined = ch_norm + db_norm     # simple combination
    best_idx = int(np.argmax(combined))
    best_k = ks[best_idx]

    # print("[DEBUG] Auto-k selection finished.")
    # for i, k in enumerate(ks):
    #     print(f"  k={k:3d}  combined_score={combined[i]:.3f}")
    print(f"[INFO] Selected k = {best_k}")

    return best_k
