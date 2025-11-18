from typing import Tuple, List
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# preprocess non-sensitive features
def preprocess_X_ns(X_ns) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ns)
    return X_scaled, scaler


def run_kmeans(X_scaled: np.ndarray, sample_size: int, k: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    n = X_scaled.shape[0]
    print(f"[INFO] Running KMeans with k={k}, n={n} ...")

    rng = np.random.default_rng(random_state)

    # clustering
    km = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init="auto", # "auto" seems equal to 1, can modify to 3, no more than 10
        max_iter=100,
        algorithm="elkan"  # faster when using Euclidean distance than full
    )
    labels = km.fit_predict(X_scaled)
    centers = km.cluster_centers_

    # cluster statistics
    cluster_indices = [np.where(labels == j)[0] for j in range(k)]
    cluster_sizes = [len(idx) for idx in cluster_indices]

    # compute per-cluster quotas s_j (proportional to cluster size)
    cluster_sizes_arr = np.array(cluster_sizes, dtype=float)
    raw_quota = sample_size * cluster_sizes_arr / max(cluster_sizes_arr.sum(), 1.0)

    # floor quotas and distribute remaining by largest fractional part
    s_j = np.floor(raw_quota).astype(int)
    diff = sample_size - int(s_j.sum())  # diff >= 0 because we used floor

    if diff > 0:
        frac = raw_quota - s_j
        # indices of clusters sorted by fractional part descending
        order = np.argsort(-frac)
        for j_idx in order[:diff]:
            s_j[j_idx] += 1

    # make sure numerical reasons sum < sample_size
    if s_j.sum() > sample_size:
        over = int(s_j.sum() - sample_size)
        frac = raw_quota - np.floor(raw_quota)
        order = np.argsort(frac)  # ascending
        for j_idx in order[:over]:
            if s_j[j_idx] > 0:
                s_j[j_idx] -= 1

    assert s_j.sum() <= sample_size, "Internal quota error: sum(s_j) > sample_size"

    # sample within each cluster
    sampled_indices: List[int] = []

    for j, idx in enumerate(cluster_indices):
        if len(idx) == 0:
            continue

        quota = int(s_j[j])
        if quota <= 0:
            continue

        cluster_points = X_scaled[idx]

        # choose the point closest to centroid as representative
        center = centers[j]
        dists_to_center = np.linalg.norm(cluster_points - center, axis=1)
        medoid_pos = int(np.argmin(dists_to_center))
        medoid_idx = idx[medoid_pos]
        chosen = [medoid_idx]

        # fill the remaining quota with random points from this cluster (increase randomness)
        remaining_quota = quota - 1
        if remaining_quota > 0:
            candidates = [i for i in idx if i != medoid_idx]
            if len(candidates) > 0:
                num_to_pick = min(remaining_quota, len(candidates))
                extra = rng.choice(candidates, size=num_to_pick, replace=False)
                chosen.extend(extra.tolist())

        sampled_indices.extend(chosen)

    sampled_indices = np.array(sampled_indices, dtype=int)

    # debug: ensure |S| == sample_size and no duplicates
    # print(f"[DEBUG] Desired sample_size = {sample_size}")
    # print(f"[DEBUG] Sum of quotas s_j   = {int(s_j.sum())}")
    # print(f"[DEBUG] Raw sampled count   = {len(sampled_indices)}")

    # remove duplicates
    sampled_indices = np.unique(sampled_indices)
    unique_count = len(sampled_indices)
    # print(f"[DEBUG] Unique sampled count = {unique_count}")

    # too many, down-sample
    if unique_count > sample_size:
        sampled_indices = rng.choice(sampled_indices, size=sample_size, replace=False)
    # too few, randomly fill from remaining points
    elif unique_count < sample_size:
        all_indices = np.arange(n)
        remaining = np.setdiff1d(all_indices, sampled_indices, assume_unique=True)
        need = sample_size - unique_count
        if need > 0 and len(remaining) > 0:
            extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
            sampled_indices = np.concatenate([sampled_indices, extra])

    sampled_indices = np.array(sampled_indices, dtype=int)
    print(f"\n[INFO] Sampling done. Selected {len(sampled_indices)} samples.\n")

    return sampled_indices, labels, cluster_sizes
