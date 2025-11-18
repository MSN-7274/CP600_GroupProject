1. Run command
   1. if config.json and main.py are in the same directory
   > python main.py
   2. If config.json and main.py are in different directories or you want to specify a .json file
   > python main.py --config <file path>
2. Config arameters
   1. **sample_size**: Sample size after sampling. Can be “null”. If non-null, this parameter takes precedence; otherwise, the sample size is calculated based on “sample_ratio”.
   2. **k_step**: Step size when automatically selecting k
   3. **bins_per_feature**: The number of bins used when creating histograms/frequency comparisons for numerical features in distribution_diff. Higher values provide finer granularity but result in slightly slower computation; 10 is a common compromise.
3. Algorithm brief intro
   1. Sampling
      1. Randomly sample a small subset from the entire dataset. Run K-means clustering on each candidate value of k (e.g., 8–32), calculating the Calinski–Harabasz index (preferably higher) and Davies–Bouldin index (preferably lower) for each k. Select the k value that yields the most optimal cluster structure by combining these two metrics.
      2. Using the previously selected k, run K-means once on all non-sensitive features (numeric values + encoded categories). This assigns each sample to a cluster while simultaneously obtaining the centroid (center vector) for each cluster.
      3. Count the number of samples in each cluster. Divide the target sample size m proportionally to cluster sizes into a set of integer quotas s_j (larger clusters receive larger quotas). Round and adjust these quotas to ensure the sum of all cluster quotas exactly equals m.
      4. For each cluster: First, select the sample closest to the cluster's centroid as the “representative sample.” Then, randomly draw several samples from the remaining samples in that cluster to fill the quota s_j. Combine the sampling results from all clusters, remove duplicates, and if the number of samples exceeds m, perform random pruning. If it falls below m, randomly supplement from the remaining unselected samples to obtain the initial subset S.
   2. Optimization
      1. Calculate the mean and standard deviation of numerical features D and S separately, along with the frequency of occurrence for each value of categorical features. Then construct an overall distribution difference metric (overall diff) using the weighted sum of the differences between these statistics, serving as the “target to minimize during optimization.”
      2. In each iteration, randomly select one sample from S, then pick one candidate sample from D \ S (the unselected portion) to attempt replacing the current sample. If the recalculated distribution difference metric after replacement is smaller than before, accept the replacement; otherwise, discard the attempt. Repeat this process for a specified maximum number of iterations or terminate early if no improvement occurs over consecutive iterations.
      3. After optimization, treat “whether the sample comes from D or S” as a binary classification label. Based on the features and this label, estimate a mutual information-based “separability” metric: a smaller value (closer to 0) indicates that it is difficult to determine which dataset the sample belongs to based solely on the features. This suggests that S's distribution is closer to D, making it more suitable as a reduced version of D for model training.