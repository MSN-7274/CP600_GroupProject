import pandas as pd
import os
from datetime import datetime
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from logistic_regression import LogisticRegressionModel
from svm import SVMModel
from evaluate import evaluate_model

def resolve_dataset_path(dataset_arg: str | None = None) -> tuple[str, str]:
    """
    Resolve dataset path. If dataset_arg is provided, try to locate it in given path or in ../outputs, ../data.
    If not provided, prefer a sampled dataset in ../outputs (matching sampled_*.csv) else fall back to ../data/*.csv.

    Returns (resolved_path, data_source_name)
    data_source_name = "{parent_folder}_{base_name_without_ext}" (e.g., outputs_sampled_breast_cancer)
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outputs_dir = os.path.join(root, "outputs")
    data_dir = os.path.join(root, "data")

    # Helper to build data_source from a full path
    def _make_source_name(pth: str) -> str:
        parent = os.path.basename(os.path.dirname(pth))
        base = os.path.splitext(os.path.basename(pth))[0]
        return f"{parent}_{base}"

    # If a dataset explicitly provided
    if dataset_arg:
        # Try literal path first
        if os.path.isabs(dataset_arg) and os.path.isfile(dataset_arg):
            return dataset_arg, _make_source_name(dataset_arg)

        # try relative to current folder
        path = os.path.abspath(dataset_arg)
        if os.path.isfile(path):
            return path, _make_source_name(path)

        # try in outputs folder
        candidate = os.path.join(outputs_dir, dataset_arg)
        if os.path.isfile(candidate):
            return candidate, _make_source_name(candidate)

        # try data folder
        candidate = os.path.join(data_dir, dataset_arg)
        if os.path.isfile(candidate):
            return candidate, _make_source_name(candidate)

        # try with .csv extension
        if not dataset_arg.lower().endswith('.csv'):
            return resolve_dataset_path(dataset_arg + '.csv')

        raise FileNotFoundError(f"Dataset {dataset_arg} not found in path, outputs or data directories")

    # No dataset arg: prioritize outputs/sampled_*.csv
    # Search outputs for files starting with 'sampled_' first
    if os.path.isdir(outputs_dir):
        sampled_files = [f for f in os.listdir(outputs_dir) if f.lower().endswith('.csv') and f.startswith('sampled_')]
        if sampled_files:
            # pick the first
            chosen = os.path.join(outputs_dir, sampled_files[0])
            return chosen, _make_source_name(chosen)

        # fallback: any csv in outputs
        csv_files = [f for f in os.listdir(outputs_dir) if f.lower().endswith('.csv')]
        if csv_files:
            chosen = os.path.join(outputs_dir, csv_files[0])
            return chosen, _make_source_name(chosen)

    # fallback to data folder (prefer a file corresponding to sampled_* if present, else pick first CSV)
    if os.path.isdir(data_dir):
        # try to match a sampled file name in outputs -> remove 'sampled_' prefix and check for that file in data
        if os.path.isdir(outputs_dir):
            sampled_matches = [f for f in os.listdir(outputs_dir) if f.lower().endswith('.csv') and f.startswith('sampled_')]
            if sampled_matches:
                sampled_name = sampled_matches[0]
                original_name_candidate = sampled_name[len('sampled_'):]
                if original_name_candidate in os.listdir(data_dir):
                    chosen = os.path.join(data_dir, original_name_candidate)
                    return chosen, _make_source_name(chosen)
        csvs = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
        if csvs:
            chosen = os.path.join(data_dir, csvs[0])
            return chosen, _make_source_name(chosen)

    raise FileNotFoundError("No dataset found in ../outputs or ../data")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action='append', help='Dataset path or filename (can specify multiple; searches ../outputs and ../data if not an exact path)')
    parser.add_argument('-r', '--results-dir', type=str, default=None, help='Directory to write results; defaults to ./results')
    args = parser.parse_args()

    def evaluate_and_save(dataset_path: str, data_source: str, results_dir: str, timestamp: str) -> None:
        """
        Helper: train models on dataset_path and save metrics to results_dir/{data_source}_results_{timestamp}.txt
        """
        # Detect separator
        sep = ','
        if 'bank' in dataset_path:
            sep = ';' if 'sampled' not in dataset_path else ','
        df_local = pd.read_csv(dataset_path, sep=sep)
        
        # Convert numeric columns for original bank dataset
        if 'bank' in dataset_path and 'sampled' not in dataset_path:
            numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
            for col in numeric_cols:
                if col in df_local.columns:
                    df_local[col] = pd.to_numeric(df_local[col], errors='coerce')
        
        # Basic preprocessing
        # Detect target column and map to 0/1 or multi-class
        target_col = None
        if 'diagnosis' in df_local.columns:
            target_col = 'diagnosis'
            df_local[target_col] = df_local[target_col].map({'M': 1, 'B': 0})
        elif 'y' in df_local.columns:
            target_col = 'y'
            df_local[target_col] = df_local[target_col].map({'yes': 1, 'no': 0})
        elif 'Card_Category' in df_local.columns:
            target_col = 'Card_Category'
            df_local[target_col] = df_local[target_col].map({'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3})
        elif 'income' in df_local.columns:
            target_col = 'income'
            df_local[target_col] = df_local[target_col].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})

        # Select features (exclude id and target, and only numeric for simplicity)
        exclude_cols = ['id', 'CLIENTNUM']
        if target_col:
            exclude_cols.append(target_col)
        # Exclude pre-computed prediction columns that cause data leakage
        exclude_cols.extend([col for col in df_local.columns if col.startswith('Naive_Bayes_Classifier')])
        features_local = [col for col in df_local.columns if col not in exclude_cols and df_local[col].dtype in ['int64', 'float64']]
        X_local = df_local[features_local]
        y_local = df_local[target_col] if target_col else None

        # Handle missing values (now all numeric)
        imputer_local = SimpleImputer(strategy='mean')
        X_local = imputer_local.fit_transform(X_local)

        # Split data
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_local, y_local, test_size=0.2, random_state=42)

        # Scale features
        scaler_local = StandardScaler()
        X_train_scaled_l = scaler_local.fit_transform(X_train_l)
        X_test_scaled_l = scaler_local.transform(X_test_l)
        X_test_scaled_l = scaler_local.transform(X_test_l)

        # Logistic Regression
        lr_model = LogisticRegressionModel()
        lr_model.train(X_train_scaled_l, y_train_l)
        lr_pred = lr_model.predict(X_test_scaled_l)
        lr_metrics = evaluate_model(y_test_l, lr_pred)

        # SVM
        svm_model = SVMModel()
        svm_model.train(X_train_scaled_l, y_train_l)
        svm_pred = svm_model.predict(X_test_scaled_l)
        svm_metrics = evaluate_model(y_test_l, svm_pred)

        # Save results
        results_filename_local = f"{data_source}_results_{timestamp}.txt"
        os.makedirs(results_dir, exist_ok=True)
        results_path_local = os.path.join(results_dir, results_filename_local)
        with open(results_path_local, 'w') as f:
            f.write(f"Results for: {data_source}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("="*50 + "\n\n")
            f.write("Logistic Regression Metrics:\n")
            for key, value in lr_metrics.items():
                if key != 'confusion_matrix':
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}:\n{value}\n")
            f.write("\nSVM Metrics:\n")
            for key, value in svm_metrics.items():
                if key != 'confusion_matrix':
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}:\n{value}\n")
        print(f"\nResults saved to {results_path_local}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir if args.results_dir else os.path.join(os.path.dirname(__file__), 'results')

    if args.dataset:
        for ds in args.dataset:
            try:
                dataset_path, data_source = resolve_dataset_path(ds)
                evaluate_and_save(dataset_path, data_source, results_dir, timestamp)
            except FileNotFoundError as e:
                print(f"Error for {ds}: {e}")
        return

    # If dataset_path points to a file, we will evaluate it. If no dataset arg is given and both
    # a sampled file (in outputs) and original file (in data) exist, we will evaluate both.

    # If no dataset arg was provided: prefer outputs sample and also run original data if present.
    # determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        # default to a `results` folder inside the logistic_svm_project module
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
    # determine results_dir
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(os.path.dirname(__file__), 'results')

    # If both a sampled CSV in outputs and an original csv in data exist, evaluate both
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    outputs_dir = os.path.join(project_root, 'outputs')
    data_dir = os.path.join(project_root, 'data')

    # candidate sampled file we may have auto-selected previously
    sampled_path = None
    if os.path.isdir(outputs_dir):
        sampled_candidates = [f for f in os.listdir(outputs_dir) if f.lower().endswith('.csv') and f.startswith('sampled_')]
        if sampled_candidates:
            sampled_path = os.path.join(outputs_dir, sampled_candidates[0])

    # candidate original file in data
    original_path = None
    if os.path.isdir(data_dir):
        # match the base of sampled_*.csv to the original name in data (e.g., sampled_breast_cancer -> breast_cancer.csv)
        if sampled_path:
            sampled_name = os.path.basename(sampled_path)
            if sampled_name.startswith('sampled_'):
                original_name_candidate = sampled_name[len('sampled_'):]
                if original_name_candidate in os.listdir(data_dir):
                    original_path = os.path.join(data_dir, original_name_candidate)
        if not original_path:
            csvs = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
            if csvs:
                original_path = os.path.join(data_dir, csvs[0])

    # If we found both, evaluate both and return
    if sampled_path and original_path:
        evaluate_and_save(original_path, f"data_{os.path.splitext(os.path.basename(original_path))[0]}", results_dir, timestamp)
        evaluate_and_save(sampled_path, f"outputs_{os.path.splitext(os.path.basename(sampled_path))[0]}", results_dir, timestamp)
        return

    # otherwise fall back to previous behavior and evaluate the single dataset selected earlier
    if not sampled_path and original_path:
        evaluate_and_save(original_path, f"data_{os.path.splitext(os.path.basename(original_path))[0]}", results_dir, timestamp)
        return
    if sampled_path and not original_path:
        evaluate_and_save(sampled_path, f"outputs_{os.path.splitext(os.path.basename(sampled_path))[0]}", results_dir, timestamp)
        return

    # If we get here it means no data found in ../outputs nor ../data and we failed earlier. Nothing to do.

if __name__ == "__main__":
    main()