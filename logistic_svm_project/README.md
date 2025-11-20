# Logistic Regression and SVM Classification Project

This project implements and compares Logistic Regression and Support Vector Machine (SVM) models for classification tasks using Python and scikit-learn.

## Usage

Run the main script to train and evaluate both models:
Remember to specify the path to original dataset and sampled one.

python main.py -d ../data/breast_cancer.csv -d ../outputs/sampled_breast_cancer.csv

## If use with other datasets

if test with other datasets, you need to specify target column in the training code

For an example, you need to specify target column:
def evaluate*and_save(dataset_path: str, data_source: str, results_dir: str, timestamp: str) -> None:
"""
Helper: train models on dataset_path and save metrics to results_dir/{data_source}\_results*{timestamp}.txt
""" # Detect separator
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
