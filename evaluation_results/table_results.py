import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib as mpl
import seaborn as sns
from glob import glob
import numpy as np

# --------------------------------------------------
# config 
# --------------------------------------------------
ROOT_DIR       = pathlib.Path("../results")
LABEL_FRACTIONS = [0.01, 0.05, 0.1, 0.5, 1.0]
RESULTS_FILE = "../results/all_results.csv"

MODEL_NAMES = [
    # Supervised models first
    "Supervised_CNN",
    "Supervised_TCN", 
    "Supervised_Transformer",

    # SSL models
    "TS2Vec",
    "SoftTS2Vec",
    "TSTCC",
    "SoftTSTCC",
    "SimCLR",
]

SUPERVISED_MODELS = [
    "Supervised_CNN",
    "Supervised_TCN",
    "Supervised_Transformer"
]

SSL_LINEAR_MODELS = [
    ("TS2Vec", "LinearClassifier"),
    ("SoftTS2Vec", "LinearClassifier"),
    ("TSTCC", "LinearClassifier"),
    ("SoftTSTCC", "LinearClassifier"),
    ("SimCLR", "LinearClassifier")
]

SSL_MLP_MODELS = [
    ("TS2Vec", "MLPClassifier"),
    ("SoftTS2Vec", "MLPClassifier"),
    ("TSTCC", "MLPClassifier"),
    ("SoftTSTCC", "MLPClassifier"),
    ("SimCLR", "MLPClassifier")
]

GROUPS = {
    "Supervised": SUPERVISED_MODELS,
    "Self-Supervised (Linear)": SSL_LINEAR_MODELS,
    "Self-Supervised (MLP)": SSL_MLP_MODELS
}

# --------------------------------------------
# Load or build results DataFrame
# --------------------------------------------
if os.path.exists(RESULTS_FILE):
    print(f"[Loading cached metrics from {RESULTS_FILE}]")
    data = pd.read_csv(RESULTS_FILE)
else:
    print("Parsing metrics and building dataframe...")
    records = []

    # Handle Supervised Models
    for model in SUPERVISED_MODELS:
        pattern = str(ROOT_DIR / model / "*_*") 
        matching_dirs = glob(pattern)
        for exp_path_str in matching_dirs:
            exp_path = pathlib.Path(exp_path_str)
            exp_name = exp_path.name

            # Parse classifier model and label fraction
            parts = exp_name.split("_")
            if len(parts) < 2:
                continue
            
            classifier_model = parts[0]
            try:
                label_fraction = float(parts[-1])
            except ValueError:
                continue

            csv_file = exp_path / "aggregated_metrics.csv"
            if not csv_file.exists():
                continue

            df = pd.read_csv(csv_file)

            # Normalize AUC metric name
            auc_name = 'test_auc_roc' if 'test_auc_roc' in df.metric.values else (
                'test_auroc' if 'test_auroc' in df.metric.values else None)
            if auc_name is None:
                continue

            def get_metric(metric):
                row = df[df.metric == metric]
                return (row["mean"].values[0], row["std"].values[0]) if not row.empty else (np.nan, np.nan)

            acc_mean, acc_std = get_metric("test_accuracy")
            auc_mean, auc_std = get_metric(auc_name)
            pr_mean, pr_std   = get_metric("test_pr_auc")
            f1_mean, f1_std   = get_metric("test_f1")

            records.append({
                "group": "Supervised",
                "model": model,
                "classifier_model": classifier_model,
                "label_fraction": label_fraction,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "pr_auc_mean": pr_mean,
                "pr_auc_std": pr_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std
            })
    
    # Handle SSL Models (both Linear and MLP)
    for ssl_model, classifier in SSL_LINEAR_MODELS + SSL_MLP_MODELS:
        pattern = str(ROOT_DIR / ssl_model / f"{classifier}_*")
        matching_dirs = glob(pattern)
        for exp_path_str in matching_dirs:
            exp_path = pathlib.Path(exp_path_str)
            exp_name = exp_path.name

            parts = exp_name.split("_")
            if len(parts) < 2:
                continue

            try:
                label_fraction = float(parts[-1])
            except ValueError:
                continue

            csv_file = exp_path / "aggregated_metrics.csv"
            if not csv_file.exists():
                continue
            
            df = pd.read_csv(csv_file)

            # Normalize AUC metric name
            auc_name = 'test_auc_roc' if 'test_auc_roc' in df.metric.values else (
                'test_auroc' if 'test_auroc' in df.metric.values else None)
            if auc_name is None:
                continue

            def get_metric(metric):
                row = df[df.metric == metric]
                return (row["mean"].values[0], row["std"].values[0]) if not row.empty else (np.nan, np.nan)

            acc_mean, acc_std = get_metric("test_accuracy")
            auc_mean, auc_std = get_metric(auc_name)
            pr_mean, pr_std   = get_metric("test_pr_auc")
            f1_mean, f1_std   = get_metric("test_f1")

            # Correct group assignment based on classifier type
            group = "Self-Supervised (Linear)" if classifier == "LinearClassifier" else "Self-Supervised (MLP)"
            
            records.append({
                "group": group,
                "model": ssl_model,
                "classifier_model": classifier,  
                "label_fraction": label_fraction,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "pr_auc_mean": pr_mean,
                "pr_auc_std": pr_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std
            })

    data = pd.DataFrame(records)
    # Add model type column for easier filtering
    data['model_type'] = data.apply(
        lambda x: 'Supervised' if x['group'] == 'Supervised' 
        else ('SSL-Linear' if x['classifier_model'] == 'LinearClassifier' 
              else 'SSL-MLP'), axis=1
    )
    data.to_csv(RESULTS_FILE, index=False)
    print(f"[info] Saved full results to {RESULTS_FILE}")