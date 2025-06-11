"""
Compare F1 scores of various model groups across label fractions.

Performs Welch’s t-test and checks for normality using Shapiro–Wilk.
Expected folder structure:
└── /Users/carlitos/Desktop/ECG-Project/results/
    ├── Supervised_CNN/Supervised_0.1/individual_runs.csv
    ├── TS2Vec/LinearClassifier_0.1/individual_runs.csv
    ├── TS2Vec/MLPClassifier_0.1/individual_runs.csv
    └── ...

"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

# ---------- USER CONFIG ----------
ROOT = Path("/Users/carlitos/Desktop/ECG-Project/results")
NORM_TEST_ALPHA = 0.05
# ----------------------------------

SSL_MODELS_ALL = {
    "TS2Vec", 
    "SoftTS2Vec", 
    "TSTCC", 
    "SoftTSTCC", 
    "SimCLR"
    }

SUPERVISED_MODELS_ALL = [
    "Supervised_CNN", 
    "Supervised_TCN", 
    "Supervised_Transformer"
    ]

SSL_MODELS = {
    "TS2Vec", 
    "SoftTS2Vec", 
    "TSTCC", 
    "SoftTSTCC", 
    }

SUPERVISED_MODELS = [
    "Supervised_CNN", 
    "Supervised_TCN", 
    ]

TRANSFORMER_MODEL = "Supervised_Transformer"


# Load F1 scores for individual supervised models
def get_supervised_scores(root: Path):
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    results = defaultdict(lambda: defaultdict(list))

    for model_dir in root.iterdir():
        if not model_dir.is_dir() or model_dir.name not in SUPERVISED_MODELS_ALL:
            continue
        model_name = model_dir.name.replace("Supervised_", "")

        for sub in model_dir.iterdir():
            if not sub.is_dir():
                continue
            m = pattern_label.match(sub.name)
            if not m:
                continue
            frac = float(m.group(1))

            csv_path = sub / "individual_runs.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            results[frac][model_name].extend(df["test_f1"].values)

    return results

# Load scores for individual SSL models
def get_ssl_model_scores(root: Path):
    """Load individual SSL model results."""
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    results = defaultdict(lambda: defaultdict(list))  # Changed structure

    for model_dir in root.iterdir():
        if not model_dir.is_dir() or model_dir.name not in SSL_MODELS_ALL:
            continue
        model_name = model_dir.name

        for sub in model_dir.iterdir():
            if not sub.is_dir():
                continue
            m = pattern_label.match(sub.name)
            if not m:
                continue
            frac = float(m.group(1))
            clf_type = sub.name.split("_")[0]
            
            # Create combined model name with classifier type
            model_clf = f"{model_name}_{clf_type}"
            
            csv_path = sub / "individual_runs.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            results[frac][model_clf].extend(df["test_f1"].values)

    return results

# Load f1 scores for the curated supervised and ssl models
def get_supervised_and_ssl_scores(root: Path):
    pattern_label = re.compile(r".*_(0\.\d+|1\.0)$")
    data = defaultdict(lambda: defaultdict(list))

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        model_type = model_dir.name

        if model_type == TRANSFORMER_MODEL:
            group = "Transformer"
        elif model_type.startswith("Supervised"):
            if model_type not in SUPERVISED_MODELS:
                continue
            group = "Supervised"
        elif model_type in SSL_MODELS:
            group = "SSL"
        else:
            continue

        for sub in model_dir.iterdir():
            if not sub.is_dir():
                continue
            m = pattern_label.match(sub.name)
            if not m:
                continue
            label_frac = float(m.group(1))
            clf_type = sub.name.split("_")[0]

            if group == "Transformer":
                cat = "Transformer"
            elif group == "Supervised":
                cat = "Supervised"
            elif clf_type == "LinearClassifier":
                cat = "SSL_Linear"
            elif clf_type == "MLPClassifier":
                cat = "SSL_MLP"
            else:
                continue

            csv_path = sub / "individual_runs.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            for f1 in df["test_f1"].values:
                data[label_frac][cat].append(f1)
            
            # # Check for both possible AUC metric names
            # if "test_auc_roc" in df.columns:
            #     auc_values = df["test_auc_roc"].values
            # elif "test_auroc" in df.columns:
            #     auc_values = df["test_auroc"].values
            # else:
            #     continue  # Skip if neither metric is found
                
            # for auc in auc_values:
            #     data[label_frac][cat].append(auc)

    return data

# Helper functions
def cliffs_delta(a, b):
    """Compute Cliff's Delta for two independent samples."""
    n1, n2 = len(a), len(b)
    more = sum(x > y for x in a for y in b)
    less = sum(x < y for x in a for y in b)
    delta = (more - less) / (n1 * n2)
    return delta

def cliffs_magnitude(d):
    d_abs = abs(d)
    if d_abs < 0.147:
        return "negligible"
    elif d_abs < 0.33:
        return "small"
    elif d_abs < 0.474:
        return "medium"
    else:
        return "large"

def mannwhitney_comparison(results, group1, group2):
    print(f"\nMann–Whitney U test with Cliff's Delta: {group1} vs {group2}")
    print("-" * 95)
    print(f"{'Frac':>5} | N_1  N_2 | U-stat   p-value   sig | Cliff’s d   Magnitude")
    print("-" * 95)
    for frac in sorted(results):
        v1 = results[frac].get(group1, [])
        v2 = results[frac].get(group2, [])
        if len(v1) < 2 or len(v2) < 2:
            continue
        u_stat, p_val = mannwhitneyu(v1, v2, alternative='two-sided')
        sig = "*" if p_val < 0.05 else ""
        d = cliffs_delta(v1, v2)
        mag = cliffs_magnitude(d)
        print(f"{frac:5.2f} | {len(v1):^4} {len(v2):^4} | {u_stat:7.3f}  {p_val:8.4f}  {sig} | {d:10.3f}  {mag:>9}")


"""Run all statistical comparisons."""
# Individual supervised model comparisons
print("\n=== Individual Supervised Model Comparisons ===")
sup_results = get_supervised_scores(ROOT)
sup_models = list(set([m for frac in sup_results.keys() for m in sup_results[frac].keys()]))
for m1, m2 in combinations(sup_models, 2):
    mannwhitney_comparison(sup_results, m1, m2)

# Individual SSL model comparisons
print("\n=== Individual SSL Model Comparisons ===")
ssl_results = get_ssl_model_scores(ROOT)
# Group models by classifier type
linear_models = []
mlp_models = []
for frac in ssl_results.keys():
    for model in ssl_results[frac].keys():
        if "LinearClassifier" in model:
            linear_models.append(model)
        elif "MLPClassifier" in model:
            mlp_models.append(model)
# Remove duplicates and sort
linear_models = sorted(list(set(linear_models)))
mlp_models = sorted(list(set(mlp_models)))
# Compare Linear models
print("\n--- Linear Classifier Models ---")
for m1, m2 in combinations(linear_models, 2):
    mannwhitney_comparison(ssl_results, m1, m2)
# Compare MLP models
print("\n--- MLP Classifier Models ---")
for m1, m2 in combinations(mlp_models, 2):
    mannwhitney_comparison(ssl_results, m1, m2)

# Group comparisons
print("\n=== Group Comparisons (F1 scores) ===")
all_results = get_supervised_and_ssl_scores(ROOT)
mannwhitney_comparison(all_results, "Supervised", "SSL_Linear")
mannwhitney_comparison(all_results, "SSL_Linear", "SSL_MLP")
mannwhitney_comparison(all_results, "Supervised", "SSL_MLP")

print("\n=== Transformer vs Other Groups ===")
mannwhitney_comparison(all_results, "Transformer", "Supervised")
mannwhitney_comparison(all_results, "Transformer", "SSL_Linear")
mannwhitney_comparison(all_results, "Transformer", "SSL_MLP")